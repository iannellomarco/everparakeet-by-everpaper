import CoreML
import Foundation

/// The Core ML encoder backend (macOS 15+, iOS 18+): the `.mlpackage` is compiled once into a cache folder, then one
/// `MLModel` per window function is loaded (asynchronously, one after the other, so a caller can cancel between them).
public struct CoreMLEncoderFactory: EncoderFactory {
    public var cache: CompiledModelCache
    public var computeUnits: MLComputeUnits
    /// Windows that are skipped, instead of failing the load, when the package has no such function (the Core ML
    /// packages of Standard 1.1 carry main64, main192 and main384; main128 is only in the Core AI asset).
    public var optionalWindows: Set<Int>
    var compile: CompiledModelCache.Compile

    public init(cache: CompiledModelCache = CompiledModelCache(), computeUnits: MLComputeUnits = .cpuAndNeuralEngine,
                optionalWindows: Set<Int> = []) {
        self.cache = cache; self.computeUnits = computeUnits; self.optionalWindows = optionalWindows
        self.compile = CompiledModelCache.defaultCompile
    }

    public func makeEncoder(root: URL, config: EverParakeetConfig, windows: [Int], progress: LoadProgressHandler?) async throws -> EncoderBackend {
        let package = root.appendingPathComponent("models/encoder.mlpackage")
        guard FileManager.default.fileExists(atPath: package.path) else { throw EverParakeetError.missing("models/encoder.mlpackage") }
        let compiled = try await cache.resolve(packageURL: package, compile: compile, onCompile: { progress?(.compiling) })
        var models: [Int: MLModel] = [:]
        for (index, w) in windows.enumerated() {
            try Task.checkCancellation()
            progress?(.loadingFunction(name: "main\(w)", index: index, of: windows.count))
            let configuration = MLModelConfiguration()
            configuration.computeUnits = computeUnits
            configuration.functionName = "main\(w)"
            do {
                models[w] = try await MLModel.load(contentsOf: compiled, configuration: configuration)
            } catch {
                if optionalWindows.contains(w) { continue }
                throw error
            }
        }
        guard !models.isEmpty else { throw EverParakeetError.missing("encoder functions") }
        return CoreMLEncoder(models: models, hidden: config.hidden, nMels: config.nMels)
    }
}

final class CoreMLEncoder: EncoderBackend, @unchecked Sendable {
    /// One lane's input arrays for one window.
    final class Buffers {
        let feats: MLMultiArray, m1: MLMultiArray, m2: MLMultiArray, m3: MLMultiArray
        init(window w: Int, nMels: Int) throws {
            feats = try CoreMLEncoder.array([1, 1, nMels, 8 * w])
            m1 = try CoreMLEncoder.array([1, 1, 1, 4 * w])
            m2 = try CoreMLEncoder.array([1, 1, 1, 2 * w])
            m3 = try CoreMLEncoder.array([1, 1, 1, w])
        }
    }
    final class Inputs: EncoderInputs {
        var windows: [Int: Buffers] = [:]
    }

    private let models: [Int: MLModel]
    let windows: [Int]
    let hidden: Int
    let nMels: Int

    init(models: [Int: MLModel], hidden: Int, nMels: Int) {
        self.models = models
        self.windows = models.keys.sorted()
        self.hidden = hidden
        self.nMels = nMels
    }

    func makeInputs() throws -> EncoderInputs {
        let inputs = Inputs()
        for w in windows { inputs.windows[w] = try Buffers(window: w, nMels: nMels) }
        return inputs
    }

    static func array(_ shape: [Int]) throws -> MLMultiArray {
        try MLMultiArray(shape: shape.map { NSNumber(value: $0) }, dataType: .float16)
    }

    /// Writes a row-major source into the trailing two dimensions of an MLMultiArray.
    private static func write(_ array: MLMultiArray, rows: Int, cols: Int,
                             _ fill: (Int, UnsafeMutablePointer<Float16>) -> Void) {
        let shape = array.shape.map(\.intValue)
        let strides = array.strides.map(\.intValue)
        let rank = shape.count
        let rowStride = strides[rank - 2]
        let colStride = strides[rank - 1]
        let pointer = array.dataPointer.bindMemory(to: Float16.self, capacity: array.count)
        if colStride == 1 {
            for row in 0..<rows { fill(row, pointer + row * rowStride) }
        } else {
            var row = [Float16](repeating: 0, count: cols)
            for r in 0..<rows {
                row.withUnsafeMutableBufferPointer { fill(r, $0.baseAddress!) }
                for c in 0..<cols { pointer[r * rowStride + c * colStride] = row[c] }
            }
        }
    }

    private static func flatten(_ array: MLMultiArray) throws -> [Float] {
        let shape = array.shape.map(\.intValue)
        let strides = array.strides.map(\.intValue)
        let total = shape.reduce(1, *)
        var result = [Float](repeating: 0, count: total)
        switch array.dataType {
        case .float16:
            let p = array.dataPointer.bindMemory(to: Float16.self, capacity: total)
            for i in 0..<total {
                var remaining = i, offset = 0
                for d in stride(from: shape.count - 1, through: 0, by: -1) {
                    let index = remaining % shape[d]
                    remaining /= shape[d]
                    offset += index * strides[d]
                }
                result[i] = Float(p[offset])
            }
        case .float32:
            let p = array.dataPointer.bindMemory(to: Float.self, capacity: total)
            for i in 0..<total {
                var remaining = i, offset = 0
                for d in stride(from: shape.count - 1, through: 0, by: -1) {
                    let index = remaining % shape[d]
                    remaining /= shape[d]
                    offset += index * strides[d]
                }
                result[i] = p[offset]
            }
        default:
            throw EverParakeetError.invalid("Core ML encoder output is not floating point")
        }
        return result
    }

    func encode(mel: [Float], frames: Int, valid: Int, window w: Int, inputs: EncoderInputs) async throws -> (enc: [Float], validFrames: Int) {
        guard let model = models[w] else { throw EverParakeetError.missing("Core ML encoder window \(w)") }
        guard let buffers = (inputs as? Inputs)?.windows[w] else { throw EverParakeetError.invalid("encoder inputs of another backend, or window \(w) missing") }
        let lengths = maskLengths(valid)
        let capacity = 8 * w
        mel.withUnsafeBufferPointer { source in
            Self.write(buffers.feats, rows: nMels, cols: capacity) { row, destination in
                let sourceRow = source.baseAddress! + row * frames
                for c in 0..<capacity { destination[c] = c < valid && c < frames ? Float16(sourceRow[c]) : 0 }
            }
        }
        let widths = [4 * w, 2 * w, w]
        for i in 0..<3 {
            let len = lengths[i]
            let fill: (Int, UnsafeMutablePointer<Float16>) -> Void = { _, destination in
                for c in 0..<widths[i] { destination[c] = c < len ? 1 : 0 }
            }
            switch i {
            case 0: Self.write(buffers.m1, rows: 1, cols: widths[i], fill)
            case 1: Self.write(buffers.m2, rows: 1, cols: widths[i], fill)
            default: Self.write(buffers.m3, rows: 1, cols: widths[i], fill)
            }
        }

        let provider = try MLDictionaryFeatureProvider(dictionary: [
            "feats": MLFeatureValue(multiArray: buffers.feats),
            "m1": MLFeatureValue(multiArray: buffers.m1),
            "m2": MLFeatureValue(multiArray: buffers.m2),
            "m3": MLFeatureValue(multiArray: buffers.m3),
        ])
        let prediction = try await model.prediction(from: provider)
        guard let output = prediction.featureValue(for: "enc")?.multiArrayValue else {
            throw EverParakeetError.missing("Core ML encoder output enc")
        }
        let flat = try Self.flatten(output)
        guard flat.count == hidden * w else { throw EverParakeetError.invalid("Core ML encoder output size \(flat.count)") }
        var frameMajor = [Float](repeating: 0, count: hidden * w)
        for channel in 0..<hidden {
            for frame in 0..<w { frameMajor[frame * hidden + channel] = flat[channel * w + frame] }
        }
        return (frameMajor, min(lengths[2], w))
    }
}
