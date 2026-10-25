// Portions Copyright 2026 Apple Inc. Modifications Copyright 2026 Everpaper.
//
// flattenAsFloat and flattenNDArray are adapted from CoreAIShared/Runtime/NDArray+Helpers.swift in Apple's
// coreai-models repository; the NDArray input set-up follows Apple's usage pattern.
//
// This whole file is distributed under Apple's BSD 3-clause licence (full text in
// swift/THIRD_PARTY_NOTICES):
//
// Redistribution and use in source and binary forms, with or without modification, are permitted provided
// that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the
//    following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and
//    the following disclaimer in the documentation and/or other materials provided with the distribution.
// 3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or
//    promote products derived from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED
// WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
// PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
// ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR
// TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
// ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#if canImport(CoreAI)
import CoreAI
import EverParakeetKit
import Foundation

/// The Core AI encoder (macOS 27, iOS 27): the ANE-authored `encoder.aimodel`. Pass it to `EverParakeetModel.load` as
/// `LoadOptions.encoderFactory`. This target is separate from `EverParakeetKit` so an app that only uses Core ML can
/// leave it out of its build (and never links the Core AI framework).
///
/// 1.1.2: the load specialises explicitly (`AIModel.specialize`), times it and reports it (`LoadProgress.encoder`), probes
/// the smallest window on silence, and when that is far slower than the Neural Engine should be (Core AI cached a plan
/// that is not on the Neural Engine, for example after `com.apple.ANECompilerService` could not be reached) deletes this
/// model's cache entries and specialises once more. See `EncoderHealthPolicy`; `model.encoderHealth` has the verdict.
@available(macOS 27.0, iOS 27.0, *)
public struct CoreAIEncoderFactory: EncoderFactory {
    public var health: EncoderHealthPolicy
    public init(health: EncoderHealthPolicy = EncoderHealthPolicy()) { self.health = health }

    public func makeEncoder(root: URL, config: EverParakeetConfig, windows: [Int], progress: LoadProgressHandler?) async throws -> EncoderBackend {
        let preparer = CoreAIPreparer(assetURL: root.appendingPathComponent("models/encoder.aimodel"), windowSizes: windows,
                                      hidden: config.hidden, nMels: config.nMels, compute: .neuralEngine, progress: progress)
        // `.compiling` is reported by the encoder itself, and only when preparing takes long enough to be a real
        // first-run specialisation rather than a cache hit.
        guard health.enabled else { return try await preparer.prepare() }
        let policy = health
        let nMels = config.nMels
        let healer = EncoderHealer(policy: policy, memoryKey: "\(currentOSBuild())|\(preparer.assetKey)",
                                   status: { progress?(.encoder($0)) },
                                   measure: { try await probeEncoder($0, nMels: nMels, policy: policy) })
        let (encoder, verdict) = try await healer.run(preparer)
        (encoder as? CoreAIEncoder)?.health = verdict
        return encoder
    }
}

/// Specialises through the Core AI cache and deletes this model's cache entries.
@available(macOS 27.0, iOS 27.0, *)
struct CoreAIPreparer: EncoderPreparer {
    let assetURL: URL
    let windowSizes: [Int]
    let hidden: Int
    let nMels: Int
    let compute: ComputeUnitKind
    let progress: LoadProgressHandler?

    func prepare() async throws -> EncoderBackend {
        try await CoreAIEncoder(assetURL: assetURL, windowSizes: windowSizes, hidden: hidden, nMels: nMels,
                                compute: compute, progress: progress)
    }

    func purgeCache() throws { try AIModelCache.default.deleteEntries(for: assetURL) }

    /// Name and modification time of the asset: a new model file is a new key for the remembered outcome.
    var assetKey: String {
        let modified = (try? FileManager.default.attributesOfItem(atPath: assetURL.path)[.modificationDate] as? Date) ?? nil
        return "\(assetURL.lastPathComponent)-\(Int64((modified?.timeIntervalSince1970 ?? 0) * 1000))"
    }
}

/// Inputs `feats` [1,1,128,8W] fp16, `m1` [1,1,1,4W], `m2` [1,1,1,2W], `m3` [1,1,1,W]; output `enc` [1,640,1,W].
@available(macOS 27.0, iOS 27.0, *)
final class CoreAIEncoder: EncoderBackend, @unchecked Sendable {
    struct Window {
        let size: Int                      // output frames W
        let function: InferenceFunction
    }
    final class Buffers {
        var feats: NDArray
        var m1: NDArray, m2: NDArray, m3: NDArray
        init(feats: NDArray, m1: NDArray, m2: NDArray, m3: NDArray) { self.feats = feats; self.m1 = m1; self.m2 = m2; self.m3 = m3 }
    }
    final class Inputs: EncoderInputs {
        var windows: [Int: Buffers] = [:]
    }
    private var loaded: [Int: Window] = [:]
    private let model: AIModel
    /// Set by the factory after the speed probe.
    var health: EncoderHealth?
    let windows: [Int]
    let hidden: Int
    let nMels: Int

    init(assetURL: URL, windowSizes: [Int], hidden: Int, nMels: Int, compute: ComputeUnitKind = .neuralEngine,
         progress: LoadProgressHandler? = nil) async throws {
        self.hidden = hidden
        self.nMels = nMels
        // Core AI offers no "was this a cache hit" answer, so the first-run notice is time-based: a cached load takes
        // milliseconds, a first specialisation many seconds. Say "compiling" only once the wait has become noticeable.
        let notice = Task {
            try await Task.sleep(nanoseconds: Self.compilingNoticeDelay)
            progress?(.compiling)
        }
        defer { notice.cancel() }
        // Explicit specialisation through the default cache: the same work `AIModel(contentsOf:)` does, but a failure
        // surfaces as a thrown error the caller can report, and `EncoderHealer` can purge the entry it produced.
        let model = try await AIModel.specialize(contentsOf: assetURL, options: SpecializationOptions(preferredComputeUnitKind: compute),
                                                 cache: .default, cachePolicy: .default)
        self.model = model
        for (index, w) in windowSizes.enumerated() {
            try Task.checkCancellation()
            let name = "main\(w)"
            progress?(.loadingFunction(name: name, index: index, of: windowSizes.count))
            guard let fn = try await Self.loadFunction(model, name), model.functionDescriptor(for: name) != nil else {
                if w == 128 { continue }          // optional window
                throw EverParakeetError.missing("encoder function \(name)")
            }
            loaded[w] = Window(size: w, function: fn)
        }
        windows = loaded.keys.sorted()
    }

    /// How long preparing the encoder may take before the load reports `.compiling`.
    static let compilingNoticeDelay: UInt64 = 1_500_000_000

    /// `loadFunction(named:)` is synchronous and can take seconds on a first run: it runs on a dispatch thread so it never
    /// holds a thread of Swift's cooperative pool.
    private static func loadFunction(_ model: AIModel, _ name: String) async throws -> InferenceFunction? {
        struct Box: @unchecked Sendable { let model: AIModel }
        let box = Box(model: model)
        return try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<InferenceFunction?, Error>) in
            DispatchQueue.global(qos: .userInitiated).async {
                continuation.resume(with: Result { try box.model.loadFunction(named: name) })
            }
        }
    }

    func makeInputs() throws -> EncoderInputs {
        let inputs = Inputs()
        for w in loaded.keys {
            guard let descriptor = model.functionDescriptor(for: "main\(w)") else { throw EverParakeetError.missing("encoder function main\(w)") }
            func input(_ n: String, _ shape: [Int]) throws -> NDArray {
                guard case .ndArray(let d) = descriptor.inputDescriptor(of: n) else { throw EverParakeetError.invalid("encoder input \(n)") }
                return NDArray(descriptor: d.resolvingDynamicDimensions(shape))
            }
            inputs.windows[w] = Buffers(
                feats: try input("feats", [1, 1, nMels, 8 * w]),
                m1: try input("m1", [1, 1, 1, 4 * w]),
                m2: try input("m2", [1, 1, 1, 2 * w]),
                m3: try input("m3", [1, 1, 1, w]))
        }
        return inputs
    }

    /// Stride-aware write of a `[rows, cols]` row-major source into the trailing two dims of `array`.
    private static func write(_ array: inout NDArray, rows: Int, cols: Int, _ fill: (Int, UnsafeMutablePointer<Float16>) -> Void) {
        let view = array.mutableView(as: Float16.self)
        view.withUnsafeMutablePointer { ptr, shape, strides in
            let rank = shape.count
            let rs = strides[rank - 2], cs = strides[rank - 1]
            if cs == 1 {
                for r in 0..<rows { fill(r, ptr + r * rs) }
            } else {
                var row = [Float16](repeating: 0, count: cols)
                for r in 0..<rows {
                    row.withUnsafeMutableBufferPointer { fill(r, $0.baseAddress!) }
                    for c in 0..<cols { ptr[r * rs + c * cs] = row[c] }
                }
            }
        }
    }

    /// mel: channel-major [nMels, frames] fp32; returns the encoder output as [W, hidden] row-major (frame-major).
    func encode(mel: [Float], frames: Int, valid: Int, window w: Int, inputs: EncoderInputs) async throws -> (enc: [Float], validFrames: Int) {
        guard let win = loaded[w] else { throw EverParakeetError.missing("window \(w)") }
        guard let buffers = (inputs as? Inputs)?.windows[w] else { throw EverParakeetError.invalid("encoder inputs of another backend, or window \(w) missing") }
        let capacity = 8 * w
        let lengths = maskLengths(valid)
        let widths = [4 * w, 2 * w, w]
        mel.withUnsafeBufferPointer { src in
            Self.write(&buffers.feats, rows: nMels, cols: capacity) { r, dst in
                let row = src.baseAddress! + r * frames
                for c in 0..<frames { dst[c] = c < valid ? Float16(row[c]) : 0 }
                for c in frames..<capacity { dst[c] = 0 }
            }
        }
        for (i, width) in widths.enumerated() {
            let len = lengths[i]
            let fill: (Int, UnsafeMutablePointer<Float16>) -> Void = { _, dst in
                for c in 0..<width { dst[c] = c < len ? 1 : 0 }
            }
            switch i {
            case 0: Self.write(&buffers.m1, rows: 1, cols: width, fill)
            case 1: Self.write(&buffers.m2, rows: 1, cols: width, fill)
            default: Self.write(&buffers.m3, rows: 1, cols: width, fill)
            }
        }
        var outputs = try await win.function.run(inputs: ["feats": buffers.feats, "m1": buffers.m1, "m2": buffers.m2, "m3": buffers.m3])
        guard let out = outputs.remove("enc")?.ndArray else { throw EverParakeetError.missing("encoder output enc") }
        let flat = flattenAsFloat(out)                    // [1, hidden, 1, W] row-major = [hidden, W]
        guard flat.count == hidden * w else { throw EverParakeetError.invalid("encoder output size \(flat.count)") }
        var frameMajor = [Float](repeating: 0, count: w * hidden)
        // transpose [hidden, W] -> [W, hidden]
        flat.withUnsafeBufferPointer { s in
            frameMajor.withUnsafeMutableBufferPointer { d in
                for c in 0..<hidden {
                    let row = s.baseAddress! + c * w
                    for t in 0..<w { d[t * hidden + c] = row[t] }
                }
            }
        }
        let validFrames = min(lengths[2], w)        // int(masks[-1].sum())
        return (frameMajor, validFrames)
    }
}

/// Local copy of Apple's `flattenAsFloat` (stride-aware, converts from the array's own scalar type).
@available(macOS 27.0, iOS 27.0, *)
func flattenAsFloat(_ array: NDArray) -> [Float] {
    switch array.scalarType {
    case .float16: return flattenNDArray(array, as: Float16.self)
    case .float32: return flattenNDArray(array, as: Float.self)
    default: preconditionFailure("unsupported scalar type \(array.scalarType)")
    }
}

@available(macOS 27.0, iOS 27.0, *)
func flattenNDArray<T: BinaryFloatingPoint & BitwiseCopyable>(_ array: NDArray, as type: T.Type) -> [Float] {
    let total = array.shape.reduce(1, *)
    var result = [Float](repeating: 0, count: total)
    array.view(as: type).withUnsafePointer { ptr, shape, strides in
        var contiguous = true
        var expected = 1
        for d in (0..<shape.count).reversed() {
            if strides[d] != expected { contiguous = false; break }
            expected *= shape[d]
        }
        if contiguous {
            for i in 0..<total { result[i] = Float(ptr[i]) }
            return
        }
        let rank = shape.count
        var idx = [Int](repeating: 0, count: rank)
        for i in 0..<total {
            var off = 0
            for d in 0..<rank { off += idx[d] * strides[d] }
            result[i] = Float(ptr[off])
            var d = rank - 1
            while d >= 0 { idx[d] += 1; if idx[d] < shape[d] { break }; idx[d] = 0; d -= 1 }
        }
    }
    return result
}
#endif
