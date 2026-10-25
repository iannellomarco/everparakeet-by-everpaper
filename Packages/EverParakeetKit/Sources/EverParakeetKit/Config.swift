import Foundation

public enum EverParakeetError: Error, CustomStringConvertible {
    case missing(String)
    case invalid(String)
    public var description: String {
        switch self {
        case .missing(let s): return "missing: \(s)"
        case .invalid(let s): return "invalid: \(s)"
        }
    }
}

/// The constants of `models/config.json` that the runtime needs.
public struct EverParakeetConfig: Sendable {
    public let sampleRate: Int
    public let vocabSize: Int          // includes blank (8193)
    public let blankID: Int
    public let hidden: Int
    public let layers: Int
    public let durations: [Int]
    public let maxSymbols: Int
    public let nFFT: Int
    public let winLength: Int
    public let hop: Int
    public let nMels: Int
    public let preemphasis: Float
    public let logGuard: Float
    public let maxWindowSeconds: Int
    public let silenceSearchSeconds: Int
    public let silenceWindowSeconds: Double
    /// Optional `long_audio.blank_bias` of the package: the long-audio blank bias calibrated for its encoder.
    public let longBlankBias: Float?
    /// Windows (encoder output frames) the pipeline routes to by default: 5, 15 and 30 seconds. The 10 second window
    /// (`main128`, 128 frames) is routed to only when `EverParakeetModel.LoadOptions.useMain128` is set and the package has it.
    public let windows: [Int] = [64, 192, 384]

    public init(root: URL) throws {
        let data = try Data(contentsOf: root.appendingPathComponent("models/config.json"))
        guard let j = try JSONSerialization.jsonObject(with: data) as? [String: Any],
              let mel = j["mel"] as? [String: Any],
              let long = j["long_audio"] as? [String: Any]
        else { throw EverParakeetError.invalid("config.json") }
        func i(_ d: [String: Any], _ k: String) throws -> Int {
            guard let v = d[k] as? NSNumber else { throw EverParakeetError.invalid("config.json key \(k)") }
            return v.intValue
        }
        func f(_ d: [String: Any], _ k: String) throws -> Double {
            guard let v = d[k] as? NSNumber else { throw EverParakeetError.invalid("config.json key \(k)") }
            return v.doubleValue
        }
        sampleRate = try i(j, "sample_rate")
        vocabSize = try i(j, "vocab_size")
        blankID = try i(j, "blank_id")
        hidden = try i(j, "decoder_hidden_size")
        layers = try i(j, "decoder_layers")
        durations = (j["durations"] as? [NSNumber] ?? []).map { $0.intValue }
        maxSymbols = try i(j, "max_symbols_per_step")
        nFFT = try i(mel, "n_fft")
        winLength = try i(mel, "win_length")
        hop = try i(mel, "hop_length")
        nMels = try i(mel, "num_mel_bins")
        preemphasis = Float(try f(mel, "preemphasis"))
        logGuard = Float(try f(mel, "log_guard"))
        maxWindowSeconds = try i(long, "max_window_seconds")
        silenceSearchSeconds = try i(long, "silence_cut_search_seconds")
        silenceWindowSeconds = try f(long, "silence_cut_window_seconds")
        longBlankBias = (long["blank_bias"] as? NSNumber).map { $0.floatValue }
    }
}

/// Minimal safetensors reader (header JSON + raw little-endian payload).
struct SafeTensors {
    struct Entry { let dtype: String; let shape: [Int]; let start: Int; let end: Int }
    let data: Data
    let base: Int
    let entries: [String: Entry]

    init(url: URL) throws {
        data = try Data(contentsOf: url, options: .alwaysMapped)
        guard data.count > 8 else { throw EverParakeetError.invalid("safetensors too short") }
        let n = data.prefix(8).enumerated().reduce(UInt64(0)) { $0 | (UInt64($1.element) << UInt64(8 * $1.offset)) }
        guard n <= UInt64(data.count - 8) else { throw EverParakeetError.invalid("safetensors header length \(n) exceeds the file") }
        base = 8 + Int(n)
        guard let j = try JSONSerialization.jsonObject(with: data.subdata(in: 8..<base)) as? [String: Any]
        else { throw EverParakeetError.invalid("safetensors header") }
        let payload = data.count - base
        var e: [String: Entry] = [:]
        for (k, v) in j where k != "__metadata__" {
            guard let d = v as? [String: Any], let dt = d["dtype"] as? String,
                  let sh = d["shape"] as? [NSNumber], let off = d["data_offsets"] as? [NSNumber], off.count == 2
            else { throw EverParakeetError.invalid("safetensors entry \(k)") }
            let start = off[0].intValue, end = off[1].intValue
            let size = dt == "F16" ? 2 : dt == "F32" ? 4 : dt == "I8" ? 1 : 0
            let count = sh.reduce(1) { $0 * $1.intValue }
            guard sh.allSatisfy({ $0.intValue >= 0 }), start >= 0, start <= end, end <= payload else {
                throw EverParakeetError.invalid("safetensors tensor \(k): offsets \(start)..<\(end) outside the \(payload)-byte payload")
            }
            if size != 0 && end - start != count * size {
                throw EverParakeetError.invalid("safetensors tensor \(k): \(end - start) bytes for \(count) \(dt) values")
            }
            e[k] = Entry(dtype: dt, shape: sh.map { $0.intValue }, start: start, end: end)
        }
        entries = e
    }

    /// Tensor as Float16 (the checkpoint's native storage), row-major. An int8 matrix with a "<name>.scale" companion
    /// (the everparakeet-decoder-int8 format) is dequantized to fp16 here, so callers get the same buffer as from a fp16 file.
    func halves(_ name: String) throws -> [Float16] {
        guard let e = entries[name] else { throw EverParakeetError.missing("tensor \(name)") }
        if e.dtype == "I8" { return try dequantizedHalves(name, e) }
        guard e.dtype == "F16" else { throw EverParakeetError.invalid("\(name) is \(e.dtype), expected F16 or I8") }
        let count = e.shape.reduce(1, *)
        return data.withUnsafeBytes { raw -> [Float16] in
            let p = (raw.baseAddress! + base + e.start).assumingMemoryBound(to: Float16.self)
            return Array(UnsafeBufferPointer(start: p, count: count))
        }
    }

    /// int8 matrix [rows, cols] x fp16 scales -> fp16(Float(q) * Float(scale)). "<name>.scale" is [rows] (one scale per
    /// output row) or [rows, groups] (one per cols / groups input columns). The fp32 product is exact and is rounded to
    /// fp16 once, so the result is bit-identical to the Python runtime's dequantization.
    private func dequantizedHalves(_ name: String, _ e: Entry) throws -> [Float16] {
        guard e.shape.count == 2 else { throw EverParakeetError.invalid("\(name): int8 tensors must be matrices, found shape \(e.shape)") }
        let rows = e.shape[0], cols = e.shape[1]
        guard let se = entries[name + ".scale"] else { throw EverParakeetError.missing("tensor \(name).scale (needed to dequantize \(name))") }
        let groups: Int
        if se.shape == [rows] { groups = 1 }
        else if se.shape.count == 2, se.shape[0] == rows, se.shape[1] > 0, cols % se.shape[1] == 0 { groups = se.shape[1] }
        else { throw EverParakeetError.invalid("\(name).scale has shape \(se.shape), expected [\(rows)] or [\(rows), groups]") }
        let scales = try floats(name + ".scale")
        guard scales.allSatisfy({ $0.isFinite && $0 >= 0 }) else { throw EverParakeetError.invalid("\(name).scale holds a negative or non-finite value") }
        let width = cols / groups, count = rows * cols
        let threads = rows >= 256 ? 4 : 1
        let chunk = (rows + threads - 1) / threads
        return data.withUnsafeBytes { raw -> [Float16] in
            let q = (raw.baseAddress! + base + e.start).assumingMemoryBound(to: Int8.self)
            return scales.withUnsafeBufferPointer { s in
                [Float16](unsafeUninitializedCapacity: count) { buf, n in
                    let out = buf.baseAddress!
                    DispatchQueue.concurrentPerform(iterations: threads) { t in
                        for r in (t * chunk)..<max(t * chunk, min(rows, (t + 1) * chunk)) {
                            for g in 0..<groups {
                                let scale = s[r * groups + g]
                                let lo = r * cols + g * width
                                for i in lo..<(lo + width) { out[i] = Float16(Float(q[i]) * scale) }
                            }
                        }
                    }
                    n = count
                }
            }
        }
    }

    /// Tensor as Float32 (fp16 or fp32 payload), row-major.
    func floats(_ name: String) throws -> [Float] {
        guard let e = entries[name] else { throw EverParakeetError.missing("tensor \(name)") }
        guard e.dtype == "F16" || e.dtype == "F32" else { throw EverParakeetError.invalid("\(name) is \(e.dtype), expected F16 or F32") }
        let count = e.shape.reduce(1, *)
        return data.withUnsafeBytes { raw -> [Float] in
            let p = raw.baseAddress! + base + e.start
            switch e.dtype {
            case "F16":
                let h = p.assumingMemoryBound(to: Float16.self)
                return [Float](unsafeUninitializedCapacity: count) { buf, n in
                    for i in 0..<count { buf[i] = Float(h[i]) }
                    n = count
                }
            default: // F32
                return Array(UnsafeBufferPointer(start: p.assumingMemoryBound(to: Float.self), count: count))
            }
        }
    }
}
