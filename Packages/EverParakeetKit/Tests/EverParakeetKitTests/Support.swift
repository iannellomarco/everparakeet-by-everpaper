import Foundation
import XCTest
@testable import EverParakeetKit

/// Deterministic generator (SplitMix64) so the synthetic weights are the same on every run.
struct SeededRandom: RandomNumberGenerator {
    var state: UInt64
    init(_ seed: UInt64) { state = seed }
    mutating func next() -> UInt64 {
        state &+= 0x9E37_79B9_7F4A_7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
        z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
        return z ^ (z >> 31)
    }
}

/// A tiny synthetic package on disk (hidden 16, vocabulary 23 pieces + blank) with a fake encoder, so the decoder, the
/// lanes and the long-form paths run without any model files.
enum TestPackage {
    static let hidden = 16
    static let blank = 23
    static let vocabSize = 24
    static let durations = 5
    /// ids 0...22
    static let pieces = ["<unk>", "\u{2581}a", "b", "\u{2581}c", "d", "\u{2581}", "\u{2581}the", "ing", "\u{2581}\u{03CC}", "\u{03C1}",
                         "\u{03BF}", "\u{03C3}", "\u{2581}\u{03BA}\u{03B1}\u{03B9}", "\u{2581}1", "9", "6", "\u{2581}\u{03C3}", "\u{03C2}",
                         "\u{2581}x", "y", "<pad>", "<eos>", "\u{2581}b"]

    static func makeTemporaryDirectory(_ name: String = "pkg") throws -> URL {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("everparakeet-tests-\(name)-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        return url
    }

    static func safetensors(_ tensors: [(String, [Int], [Float])]) -> Data {
        var header: [String: Any] = [:]
        var payload = Data()
        for (name, shape, values) in tensors {
            let start = payload.count
            for v in values { var h = Float16(v).bitPattern.littleEndian; payload.append(Data(bytes: &h, count: 2)) }
            header[name] = ["dtype": "F16", "shape": shape, "data_offsets": [start, payload.count]] as [String: Any]
        }
        var json = try! JSONSerialization.data(withJSONObject: header, options: [.sortedKeys])
        while (json.count % 8) != 0 { json.append(0x20) }
        var length = UInt64(json.count).littleEndian
        var out = Data(bytes: &length, count: 8)
        out.append(json); out.append(payload)
        return out
    }

    /// Writes config.json, decoder_joint.safetensors and tokenizer/ into `root`.
    static let blankLift: Float = 2.2

    static func write(to root: URL, seed: UInt64 = 7, blankBias: Float? = nil) throws {
        let fm = FileManager.default
        try fm.createDirectory(at: root.appendingPathComponent("models"), withIntermediateDirectories: true)
        try fm.createDirectory(at: root.appendingPathComponent("tokenizer"), withIntermediateDirectories: true)
        var long: [String: Any] = ["max_window_seconds": 30, "silence_cut_search_seconds": 4, "silence_cut_window_seconds": 0.2]
        if let b = blankBias { long["blank_bias"] = b }
        let config: [String: Any] = [
            "sample_rate": 16000, "vocab_size": vocabSize, "blank_id": blank, "decoder_hidden_size": hidden, "decoder_layers": 2,
            "durations": [0, 1, 2, 3, 4], "max_symbols_per_step": 10,
            "mel": ["num_mel_bins": 128, "n_fft": 512, "win_length": 400, "hop_length": 160, "preemphasis": 0.97, "log_guard": 5.960464477539063e-08],
            "long_audio": long,
        ]
        try JSONSerialization.data(withJSONObject: config).write(to: root.appendingPathComponent("models/config.json"))
        var rng = SeededRandom(seed)
        func random(_ n: Int, _ scale: Float) -> [Float] { (0..<n).map { _ in Float.random(in: -scale...scale, using: &rng) } }
        let H = hidden
        var tensors: [(String, [Int], [Float])] = [("decoder.embedding.weight", [vocabSize, H], random(vocabSize * H, 0.8))]
        for l in 0..<2 {
            tensors.append(("decoder.lstm.weight_ih_l\(l)", [4 * H, H], random(4 * H * H, 0.4)))
            tensors.append(("decoder.lstm.weight_hh_l\(l)", [4 * H, H], random(4 * H * H, 0.4)))
            tensors.append(("decoder.lstm.bias_ih_l\(l)", [4 * H], random(4 * H, 0.1)))
            tensors.append(("decoder.lstm.bias_hh_l\(l)", [4 * H], random(4 * H, 0.1)))
        }
        tensors.append(("decoder.decoder_projector.weight", [H, H], random(H * H, 0.5)))
        tensors.append(("decoder.decoder_projector.bias", [H], random(H, 0.1)))
        tensors.append(("joint.head.weight", [vocabSize + durations, H], random((vocabSize + durations) * H, 0.9)))
        var headBias = random(vocabSize + durations, 0.3)
        headBias[blank] += blankLift                    // blank wins often enough that a blank bias visibly changes the output
        tensors.append(("joint.head.bias", [vocabSize + durations], headBias))
        try safetensors(tensors).write(to: root.appendingPathComponent("models/decoder_joint.safetensors"))
        var vocab: [String: Int] = [:]
        for (i, p) in pieces.enumerated() { vocab[p] = i }
        let tokenizer: [String: Any] = ["model": ["vocab": vocab]]
        try JSONSerialization.data(withJSONObject: tokenizer).write(to: root.appendingPathComponent("tokenizer/tokenizer.json"))
        let tokenizerConfig: [String: Any] = ["unk_token": "<unk>", "pad_token": "<pad>", "eos_token": "<eos>"]
        try JSONSerialization.data(withJSONObject: tokenizerConfig).write(to: root.appendingPathComponent("tokenizer/tokenizer_config.json"))
    }

    /// Model over the synthetic package with a fake encoder.
    static func model(root: URL, encoder: FakeEncoder = FakeEncoder(), decoderThreads: Int = 2) throws -> EverParakeetModel {
        let config = try EverParakeetConfig(root: root)
        return EverParakeetModel(config: config, tokens: try TokenDecoder(root: root),
                                 weights: try DecoderWeights(root: root, config: config), encoder: encoder,
                                 decoderThreads: decoderThreads)
    }

    /// Mono 16 kHz audio with a few "words" of noise and short pauses; deterministic.
    static func audio(seconds: Double, seed: UInt64 = 3, pauses: Bool = true) -> [Float] {
        var rng = SeededRandom(seed)
        let n = Int(seconds * 16000)
        return (0..<n).map { i in
            let t = Double(i) / 16000
            let gate: Float = pauses && (Int(t * 2) % 5 == 4) ? 0.001 : 1
            return Float.random(in: -0.3...0.3, using: &rng) * gate
        }
    }
}

/// Encoder stand-in: a deterministic function of the mel frame index and the channel. Counts calls, can sleep, and
/// detects two calls that use the same input buffers at the same time.
final class FakeEncoder: EncoderBackend, @unchecked Sendable {
    final class Inputs: EncoderInputs { let id = UUID(); let serial: Int; init(serial: Int) { self.serial = serial } }
    let windows: [Int]
    var delay: TimeInterval = 0
    private let lock = NSLock()
    private(set) var calls: [(window: Int, inputs: UUID, serial: Int)] = []
    private var inFlight = Set<UUID>()
    private(set) var overlappingInputsUse = false
    private(set) var inputsMade = 0
    var onCall: ((Int, UUID) -> Void)?
    var firstCallDelay: TimeInterval = 0
    /// A wait that ignores cancellation, like a real `MLModel.prediction` (added to `delay`, which is cancellable).
    var uncancellableDelay: TimeInterval = 0

    init(windows: [Int] = [64, 192, 384]) { self.windows = windows }

    func makeInputs() throws -> EncoderInputs {
        lock.lock(); inputsMade += 1; let serial = inputsMade; lock.unlock()
        return Inputs(serial: serial)
    }

    private func begin(window: Int, id: UUID, serial: Int) -> (TimeInterval, ((Int, UUID) -> Void)?) {
        lock.lock(); defer { lock.unlock() }
        let first = calls.isEmpty
        calls.append((window, id, serial))
        if inFlight.contains(id) { overlappingInputsUse = true }
        inFlight.insert(id)
        return (delay + (first ? firstCallDelay : 0), onCall)
    }

    private func end(id: UUID) {
        lock.lock(); inFlight.remove(id); lock.unlock()
    }

    func encode(mel: [Float], frames: Int, valid: Int, window: Int, inputs: EncoderInputs) async throws -> (enc: [Float], validFrames: Int) {
        let id = (inputs as! Inputs).id, serial = (inputs as! Inputs).serial
        let (wait, hook) = begin(window: window, id: id, serial: serial)
        hook?(window, id)
        if wait > 0 { try await Task.sleep(nanoseconds: UInt64(wait * 1e9)) }
        if uncancellableDelay > 0 {
            let d = uncancellableDelay
            await withCheckedContinuation { c in DispatchQueue.global().asyncAfter(deadline: .now() + d) { c.resume() } }
        }
        end(id: id)
        let H = TestPackage.hidden
        var enc = [Float](repeating: 0, count: window * H)
        for f in 0..<window {
            let energy: Float = f < frames ? abs(mel[min(f, frames - 1)]) : 0     // channel 0 of the mel
            for c in 0..<H { enc[f * H + c] = 1.8 * sin(Float(f) * 0.37 + Float(c) * 0.9) + 0.3 * energy }
        }
        return (enc, min(maskLengths(valid)[2], window))
    }
}
