import Foundation

#if !arch(arm64)
#error("EverParakeetKit requires Apple silicon (arm64): the decoder kernel in CHalfKernels is NEON-only.")
#endif

/// One emitted (non-blank) token of the TDT decoder.
///
/// An encoder frame is 80 ms (`EverParakeetConfig.secondsPerFrame`). `frame` counts from the start of the audio the
/// call was given (for long-form audio it is already shifted by the window start). `durationFrames` is the TDT
/// duration the decoder advanced by after emitting the token: 0 means the next token was emitted on the same frame,
/// and the duration is capped to 1 when the decoder hit `max_symbols_per_step`.
public struct TimedToken: Sendable, Equatable {
    public var id: Int
    public var frame: Int
    public var durationFrames: Int
    /// Softmax probability of the emitted token over the vocabulary (blank included), 0...1. When a `TokenChooser`
    /// replaced the token this is the confidence the chooser returned.
    public var confidence: Float
    public init(id: Int, frame: Int, durationFrames: Int, confidence: Float) {
        self.id = id; self.frame = frame; self.durationFrames = durationFrames; self.confidence = confidence
    }
}

/// A word assembled from consecutive tokens, with the Greek final-sigma rule applied at word level.
public struct TimedWord: Sendable, Equatable {
    public var text: String
    /// Frame of the first token.
    public var startFrame: Int
    /// Frame after the last token (last token's frame + max(1, durationFrames)).
    public var endFrame: Int
    /// Arithmetic mean of the token confidences.
    public var confidence: Float
    public var tokenCount: Int
    public init(text: String, startFrame: Int, endFrame: Int, confidence: Float, tokenCount: Int) {
        self.text = text; self.startFrame = startFrame; self.endFrame = endFrame
        self.confidence = confidence; self.tokenCount = tokenCount
    }
}

/// Per-call stage timings (ms).
public struct StageTimings: Sendable {
    public var mel = 0.0, encoder = 0.0, decoder = 0.0, steps = 0, joints = 0
    public var pieces = 0
    /// How many windows ran on each encoder function (key: output frames, e.g. 64, 128, 192, 384).
    public var windowUse: [Int: Int] = [:]
    public init() {}
}

/// The result of one `EverParakeetLane.transcribe`.
public struct Transcription: Sendable {
    /// The transcript (`words` joined by single spaces, for audio up to 30 s identical to the 1.1.x `transcribe` output).
    public var text: String
    public var tokens: [TimedToken]
    public var words: [TimedWord]
    public var timings: StageTimings
    public init(text: String, tokens: [TimedToken], words: [TimedWord], timings: StageTimings) {
        self.text = text; self.tokens = tokens; self.words = words; self.timings = timings
    }
}

/// Reported after every encoder window (30 s or less each) that has been fully decoded.
public struct TranscriptionProgress: Sendable, Equatable {
    public var processedSamples: Int
    public var totalSamples: Int
    /// Windows (pieces) finished so far.
    public var windows: Int
    public var fraction: Double { totalSamples > 0 ? min(1, Double(processedSamples) / Double(totalSamples)) : 1 }
}

public typealias ProgressHandler = @Sendable (TranscriptionProgress) -> Void

/// What `EverParakeetModel.load` reports while it prepares the encoder (compile once, then one load per function).
public enum LoadProgress: Sendable, Equatable {
    case compiling
    case loadingFunction(name: String, index: Int, of: Int)
    case loadingDecoder
    case ready
    /// 1.1.2: what the Core AI encoder is doing about its Neural Engine plan (specialise, probe, repair). Code that
    /// switches over `LoadProgress` without a `default` needs a branch for it.
    case encoder(EncoderStatus)
}

public typealias LoadProgressHandler = @Sendable (LoadProgress) -> Void

/// Cooperative cancel flag the decoder polls from its own dispatch queue (a `Task`'s cancellation is mirrored into it).
public final class CancelFlag: @unchecked Sendable {
    private let lock = NSLock()
    private var flag = false
    public init() {}
    public func cancel() { lock.lock(); flag = true; lock.unlock() }
    public var isCancelled: Bool { lock.lock(); defer { lock.unlock() }; return flag }
}

extension EverParakeetConfig {
    /// One encoder frame is 8 mel hops: 8 * 160 / 16000 = 80 ms.
    public var samplesPerFrame: Int { 8 * hop }
    public var secondsPerFrame: Double { Double(samplesPerFrame) / Double(sampleRate) }
}
