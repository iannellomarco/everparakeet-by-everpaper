import Foundation

/// A loaded EverParakeet package: configuration, encoder (all windows), decoder weights and tokenizer. Immutable and
/// thread-safe; transcription runs on `EverParakeetLane`s made from it. Every lane has its own decoder state and its own
/// encoder input buffers, so lanes are independent (two lanes decode at the same time), while calls to the one shared
/// encoder model are serialised, interactive lanes first.
public final class EverParakeetModel: @unchecked Sendable {
    public struct LoadOptions {
        /// Compile cache of the Core ML encoder (folder, weight SHA, pruning).
        public var cache = CompiledModelCache()
        /// Route 5 to 10 second clips to the 10 second window (`main128`) when the package has it; a package without that
        /// function (the Core ML packages of Standard 1.1.x have main64, main192 and main384 only) is loaded without it.
        /// On the Core AI asset the text is identical to main192's on every short test-set clip routed this way
        /// (3,188 clips of nine lab sets, 1,049 of them between 5 and 10 s).
        public var useMain128 = true
        /// Workers of the decoder's NEON gemv per lane. nil = `recommendedDecoderThreads`.
        public var decoderThreads: Int? = nil
        /// Called while the encoder is prepared (compile, then one load per function).
        public var progress: LoadProgressHandler? = nil
        /// nil = the Core ML encoder. `EverParakeetCoreAI.CoreAIEncoderFactory` gives the Core AI one (macOS 27).
        public var encoderFactory: EncoderFactory? = nil
        public init() {}
    }

    public let config: EverParakeetConfig
    public let tokens: TokenDecoder
    let weights: DecoderWeights
    let encoder: EncoderBackend
    let gate = EncoderGate()
    let mel: MelFrontEnd
    let decoderThreads: Int

    /// 4 gemv workers; 2 on Macs with 8 GB of memory or fewer or with four performance cores or fewer (base chips).
    public static var recommendedDecoderThreads: Int {
        var performanceCores: Int32 = 0
        var size = MemoryLayout<Int32>.size
        let known = sysctlbyname("hw.perflevel0.physicalcpu", &performanceCores, &size, nil, 0) == 0
        let small = ProcessInfo.processInfo.physicalMemory <= 8 << 30 || (known && performanceCores <= 4)
        return small ? 2 : 4
    }

    init(config: EverParakeetConfig, tokens: TokenDecoder, weights: DecoderWeights, encoder: EncoderBackend, decoderThreads: Int) {
        self.config = config; self.tokens = tokens; self.weights = weights
        self.encoder = encoder; self.decoderThreads = decoderThreads
        self.mel = MelFrontEnd(config: config)
    }

    /// Windows (output frames) this model can run.
    public var windows: [Int] { encoder.windows }

    /// Verdict of the encoder's post-load speed probe (Core AI with `EncoderHealthPolicy.enabled`); nil for an encoder
    /// that is not probed (Core ML). When `onNeuralEngine` is false the model still works, only slowly: load the Core ML
    /// encoder instead.
    public var encoderHealth: EncoderHealth? { encoder.health }

    /// Loads a package folder (the one that holds `models/` and `tokenizer/`; its `models/` subfolder also works).
    ///
    /// The encoder is compiled once into the cache folder; the first load on a machine also specialises it for the Neural
    /// Engine (a few minutes, once). Loading is asynchronous and cancellable: cancel the task and the load stops before
    /// the next function.
    /// - Throws: `EverParakeetError` for a missing or malformed file, `CancellationError`, or a Core ML error.
    public static func load(root: URL, options: LoadOptions = LoadOptions()) async throws -> EverParakeetModel {
        var root = root
        if root.lastPathComponent == "models" { root.deleteLastPathComponent() }
        let config = try EverParakeetConfig(root: root)
        try Task.checkCancellation()
        options.progress?(.loadingDecoder)
        let (weights, tokens) = try await Task.detached(priority: .userInitiated) {
            (try DecoderWeights(root: root, config: config), try TokenDecoder(root: root))
        }.value
        try Task.checkCancellation()
        let factory = options.encoderFactory ?? CoreMLEncoderFactory(cache: options.cache, optionalWindows: [128])
        var windows = config.windows
        if options.useMain128 { windows = (windows + [128]).sorted() }
        let encoder = try await factory.makeEncoder(root: root, config: config, windows: windows, progress: options.progress)
        options.progress?(.ready)
        return EverParakeetModel(config: config, tokens: tokens, weights: weights, encoder: encoder,
                                 decoderThreads: options.decoderThreads ?? recommendedDecoderThreads)
    }

    /// A new lane: its own decoder state, encoder input buffers and decoder queue.
    public func makeLane(_ options: LaneOptions = LaneOptions()) throws -> EverParakeetLane {
        try EverParakeetLane(model: self, options: options)
    }

    /// Raw SentencePiece text of a token id (see `TokenDecoder.piece(for:)`).
    public func piece(for id: Int) -> String { tokens.piece(for: id) }

    /// Words of a token list, Greek final sigma applied per word (see `TokenDecoder.words(from:)`).
    public func words(from timed: [TimedToken]) -> [TimedWord] { tokens.words(from: timed) }
}
