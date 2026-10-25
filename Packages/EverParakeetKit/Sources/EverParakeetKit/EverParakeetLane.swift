import Foundation

public struct LaneOptions: Sendable {
    /// Who gets the shared encoder first when lanes compete: dictation lanes `.interactive`, final passes `.background`.
    public var priority: LanePriority = .interactive
    /// QoS of the lane's decoder queue. nil: `.userInitiated` for an interactive lane, `.utility` for a background one.
    public var qos: DispatchQoS? = nil
    /// Workers of the decoder's gemv. nil = the model's setting.
    public var decoderThreads: Int? = nil
    public var label = "lane"
    public init() {}
}

public struct TranscribeOptions: Sendable {
    /// Used for audio over 30 s.
    public var longForm = LongFormOptions()
    /// Size of the top-k list a `TokenChooser` gets.
    public var chooserTopK = 16
    public init() {}
}

/// One independent transcription stream over a shared `EverParakeetModel`.
///
/// An actor: `transcribe` calls on one lane run one after the other. The mel front end and the decoder run on the lane's
/// own serial `DispatchQueue` (so they never occupy Swift's cooperative thread pool); only the encoder call is awaited.
/// Cancel the calling `Task` and the lane stops between windows, every few joint calls inside the decoder (within
/// milliseconds), and while it is queued: behind another call on the same lane, or for the shared encoder. (An encoder
/// prediction that is already running cannot be interrupted and finishes first; that is one window at most.)
public actor EverParakeetLane {
    public let model: EverParakeetModel
    public let priority: LanePriority

    final class State: @unchecked Sendable {
        let decoder: CPUTDTDecoder
        let inputs: EncoderInputs
        var chooser: TokenChooser?
        init(decoder: CPUTDTDecoder, inputs: EncoderInputs) { self.decoder = decoder; self.inputs = inputs }
    }
    private let state: State
    private let queue: DispatchQueue
    private var busy = false
    private var waiters = LockWaiters(priorities: 1)
    public private(set) var lastTimings = StageTimings()

    init(model: EverParakeetModel, options: LaneOptions) throws {
        self.model = model
        self.priority = options.priority
        let qos = options.qos ?? (options.priority == .interactive ? .userInitiated : .utility)
        queue = DispatchQueue(label: "app.everpaper.everparakeet.\(options.label).\(UUID().uuidString.prefix(8))", qos: qos)
        state = State(decoder: CPUTDTDecoder(weights: model.weights, threads: options.decoderThreads ?? model.decoderThreads),
                      inputs: try model.encoder.makeInputs())
    }

    /// Identity of this lane's encoder input buffers (for tests: two lanes must differ).
    nonisolated var inputsIdentity: ObjectIdentifier { ObjectIdentifier(state.inputs) }

    /// Installs (or removes) the lane's token chooser. Every `transcribe` that starts later uses it, whole: a call
    /// reads the chooser once when it starts, so a change never lands between two windows of one recording.
    public func setChooser(_ chooser: TokenChooser?) { state.chooser = chooser }

    /// Returns holding the lane, or throws `CancellationError` without it (a call queued behind another stops
    /// waiting the moment its task is cancelled, however long the running call has left).
    private func acquire() async throws {
        try Task.checkCancellation()
        if !busy { busy = true; return }
        let id = waiters.newID()
        try await withTaskCancellationHandler {
            try await withCheckedThrowingContinuation { (continuation: LockWaiters.Waiter) in
                if Task.isCancelled { continuation.resume(throwing: CancellationError()); return }
                waiters.add(continuation, id: id)     // the lock is handed over directly
            }
        } onCancel: {
            Task { await self.cancelWaiter(id) }
        }
    }

    private func cancelWaiter(_ id: UInt64) {
        waiters.remove(id: id)?.resume(throwing: CancellationError())
    }

    private func release() {
        if let next = waiters.next() { next.resume() } else { busy = false }
    }

    /// Transcribes mono 16 kHz float PCM.
    public func transcribe(pcm16k samples: [Float], options: TranscribeOptions = TranscribeOptions(),
                           progress: ProgressHandler? = nil) async throws -> Transcription {
        try await transcribe(source: ArraySampleSource(samples), options: options, progress: progress)
    }

    /// Transcribes a sample source. Audio of 30 s or less is one window; longer audio goes through `options.longForm`.
    /// For audio up to 30 s, and with `LongFormOptions.off` for longer audio, the text is bit-identical to Kit 1.1.x.
    public func transcribe(source: SampleSource, options: TranscribeOptions = TranscribeOptions(),
                           progress: ProgressHandler? = nil) async throws -> Transcription {
        try await acquire()
        defer { release() }
        try Task.checkCancellation()
        var timings = StageTimings()
        let config = model.config
        let total = source.sampleCount
        let maxSamples = config.maxWindowSeconds * config.sampleRate
        let isLong = total > maxSamples
        let lf = options.longForm
        var params = CPUTDTDecoder.Params()
        if isLong || lf.biasShort {
            params.blankBias = lf.blankBias ?? config.longBlankBias ?? LongFormOptions.defaultBlankBias
            params.durationCap = lf.durationCap
        }
        params.loopRepeats = lf.loopRepeats
        params.loopRepeats1 = lf.loopRepeats1
        params.chooserTopK = options.chooserTopK
        let chooser = state.chooser

        let result: (text: String, tokens: [TimedToken], words: [TimedWord])
        if isLong && lf.mode == .lcs {
            result = try await transcribeLCS(source, lf, params, chooser, &timings, progress)
        } else {
            result = try await transcribeCut(source, params, chooser, &timings, progress)
        }
        lastTimings = timings
        return Transcription(text: result.text, tokens: result.tokens, words: result.words, timings: timings)
    }

    // MARK: pieces

    /// Mel -> encoder -> decoder for one window of audio; tokens carry global frames (`frameOffset` + local).
    private func decodeWindow(_ pcm: [Float], frameOffset: Int, params: CPUTDTDecoder.Params, chooser: TokenChooser?,
                              _ timings: inout StageTimings) async throws -> [TimedToken] {
        try Task.checkCancellation()
        let model = self.model, state = self.state
        let t0 = DispatchTime.now()
        let mel = try await onQueue { _ in model.mel.compute(pcm) }
        let t1 = DispatchTime.now()
        guard let window = routeWindow(melFrames: mel.frames, windows: model.encoder.windows) else {
            throw EverParakeetError.invalid("audio piece exceeds the 30-second encoder window")
        }
        timings.windowUse[window, default: 0] += 1
        var enc: [Float] = []
        var valid = 0
        if mel.valid > 0 {              // no valid frame: the decoder would emit nothing, skip the encoder call
            let priority = self.priority
            (enc, valid) = try await model.gate.run(priority) {
                try await model.encoder.encode(mel: mel.mel, frames: mel.frames, valid: mel.valid, window: window, inputs: state.inputs)
            }
        }
        let t2 = DispatchTime.now()
        let piece: @Sendable (Int) -> String = { model.tokens.piece(for: $0) }
        let encoded = enc, validFrames = valid
        let decoded = try await onQueue { cancel -> (tokens: [TimedToken], steps: Int, joints: Int) in
            let tokens = try encoded.withUnsafeBufferPointer {
                try state.decoder.decode(encoder: $0.baseAddress ?? UnsafePointer(bitPattern: 1)!, validFrames: validFrames,
                                         frameOffset: frameOffset, params: params, chooser: chooser,
                                         piece: piece, cancel: cancel)
            }
            return (tokens, state.decoder.stepCalls, state.decoder.jointCalls)
        }
        let t3 = DispatchTime.now()
        timings.mel += ms(t0, t1); timings.encoder += ms(t1, t2); timings.decoder += ms(t2, t3)
        timings.steps += decoded.steps; timings.joints += decoded.joints
        return decoded.tokens
    }

    /// Runs CPU work on the lane's queue; the calling task's cancellation raises the flag the work polls.
    private func onQueue<T: Sendable>(_ work: @escaping @Sendable (CancelFlag) throws -> T) async throws -> T {
        let flag = CancelFlag()
        let queue = self.queue
        return try await withTaskCancellationHandler {
            try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<T, Error>) in
                queue.async { continuation.resume(with: Result { try work(flag) }) }
            }
        } onCancel: { flag.cancel() }
    }

    // MARK: 1.1.x cuts (up to 30 s: one piece; longer: quietest 200 ms in the last 4 s of each 30 s)

    private func transcribeCut(_ source: SampleSource, _ params: CPUTDTDecoder.Params, _ chooser: TokenChooser?,
                               _ timings: inout StageTimings,
                               _ progress: ProgressHandler?) async throws -> (text: String, tokens: [TimedToken], words: [TimedWord]) {
        let config = model.config, tokenizer = model.tokens
        let rate = config.sampleRate
        let total = source.sampleCount
        let maxSamples = config.maxWindowSeconds * rate
        let single = total <= maxSamples
        var texts: [String] = []
        var tokens: [TimedToken] = [], words: [TimedWord] = []
        var windows = 0
        var start = 0
        while start < total || (single && windows == 0) {
            try Task.checkCancellation()
            var end = min(total, start + maxSamples)
            if end < total {
                let low = max(start + maxSamples / 2, end - config.silenceSearchSeconds * rate)
                let region = try source.read(low..<end)
                end = low + LongFormPlan.quietestCentre(region, width: config.silenceWindowSeconds, rate: rate)
            }
            if single || end - start >= rate / 10 {
                let pcm = try source.read(start..<end)
                let piece = try await decodeWindow(pcm, frameOffset: start / config.samplesPerFrame, params: params,
                                                chooser: chooser, &timings)
                let text = tokenizer.decode(piece.map(\.id))
                if !text.isEmpty { texts.append(text) }
                tokens += piece
                words += tokenizer.words(from: piece)
                windows += 1
                progress?(TranscriptionProgress(processedSamples: end, totalSamples: total, windows: windows))
            }
            if end >= total { break }
            start = end
        }
        timings.pieces = windows
        return (texts.joined(separator: " ").trimmingCharacters(in: .whitespacesAndNewlines), tokens, words)
    }

    // MARK: overlapped windows merged by the longest common word subsequence

    private func transcribeLCS(_ source: SampleSource, _ o: LongFormOptions, _ params: CPUTDTDecoder.Params,
                               _ chooser: TokenChooser?,
                               _ timings: inout StageTimings,
                               _ progress: ProgressHandler?) async throws -> (text: String, tokens: [TimedToken], words: [TimedWord]) {
        let config = model.config, tokenizer = model.tokens
        let fps = 1 / config.secondsPerFrame
        let W = LongFormPlan.windowFrames(o, framesPerSecond: fps)
        let ov = max(0, Int((o.overlap * fps).rounded()))
        let plan = LongFormPlan.lcsWindows(totalSamples: source.sampleCount, samplesPerFrame: config.samplesPerFrame,
                                           windowFrames: W, overlapFrames: ov)
        var merged: [TimedToken] = []
        for (n, w) in plan.enumerated() {
            try Task.checkCancellation()
            let pcm = try source.read(w.startSample..<w.endSample)
            let window = try await decodeWindow(pcm, frameOffset: w.startFrame, params: params, chooser: chooser,
                                                &timings)
            merged = merged.isEmpty ? window : LongFormMerge.merge(merged, window, overlapStart: w.startFrame,
                                                                  overlapEnd: w.previousEndFrame, piece: { tokenizer.piece(for: $0) })
            progress?(TranscriptionProgress(processedSamples: w.endSample, totalSamples: source.sampleCount, windows: n + 1))
        }
        timings.pieces = plan.count
        return (tokenizer.decode(merged.map(\.id)), merged, tokenizer.words(from: merged))
    }
}

func ms(_ a: DispatchTime, _ b: DispatchTime) -> Double {
    Double(b.uptimeNanoseconds - a.uptimeNanoseconds) / 1e6
}

extension MelFrontEnd.Result: @unchecked Sendable {}
