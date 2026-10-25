import Foundation
import Observation
import UIKit
import EverParakeetKit
#if canImport(CoreAI)
import EverParakeetCoreAI
#endif

/// What one transcription produced, with its measured numbers.
struct Transcript: Codable, Identifiable, Hashable {
    var id = UUID()
    var date = Date()
    var source: String          // "Microphone", "Sample: ...", file name
    var text: String
    var audioSeconds: Double
    var computeMs: Double
    var backend: String
    var diag: Diag?             // per-stage timings of this run (not shown in the UI)

    var realtimeFactor: Double { computeMs > 0 ? audioSeconds / (computeMs / 1000) : 0 }
}

/// Where the time of one transcription went, as the package measured it (ms).
struct Diag: Codable, Hashable {
    var mel = 0.0, encoder = 0.0, decoder = 0.0
    var windows = 0
}

/// Loads the EverParakeet model from the app bundle (Core AI encoder on the Neural Engine; Core ML only when Core AI is
/// not on the Neural Engine on this iPhone even after the package repaired it, or does not load) and runs
/// transcriptions on one lane of it.
@MainActor @Observable
final class EngineHost {
    enum Phase: Equatable { case idle, preparing, ready, failed(String) }

    var phase: Phase = .idle
    var elapsed = 0.0
    var stage = ""                  // what the loader is doing right now
    var compiling = false           // the loader reported that it is preparing the encoder for this device
    var repairing = false           // the package found a plan that is not on the Neural Engine and is redoing it
    var backend = ""                // "Core AI" or "Core ML"
    var note = ""                   // why the fallback is in use, if it is
    var loadMs = 0.0
    var health: EncoderHealth?      // the package's speed probe of the Core AI encoder (nil on Core ML)
    var firstLaunch: Bool
    private var model: EverParakeetModel?
    private var lane: EverParakeetLane?
    private var options = TranscribeOptions()   // package defaults: LCS merge, 8 s overlap, blank bias 1.25

    /// The package repairs a bad Neural Engine plan only while this gate is open: the app is in front and nobody is
    /// recording. A repair needs the Neural Engine compiler service, which iOS does not give to an app in the background.
    @ObservationIgnored let gate = RepairGate(allowed: false)
    private var appActive = false
    private var recording = false

    private static let preparedKey = "everparakeet.prepared"

    init() { firstLaunch = !UserDefaults.standard.bool(forKey: Self.preparedKey) }

    func setActive(_ active: Bool) { appActive = active; updateGate() }
    func setRecording(_ on: Bool) { recording = on; updateGate() }
    private func updateGate() { gate.isAllowed = appActive && !recording }

    /// The folder fetched by scripts/fetch-model.sh and copied into the app: it holds `models/` and `tokenizer/`.
    static var modelRoot: URL { Bundle.main.resourceURL!.appendingPathComponent("Model", isDirectory: true) }
    static var modelPresent: Bool {
        FileManager.default.fileExists(atPath: modelRoot.appendingPathComponent("models/config.json").path)
    }

    func prepare() async {
        switch phase {
        case .idle, .failed: break
        default: return
        }
        guard Self.modelPresent else {
            phase = .failed("The model files are not in the app. Run scripts/fetch-model.sh, then build and run again.")
            return
        }
        phase = .preparing
        stage = "Reading the model files"
        let started = Date()
        let ticker = Task { @MainActor in
            while !Task.isCancelled {
                elapsed = Date().timeIntervalSince(started)
                try? await Task.sleep(for: .milliseconds(250))
            }
        }
        // A repair takes minutes in front: do not let the screen lock and send the app to the background half way.
        UIApplication.shared.isIdleTimerDisabled = true
        defer { ticker.cancel(); UIApplication.shared.isIdleTimerDisabled = false }
        do {
            let engine = try await bringUp(preferCoreAI: !Self.forceCoreML, options: options)
            model = engine.model
            lane = engine.lane
            backend = engine.backend
            health = engine.health
            if !engine.note.isEmpty { note = engine.note }
            loadMs = Date().timeIntervalSince(started) * 1000
            // "Prepared" (the long first-launch screen is no longer needed) only after a healthy probe: Core AI confirmed on
            // the Neural Engine, or a conclusive verdict that this iPhone is better served by Core ML.
            if engine.settled { UserDefaults.standard.set(true, forKey: Self.preparedKey) }
            phase = .ready
        } catch {
            phase = .failed("\(error)")
        }
    }

    private struct Engine {
        var model: EverParakeetModel
        var lane: EverParakeetLane
        var backend: String
        var note = ""
        var health: EncoderHealth?
        var settled = false     // the encoder choice is final (a healthy probe, or Core ML after a conclusive verdict)
    }

    /// Loads the model on Core AI, which probes its own speed and repairs a plan that is not on the Neural Engine
    /// (see `EncoderHealthPolicy`). Core ML is used only when the package reports that Core AI is not on the Neural
    /// Engine even after the repair, or when Core AI does not load at all.
    private func bringUp(preferCoreAI: Bool, options: TranscribeOptions) async throws -> Engine {
        var loaded: EverParakeetModel?
        var backend = "Core ML", note = "", health: EncoderHealth?, settled = true
        if preferCoreAI {
            do {
                var model: EverParakeetModel? = try await load(coreAI: true)
                // The app was not in front when the probe failed: nothing was deleted. Wait for the foreground, load again.
                // The slow model is dropped first: Core AI refuses to delete a cache entry while a model made from it is
                // alive ("assets still in use"), so keeping it would make the repair of the second load fail.
                var retries = 0
                while model?.encoderHealth?.state == .repairDeferred, retries < 3 {
                    model = nil
                    stage = "Waiting for the app to be in front"
                    while !gate.isAllowed { try Task.checkCancellation(); try await Task.sleep(for: .milliseconds(500)) }
                    model = try await load(coreAI: true)
                    retries += 1
                }
                health = model?.encoderHealth
                if health?.onNeuralEngine != false {
                    loaded = model
                    backend = "Core AI"
                } else if let h = health {
                    let ms = Int((h.probeMilliseconds ?? 0).rounded())
                    if h.state == .repairDeferred {
                        note = "The Neural Engine plan of this iPhone could not be repaired just now. Core ML is in use for this launch."
                        settled = false
                    } else {
                        note = "Core AI is not on the Neural Engine on this iPhone (\(ms) ms for the smallest window, after a repair), so Core ML is in use."
                    }
                }
            } catch is CancellationError {
                throw CancellationError()
            } catch {
                note = "Core AI did not load (\(error)). Using Core ML."
                #if canImport(CoreAI)
                settled = false
                #endif
            }
        }
        if loaded == nil {
            stage = "Loading the Core ML encoder"
            loaded = try await load(coreAI: false)
        } else {
            settled = health?.onNeuralEngine == true
        }
        let lane = try loaded!.makeLane()
        // Run every encoder window and the long-form path once, so the first real transcription is not the one that
        // pays for lazy setup. (A single second of silence only runs the 5 s window.)
        stage = "Warming up"
        try await Self.warmUp(lane, options: options)
        return Engine(model: loaded!, lane: lane, backend: backend, note: note, health: health, settled: settled)
    }

    /// `-coreml` launch argument (debug builds): skip Core AI.
    private static var forceCoreML: Bool {
        #if DEBUG
        CommandLine.arguments.contains("-coreml")
        #else
        false
        #endif
    }

    /// Decodes silence that routes to each encoder window (about 5, 10, 15 and 30 s) and a clip over 30 s that takes the
    /// overlapped long-form path, then one pass over the bundled long clip, which is real speech, so the decoder's
    /// weights and the long-form merge have run once too. Takes well under a second on a phone.
    nonisolated private static func warmUp(_ lane: EverParakeetLane, options: TranscribeOptions) async throws {
        for seconds in [4.5, 9.5, 14.5, 29.5, 34.0] {
            _ = try await lane.transcribe(pcm16k: [Float](repeating: 0, count: Int(seconds * 16000)), options: options)
        }
        guard let url = Bundle.main.url(forResource: "long_italian", withExtension: "wav", subdirectory: "Samples"),
              let speech = try? await AudioFile.load(url) else { return }
        _ = try? await lane.transcribe(pcm16k: speech, options: options)
    }

    private func load(coreAI: Bool) async throws -> EverParakeetModel {
        var load = EverParakeetModel.LoadOptions()
        #if canImport(CoreAI)
        if coreAI {
            var policy = EncoderHealthPolicy()
            policy.gate = gate
            #if DEBUG
            if CommandLine.arguments.contains("-force-repair") { policy.forceRepair = true }   // ignore a remembered failed repair
            #endif
            load.encoderFactory = CoreAIEncoderFactory(health: policy)
        }
        #else
        // The iOS Simulator SDK has no Core AI: only the Core ML encoder is available there.
        if coreAI { throw EverParakeetError.missing("Core AI is not available in this SDK") }
        #endif
        load.progress = { [weak self] event in
            Task { @MainActor in self?.report(event) }
        }
        return try await EverParakeetModel.load(root: Self.modelRoot, options: load)
    }

    private func report(_ event: LoadProgress) {
        #if DEBUG
        AutoTest.say("ENGINE \(event)")
        #endif
        switch event {
        case .loadingDecoder: stage = "Reading the decoder"
        case .compiling: compiling = true; if !repairing { stage = "Preparing the encoder for this iPhone" }
        case .loadingFunction(_, let index, let count): if !repairing { stage = "Loading encoder window \(index + 1) of \(count)" }
        case .ready: stage = "Ready"
        case .encoder(let status): report(status)
        }
    }

    private func report(_ status: EncoderStatus) {
        switch status {
        case .specialising: stage = repairing ? "Repairing the model preparation" : "Preparing the encoder for this iPhone"
        case .specialised: break
        case .specialisationFailed: stage = "Trying the encoder preparation again"
        case .probing: stage = "Checking the Neural Engine"
        case .probed: break
        case .repairing: repairing = true; compiling = true; stage = "Repairing the model preparation"
        case .healthy: repairing = false; stage = "Neural Engine confirmed"
        case .notOnNeuralEngine: repairing = false; stage = "Using the Core ML encoder"
        case .repairDeferred: stage = "Waiting for the app to be in front"
        }
    }

    /// Transcribes 16 kHz mono PCM. Audio over 30 s goes through the long-form path (LCS merge, 8 s overlap,
    /// blank bias 1.25, repetition guard).
    func transcribe(_ pcm: [Float], source: String,
                    progress: (@Sendable (Double) -> Void)? = nil) async throws -> Transcript {
        guard let lane else { throw EverParakeetError.missing("model not loaded") }
        let (result, ms) = try await Self.timed(lane, pcm, options, progress.map { report in
            { @Sendable p in report(p.fraction) }
        })
        let t = result.timings
        return Transcript(source: source, text: result.text, audioSeconds: Double(pcm.count) / 16000,
                          computeMs: ms, backend: backend,
                          diag: Diag(mel: t.mel, encoder: t.encoder, decoder: t.decoder, windows: t.pieces))
    }

    /// The transcription and its wall time, both taken off the main actor: the clock stops when the lane returns, not
    /// when the main thread is next free to resume this method (it can be busy drawing the result).
    nonisolated private static func timed(_ lane: EverParakeetLane, _ pcm: [Float], _ options: TranscribeOptions,
                                          _ progress: ProgressHandler?) async throws -> (Transcription, Double) {
        let t0 = DispatchTime.now()
        let result = try await lane.transcribe(pcm16k: pcm, options: options, progress: progress)
        return (result, Double(DispatchTime.now().uptimeNanoseconds - t0.uptimeNanoseconds) / 1e6)
    }

    /// Plain decode used by the live view: no progress hook, no record.
    func decode(_ pcm: [Float]) async -> String {
        guard let lane else { return "" }
        return (try? await lane.transcribe(pcm16k: pcm, options: options))?.text ?? ""
    }
}
