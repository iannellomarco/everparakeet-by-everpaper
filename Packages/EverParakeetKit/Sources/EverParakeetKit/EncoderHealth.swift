import Foundation

// Self-healing for an encoder that was prepared for the wrong compute unit.
//
// Core AI specialises `encoder.aimodel` for the Neural Engine the first time it is loaded and caches the result. When the
// Neural Engine compiler service cannot be reached at that moment (NSCocoaErrorDomain 4097, "connection to service named
// com.apple.ANECompilerService"), Core AI does not fail: it quietly caches a plan that is not on the Neural Engine, and
// every later launch reuses it. Core AI exposes no way to ask which unit a plan uses, so the only evidence is time. This
// file holds the pure decision logic (probe, repair once, remember the outcome per OS build); the Core AI target plugs
// the real specialisation and cache deletion into it, and the unit tests plug in fakes.

/// Gate an app flips to say whether a repair may run now. A repair deletes the cached plan and specialises again, which
/// takes minutes and needs the Neural Engine compiler service, so it must not run while the app is in the background
/// (the likely cause of a bad plan in the first place) or while the user is transcribing live.
public final class RepairGate: @unchecked Sendable {
    private let lock = NSLock()
    private var allowed: Bool
    public init(allowed: Bool = true) { self.allowed = allowed }
    /// true: a repair may start (and its result be trusted). Set it from the app's scene phase and recording state.
    public var isAllowed: Bool {
        get { lock.lock(); defer { lock.unlock() }; return allowed }
        set { lock.lock(); allowed = newValue; lock.unlock() }
    }
}

/// How an encoder that may be on the wrong compute unit is checked and repaired. Used by `CoreAIEncoderFactory`.
public struct EncoderHealthPolicy: Sendable {
    /// Probe the encoder after the load. false = 1.1.1 behaviour (no probe, no repair).
    public var enabled = true
    /// The smallest window (5 s, 64 frames) takes 20 to 35 ms on the Neural Engine of an iPhone 17 Pro and about 490 ms on
    /// the plan that was cached after a failed specialisation, whatever the window size. Slower than this (best of
    /// `measuredRuns`, so one hiccup does not decide) counts as "not on the Neural Engine": about 4x the slowest
    /// Neural Engine figure measured, and far below the bad plan.
    public var slowThresholdMilliseconds = 150.0
    /// Runs of the smallest window on silence that are not timed (the first one pays for lazy setup).
    public var warmupRuns = 1
    /// Timed runs; the fastest is the probe's result.
    public var measuredRuns = 3
    /// Delete this model's Core AI cache entries and specialise once more when the probe fails. false = only report.
    public var repair = true
    /// Core AI will not delete a cache entry while a model made from it is alive. The slow encoder is released before the
    /// delete, which is tried this many times, `purgeRetryDelaySeconds` apart (the release can finish a moment later).
    public var purgeAttempts = 6
    public var purgeRetryDelaySeconds = 0.5
    /// Repair even when a repair already failed on this OS build (the remembered outcome is ignored).
    public var forceRepair = false
    /// The app controls this: a repair only starts, and a "still slow" result is only remembered, while it is allowed.
    public var gate = RepairGate()
    /// Where the outcome per OS build is remembered. nil = `Application Support/EverParakeet/coreai-health.json`.
    public var memoryFile: URL? = nil
    /// Remember outcomes so a repair that did not help is not retried at every launch of the same OS build.
    public var rememberOutcome = true
    public init() {}
}

/// What the load reports about the encoder while it specialises, probes and repairs (`LoadProgress.encoder`).
public enum EncoderStatus: Sendable, Equatable {
    /// Explicit specialisation (through the Core AI cache) started.
    case specialising
    /// Specialisation finished; a cache hit takes well under a second, a first specialisation minutes.
    case specialised(seconds: Double)
    /// Specialisation threw. `message` is Core AI's own error text.
    case specialisationFailed(message: String)
    /// The speed probe started.
    case probing
    /// The probe result: the fastest of the timed runs of the smallest window.
    case probed(milliseconds: Double, healthy: Bool)
    /// The cached plan is being deleted and the model specialised again. `reason` says why (probe time or Core AI's error).
    case repairing(reason: String)
    /// The encoder is on the Neural Engine. `repaired` is true when it took a repair to get there.
    case healthy(milliseconds: Double, repaired: Bool)
    /// The encoder runs, but not on the Neural Engine, and a repair was tried or is not possible. An app should use
    /// another encoder (Core ML). `repairTried` is false when the policy has repair off or a repair already failed on this
    /// OS build (so none was attempted this time).
    case notOnNeuralEngine(milliseconds: Double, repairTried: Bool)
    /// The encoder is slow but a repair is not allowed right now (`RepairGate`). Load again when it is.
    case repairDeferred(milliseconds: Double)
}

/// The final verdict on an encoder, `EverParakeetModel.encoderHealth`.
public struct EncoderHealth: Sendable, Equatable {
    public enum State: String, Sendable, Equatable {
        /// Probed and fast.
        case healthy
        /// Was slow, repaired by deleting the cached plan and specialising again, and is fast now.
        case repaired
        /// Slow after a repair, or slow with repair off or already failed on this OS build.
        case notOnNeuralEngine
        /// Slow, and a repair was not allowed at the time.
        case repairDeferred
    }
    public var state: State
    /// Fastest timed run of the smallest window (ms) at the end.
    public var probeMilliseconds: Double?
    /// Seconds the (last) specialisation took.
    public var specialisationSeconds: Double?
    /// One line for a log or the screen.
    public var detail: String
    public var onNeuralEngine: Bool { state == .healthy || state == .repaired }
    public init(state: State, probeMilliseconds: Double? = nil, specialisationSeconds: Double? = nil, detail: String = "") {
        self.state = state; self.probeMilliseconds = probeMilliseconds
        self.specialisationSeconds = specialisationSeconds; self.detail = detail
    }
}

extension EncoderBackend {
    /// nil for an encoder that is not checked (Core ML, custom backends).
    public var health: EncoderHealth? { nil }
}

// MARK: - Outcome memory

/// What a previous load learned on one OS build.
struct HealthRecord: Codable, Equatable {
    enum Outcome: String, Codable { case healthy, repaired, stillSlow, failed }
    var outcome: Outcome
    var probeMilliseconds: Double?
    var date: Date
}

/// `coreai-health.json`: outcomes keyed by "<OS build>|<asset key>". Small, atomic, never throws to the caller.
struct HealthMemory {
    let file: URL?
    static let lock = NSLock()
    static let maxEntries = 8

    static func defaultFile() -> URL {
        let fm = FileManager.default
        return (fm.urls(for: .applicationSupportDirectory, in: .userDomainMask).first ?? fm.temporaryDirectory)
            .appendingPathComponent("EverParakeet/coreai-health.json")
    }

    private func read() -> [String: HealthRecord] {
        guard let file, let data = try? Data(contentsOf: file) else { return [:] }
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .secondsSince1970
        return (try? decoder.decode([String: HealthRecord].self, from: data)) ?? [:]
    }

    func record(for key: String) -> HealthRecord? {
        Self.lock.lock(); defer { Self.lock.unlock() }
        return read()[key]
    }

    func store(_ record: HealthRecord, for key: String) {
        guard let file else { return }
        Self.lock.lock(); defer { Self.lock.unlock() }
        var table = read()
        table[key] = record
        if table.count > Self.maxEntries {
            for (k, _) in table.sorted(by: { $0.value.date < $1.value.date }).prefix(table.count - Self.maxEntries) { table[k] = nil }
        }
        let encoder = JSONEncoder()
        encoder.dateEncodingStrategy = .secondsSince1970
        guard let data = try? encoder.encode(table) else { return }
        try? FileManager.default.createDirectory(at: file.deletingLastPathComponent(), withIntermediateDirectories: true)
        try? data.write(to: file, options: .atomic)
    }
}

/// The OS build string (for example "24A335"), the key under which outcomes are remembered.
package func currentOSBuild() -> String {
    var size = 0
    guard sysctlbyname("kern.osversion", nil, &size, nil, 0) == 0, size > 0 else { return ProcessInfo.processInfo.operatingSystemVersionString }
    var buffer = [CChar](repeating: 0, count: size)
    guard sysctlbyname("kern.osversion", &buffer, &size, nil, 0) == 0 else { return ProcessInfo.processInfo.operatingSystemVersionString }
    return String(cString: buffer)
}

// MARK: - Probe

/// Times the smallest window on silence: `warmupRuns` untimed runs, then `measuredRuns` timed ones. Returns the fastest in
/// milliseconds. The cost is a few runs of the smallest window (a hundred milliseconds on the Neural Engine, about 2 s on
/// the bad plan).
package func probeEncoder(_ encoder: EncoderBackend, nMels: Int, policy: EncoderHealthPolicy) async throws -> Double {
    guard let window = encoder.windows.min() else { throw EverParakeetError.missing("encoder window to probe") }
    let frames = 8 * window
    let mel = [Float](repeating: 0, count: nMels * frames)
    let inputs = try encoder.makeInputs()
    for _ in 0..<max(0, policy.warmupRuns) {
        try Task.checkCancellation()
        _ = try await encoder.encode(mel: mel, frames: frames, valid: frames - 8, window: window, inputs: inputs)
    }
    var best = Double.infinity
    for _ in 0..<max(1, policy.measuredRuns) {
        try Task.checkCancellation()
        let start = DispatchTime.now().uptimeNanoseconds
        _ = try await encoder.encode(mel: mel, frames: frames, valid: frames - 8, window: window, inputs: inputs)
        best = min(best, Double(DispatchTime.now().uptimeNanoseconds - start) / 1e6)
    }
    return best
}

// MARK: - Decision logic

/// Prepares an encoder and can delete what the platform cached for it. Core AI implements it with `AIModel.specialize`
/// and `AIModelCache.deleteEntries(for:)`; the tests use a fake.
package protocol EncoderPreparer {
    /// Specialises (through the platform cache) and loads every window. Throws the platform's error.
    func prepare() async throws -> EncoderBackend
    /// Deletes this model's cached specialisation.
    func purgeCache() throws
}

/// Specialise, probe, repair once, remember. Pure of any framework: everything it touches comes in through `EncoderPreparer`,
/// the probe closure, the policy and the clock.
package struct EncoderHealer {
    package var policy: EncoderHealthPolicy
    /// "<OS build>|<asset key>"
    package var memoryKey: String
    package var status: @Sendable (EncoderStatus) -> Void
    /// Fastest timed run of the smallest window (ms) of an encoder. Defaults to the real probe.
    package var measure: (EncoderBackend) async throws -> Double
    package var now: () -> Double = { Double(DispatchTime.now().uptimeNanoseconds) / 1e9 }

    package init(policy: EncoderHealthPolicy, memoryKey: String, status: @escaping @Sendable (EncoderStatus) -> Void,
                 measure: @escaping (EncoderBackend) async throws -> Double,
                 now: @escaping () -> Double = { Double(DispatchTime.now().uptimeNanoseconds) / 1e9 }) {
        self.policy = policy; self.memoryKey = memoryKey; self.status = status; self.measure = measure; self.now = now
    }

    private var memory: HealthMemory {
        HealthMemory(file: policy.rememberOutcome ? (policy.memoryFile ?? HealthMemory.defaultFile()) : nil)
    }

    private struct Attempt {
        var encoder: EncoderBackend?
        var error: Error?
        var milliseconds: Double?
        var seconds: Double
        var healthy: Bool { encoder != nil && (milliseconds ?? .infinity) < policyThreshold }
        var policyThreshold: Double
    }

    private func attempt(_ preparer: EncoderPreparer) async throws -> Attempt {
        var result = Attempt(encoder: nil, error: nil, milliseconds: nil, seconds: 0, policyThreshold: policy.slowThresholdMilliseconds)
        status(.specialising)
        let start = now()
        do {
            result.encoder = try await preparer.prepare()
        } catch is CancellationError {
            throw CancellationError()
        } catch {
            result.seconds = now() - start
            result.error = error
            status(.specialisationFailed(message: "\(error)"))
            return result
        }
        result.seconds = now() - start
        status(.specialised(seconds: result.seconds))
        try Task.checkCancellation()
        status(.probing)
        let ms = try await measure(result.encoder!)
        result.milliseconds = ms
        status(.probed(milliseconds: ms, healthy: ms < policy.slowThresholdMilliseconds))
        return result
    }

    private func reason(_ a: Attempt) -> String {
        if let e = a.error { return "specialisation failed: \(e)" }
        return String(format: "the smallest window took %.0f ms (limit %.0f ms)", a.milliseconds ?? 0, policy.slowThresholdMilliseconds)
    }

    private func finish(_ a: Attempt, state: EncoderHealth.State, detail: String) -> EncoderHealth {
        EncoderHealth(state: state, probeMilliseconds: a.milliseconds, specialisationSeconds: a.seconds, detail: detail)
    }

    /// - Returns: the encoder (possibly slow: check `health.onNeuralEngine`) and the verdict. Throws when no encoder could be
    ///   prepared at all, or on cancellation.
    package func run(_ preparer: EncoderPreparer) async throws -> (encoder: EncoderBackend, health: EncoderHealth) {
        var first = try await attempt(preparer)
        if first.healthy {
            note(.healthy, first)
            status(.healthy(milliseconds: first.milliseconds!, repaired: false))
            return (first.encoder!, finish(first, state: .healthy, detail: "on the Neural Engine"))
        }
        let why = reason(first)
        let prior = memory.record(for: memoryKey)
        let alreadyFailed = (prior?.outcome == .stillSlow || prior?.outcome == .failed) && !policy.forceRepair

        // No repair: say so and hand back what we have (or the original error when there is nothing).
        if !policy.repair || alreadyFailed {
            guard let encoder = first.encoder else { throw first.error ?? EverParakeetError.invalid("encoder") }
            let tried = alreadyFailed
            status(.notOnNeuralEngine(milliseconds: first.milliseconds ?? 0, repairTried: tried))
            let detail = tried ? "\(why); a repair already failed on this OS build, so none was tried"
                               : "\(why); repair is off"
            return (encoder, finish(first, state: .notOnNeuralEngine, detail: detail))
        }
        // Not now: the app is in the background or busy (the gate), so leave the cached plan alone.
        guard policy.gate.isAllowed else {
            guard let encoder = first.encoder else { throw first.error ?? EverParakeetError.invalid("encoder") }
            status(.repairDeferred(milliseconds: first.milliseconds ?? 0))
            return (encoder, finish(first, state: .repairDeferred, detail: "\(why); a repair is not allowed right now"))
        }

        // Repair, once: delete this model's cache entries, specialise again, probe again.
        status(.repairing(reason: why))
        // Core AI refuses to delete a cache entry while a model made from it is alive ("Deletion could not be completed,
        // assets still in use", seen on an iPhone 17 Pro), so the slow encoder is released first and the delete is retried
        // for a moment, because the release can complete a little after the last reference is gone.
        let firstMilliseconds = first.milliseconds
        let nothingLoaded = first.encoder == nil
        first.encoder = nil
        var purgeError: Error?
        for tryNumber in 0..<(nothingLoaded ? 1 : max(1, policy.purgeAttempts)) {
            do { try preparer.purgeCache(); purgeError = nil; break } catch {
                purgeError = error
                if tryNumber + 1 < policy.purgeAttempts, !nothingLoaded {
                    try await Task.sleep(nanoseconds: UInt64(max(0, policy.purgeRetryDelaySeconds) * 1e9))
                }
            }
        }
        if let purgeError, !nothingLoaded {
            // The cached plan cannot be deleted, so specialising again would only load it again. Hand back a working
            // (slow) encoder and say so; nothing is remembered, the next launch tries again.
            let encoder: EncoderBackend
            do { encoder = try await preparer.prepare() } catch is CancellationError { throw CancellationError() } catch { throw error }
            status(.notOnNeuralEngine(milliseconds: firstMilliseconds ?? 0, repairTried: false))
            return (encoder, EncoderHealth(state: .notOnNeuralEngine, probeMilliseconds: firstMilliseconds,
                                           detail: "\(why); the cached plan could not be deleted: \(purgeError)"))
        }
        let second = try await attempt(preparer)
        if second.healthy {
            note(.repaired, second)
            status(.healthy(milliseconds: second.milliseconds!, repaired: true))
            return (second.encoder!, finish(second, state: .repaired, detail: "repaired: \(why) before, \(Int(second.milliseconds!.rounded())) ms after"))
        }
        // Still slow, or failing again. A background spell during the repair makes the result inconclusive: do not remember it.
        if policy.gate.isAllowed { note(second.encoder == nil ? .failed : .stillSlow, second) }
        guard let encoder = second.encoder else { throw second.error ?? first.error ?? EverParakeetError.invalid("encoder") }
        status(.notOnNeuralEngine(milliseconds: second.milliseconds ?? 0, repairTried: true))
        return (encoder, finish(second, state: .notOnNeuralEngine, detail: "still slow after a repair: \(reason(second))"))
    }

    private func note(_ outcome: HealthRecord.Outcome, _ a: Attempt) {
        guard policy.rememberOutcome else { return }
        memory.store(HealthRecord(outcome: outcome, probeMilliseconds: a.milliseconds, date: Date()), for: memoryKey)
    }
}
