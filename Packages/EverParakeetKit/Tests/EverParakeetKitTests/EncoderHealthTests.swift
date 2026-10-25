import XCTest
@testable import EverParakeetKit

/// An encoder whose "speed" is a number the test chooses (the probe closure reads it back).
final class TimedEncoder: EncoderBackend, @unchecked Sendable {
    let windows: [Int]
    let milliseconds: Double
    var health: EncoderHealth?
    init(milliseconds: Double, windows: [Int] = [64, 192, 384]) { self.milliseconds = milliseconds; self.windows = windows }
    func makeInputs() throws -> EncoderInputs { FakeEncoder.Inputs(serial: 0) }
    func encode(mel: [Float], frames: Int, valid: Int, window: Int, inputs: EncoderInputs) async throws -> (enc: [Float], validFrames: Int) {
        (Array(repeating: 0, count: window * TestPackage.hidden), 1)
    }
}

/// A preparer that plays a script: each `prepare()` takes the next step (an encoder of a given speed, or an error).
final class ScriptedPreparer: EncoderPreparer, @unchecked Sendable {
    enum Step { case encoder(Double), failure(String), cancel }
    struct Failure: Error, CustomStringConvertible { let description: String }
    private let lock = NSLock()
    private var steps: [Step]
    private(set) var prepares = 0
    private(set) var purges = 0
    var purgeError: Error?
    /// The first this many purges fail (the release of the old model completing late).
    var rejectFirstPurges = 0
    /// Purges rejected because an encoder made by this preparer was still alive, like Core AI's "assets still in use".
    private(set) var rejectedForLiveEncoder = 0
    private var made: [() -> Bool] = []
    /// Runs inside `prepare()` (for example to close the repair gate while the "specialisation" runs).
    var onPrepare: ((Int) -> Void)?

    init(_ steps: [Step]) { self.steps = steps }

    func prepare() async throws -> EncoderBackend {
        lock.lock()
        prepares += 1
        let n = prepares
        let step = steps.isEmpty ? Step.failure("script exhausted") : steps.removeFirst()
        lock.unlock()
        onPrepare?(n)
        switch step {
        case .encoder(let ms):
            let encoder = TimedEncoder(milliseconds: ms)
            lock.lock(); made.append({ [weak encoder] in encoder != nil }); lock.unlock()
            return encoder
        case .failure(let message): throw Failure(description: message)
        case .cancel: throw CancellationError()
        }
    }

    func purgeCache() throws {
        lock.lock()
        purges += 1
        let alive = made.contains { $0() }
        let reject = purges <= rejectFirstPurges
        if alive { rejectedForLiveEncoder += 1 }
        lock.unlock()
        if alive { throw Failure(description: "Deletion could not be completed, assets still in use") }
        if reject { throw Failure(description: "release not finished yet") }
        if let purgeError { throw purgeError }
    }
}

final class StatusLog: @unchecked Sendable {
    private let lock = NSLock()
    private var items: [EncoderStatus] = []
    func add(_ s: EncoderStatus) { lock.lock(); items.append(s); lock.unlock() }
    var all: [EncoderStatus] { lock.lock(); defer { lock.unlock() }; return items }
    /// The statuses with the numbers removed, to compare the order of events.
    var shape: [String] {
        all.map {
            switch $0 {
            case .specialising: return "specialising"
            case .specialised: return "specialised"
            case .specialisationFailed: return "specialisationFailed"
            case .probing: return "probing"
            case .probed(_, let healthy): return healthy ? "probed-fast" : "probed-slow"
            case .repairing: return "repairing"
            case .healthy(_, let repaired): return repaired ? "healthy-repaired" : "healthy"
            case .notOnNeuralEngine(_, let tried): return tried ? "notOnNeuralEngine-tried" : "notOnNeuralEngine-untried"
            case .repairDeferred: return "repairDeferred"
            }
        }
    }
}

final class EncoderHealthTests: XCTestCase {
    var folder: URL!
    var memoryFile: URL!
    override func setUpWithError() throws {
        folder = try TestPackage.makeTemporaryDirectory("health")
        memoryFile = folder.appendingPathComponent("coreai-health.json")
    }
    override func tearDownWithError() throws { try? FileManager.default.removeItem(at: folder) }

    private func policy(repair: Bool = true, force: Bool = false, gate: RepairGate = RepairGate(), remember: Bool = true) -> EncoderHealthPolicy {
        var p = EncoderHealthPolicy()
        p.repair = repair; p.forceRepair = force; p.gate = gate; p.rememberOutcome = remember
        p.memoryFile = memoryFile
        p.purgeRetryDelaySeconds = 0.001
        return p
    }

    private func healer(_ p: EncoderHealthPolicy, key: String = "27A1|encoder.aimodel-1", log: StatusLog) -> EncoderHealer {
        EncoderHealer(policy: p, memoryKey: key, status: { log.add($0) }, measure: { ($0 as! TimedEncoder).milliseconds })
    }

    // MARK: decisions

    func testHealthyFirstTimeNeedsNoRepair() async throws {
        let prep = ScriptedPreparer([.encoder(25)])
        let log = StatusLog()
        let (encoder, health) = try await healer(policy(), log: log).run(prep)
        XCTAssertEqual((encoder as! TimedEncoder).milliseconds, 25)
        XCTAssertEqual(health.state, .healthy)
        XCTAssertTrue(health.onNeuralEngine)
        XCTAssertEqual(prep.prepares, 1)
        XCTAssertEqual(prep.purges, 0)
        XCTAssertEqual(log.shape, ["specialising", "specialised", "probing", "probed-fast", "healthy"])
        XCTAssertEqual(HealthMemory(file: memoryFile).record(for: "27A1|encoder.aimodel-1")?.outcome, .healthy)
    }

    func testThresholdIsExclusive() async throws {
        // 149 ms is healthy, exactly the limit is not (the limit is "slower than").
        let fast = try await healer(policy(remember: false), log: StatusLog()).run(ScriptedPreparer([.encoder(149)]))
        XCTAssertEqual(fast.health.state, .healthy)
        let prep = ScriptedPreparer([.encoder(150), .encoder(150)])
        let slow = try await healer(policy(remember: false), log: StatusLog()).run(prep)
        XCTAssertEqual(slow.health.state, .notOnNeuralEngine)
        XCTAssertEqual(prep.purges, 1)
    }

    func testSlowProbeDeletesTheCacheSpecialisesAgainAndUsesCoreAIWhenHealthy() async throws {
        let prep = ScriptedPreparer([.encoder(490), .encoder(27)])
        let log = StatusLog()
        let (encoder, health) = try await healer(policy(), log: log).run(prep)
        XCTAssertEqual((encoder as! TimedEncoder).milliseconds, 27)
        XCTAssertEqual(health.state, .repaired)
        XCTAssertTrue(health.onNeuralEngine)
        XCTAssertEqual(prep.prepares, 2)
        XCTAssertEqual(prep.purges, 1)
        XCTAssertEqual(log.shape, ["specialising", "specialised", "probing", "probed-slow", "repairing",
                                   "specialising", "specialised", "probing", "probed-fast", "healthy-repaired"])
        XCTAssertEqual(HealthMemory(file: memoryFile).record(for: "27A1|encoder.aimodel-1")?.outcome, .repaired)
        XCTAssertTrue(health.detail.contains("490"))
    }

    func testStillSlowAfterRepairIsReportedAndRemembered() async throws {
        let prep = ScriptedPreparer([.encoder(490), .encoder(480)])
        let log = StatusLog()
        let (encoder, health) = try await healer(policy(), log: log).run(prep)
        XCTAssertEqual((encoder as! TimedEncoder).milliseconds, 480)
        XCTAssertEqual(health.state, .notOnNeuralEngine)
        XCTAssertFalse(health.onNeuralEngine)
        XCTAssertEqual(prep.purges, 1)
        XCTAssertEqual(log.shape.last, "notOnNeuralEngine-tried")
        XCTAssertEqual(HealthMemory(file: memoryFile).record(for: "27A1|encoder.aimodel-1")?.outcome, .stillSlow)
    }

    func testARepairThatFailedIsNotRetriedOnTheNextLaunchOfTheSameOSBuild() async throws {
        _ = try await healer(policy(), log: StatusLog()).run(ScriptedPreparer([.encoder(490), .encoder(480)]))
        // Next launch, same build: the cached (slow) plan loads in no time, the probe fails, and nothing is deleted.
        let prep = ScriptedPreparer([.encoder(485)])
        let log = StatusLog()
        let (_, health) = try await healer(policy(), log: log).run(prep)
        XCTAssertEqual(health.state, .notOnNeuralEngine)
        XCTAssertEqual(prep.prepares, 1)
        XCTAssertEqual(prep.purges, 0)
        XCTAssertEqual(log.shape.last, "notOnNeuralEngine-tried")
        XCTAssertTrue(health.detail.contains("already failed"))
    }

    func testANewOSBuildTriesTheRepairAgain() async throws {
        _ = try await healer(policy(), key: "27A1|asset", log: StatusLog()).run(ScriptedPreparer([.encoder(490), .encoder(480)]))
        let prep = ScriptedPreparer([.encoder(485), .encoder(30)])
        let (_, health) = try await healer(policy(), key: "27A2|asset", log: StatusLog()).run(prep)
        XCTAssertEqual(health.state, .repaired)
        XCTAssertEqual(prep.purges, 1)
        // And the other build's memory is untouched.
        XCTAssertEqual(HealthMemory(file: memoryFile).record(for: "27A1|asset")?.outcome, .stillSlow)
        XCTAssertEqual(HealthMemory(file: memoryFile).record(for: "27A2|asset")?.outcome, .repaired)
    }

    func testForceRepairIgnoresTheMemory() async throws {
        _ = try await healer(policy(), log: StatusLog()).run(ScriptedPreparer([.encoder(490), .encoder(480)]))
        let prep = ScriptedPreparer([.encoder(485), .encoder(30)])
        let (_, health) = try await healer(policy(force: true), log: StatusLog()).run(prep)
        XCTAssertEqual(health.state, .repaired)
        XCTAssertEqual(prep.purges, 1)
    }

    func testAHealthyLaunchOverwritesARememberedFailure() async throws {
        _ = try await healer(policy(), log: StatusLog()).run(ScriptedPreparer([.encoder(490), .encoder(480)]))
        let (_, health) = try await healer(policy(), log: StatusLog()).run(ScriptedPreparer([.encoder(28)]))
        XCTAssertEqual(health.state, .healthy)
        XCTAssertEqual(HealthMemory(file: memoryFile).record(for: "27A1|encoder.aimodel-1")?.outcome, .healthy)
    }

    func testRepairOffOnlyReports() async throws {
        let prep = ScriptedPreparer([.encoder(490)])
        let log = StatusLog()
        let (_, health) = try await healer(policy(repair: false), log: log).run(prep)
        XCTAssertEqual(health.state, .notOnNeuralEngine)
        XCTAssertEqual(prep.purges, 0)
        XCTAssertEqual(prep.prepares, 1)
        XCTAssertEqual(log.shape.last, "notOnNeuralEngine-untried")
    }

    // MARK: the foreground gate

    func testNoRepairWhileTheGateIsClosed() async throws {
        let gate = RepairGate(allowed: false)
        let prep = ScriptedPreparer([.encoder(490)])
        let log = StatusLog()
        let (_, health) = try await healer(policy(gate: gate), log: log).run(prep)
        XCTAssertEqual(health.state, .repairDeferred)
        XCTAssertEqual(prep.purges, 0)
        XCTAssertEqual(log.shape.last, "repairDeferred")
        XCTAssertNil(HealthMemory(file: memoryFile).record(for: "27A1|encoder.aimodel-1"), "a deferred repair must not be remembered as a failure")
        // The app opens the gate and loads again: now it repairs.
        gate.isAllowed = true
        let again = ScriptedPreparer([.encoder(490), .encoder(26)])
        let (_, second) = try await healer(policy(gate: gate), log: StatusLog()).run(again)
        XCTAssertEqual(second.state, .repaired)
    }

    func testABackgroundSpellDuringTheRepairDoesNotCountAsAFailure() async throws {
        let gate = RepairGate()
        let prep = ScriptedPreparer([.encoder(490), .encoder(488)])
        prep.onPrepare = { n in if n == 2 { gate.isAllowed = false } }      // the app leaves the foreground during the repair
        let (_, health) = try await healer(policy(gate: gate), log: StatusLog()).run(prep)
        XCTAssertEqual(health.state, .notOnNeuralEngine)
        XCTAssertNil(HealthMemory(file: memoryFile).record(for: "27A1|encoder.aimodel-1"), "an inconclusive repair is not remembered")
    }

    // MARK: specialisation errors

    func testASpecialisationErrorIsReportedWithItsTextAndRepaired() async throws {
        let text = #"NSCocoaErrorDomain 4097 "connection to service named com.apple.ANECompilerService""#
        let prep = ScriptedPreparer([.failure(text), .encoder(30)])
        let log = StatusLog()
        let (_, health) = try await healer(policy(), log: log).run(prep)
        XCTAssertEqual(health.state, .repaired)
        XCTAssertEqual(prep.purges, 1)
        let failed = log.all.compactMap { s -> String? in if case .specialisationFailed(let m) = s { return m } else { return nil } }
        XCTAssertEqual(failed.count, 1)
        XCTAssertTrue(failed[0].contains("ANECompilerService"), "Core AI's own error text reaches the status channel")
        let reasons = log.all.compactMap { s -> String? in if case .repairing(let r) = s { return r } else { return nil } }
        XCTAssertTrue(reasons.first?.contains("ANECompilerService") == true)
    }

    func testTwoSpecialisationErrorsThrowTheLastAndAreRemembered() async throws {
        let prep = ScriptedPreparer([.failure("first"), .failure("second")])
        do {
            _ = try await healer(policy(), log: StatusLog()).run(prep)
            XCTFail("expected a throw")
        } catch {
            XCTAssertEqual("\(error)", "second")
        }
        XCTAssertEqual(HealthMemory(file: memoryFile).record(for: "27A1|encoder.aimodel-1")?.outcome, .failed)
        // Next launch: it throws the plain error without deleting anything again.
        let again = ScriptedPreparer([.failure("third")])
        do {
            _ = try await healer(policy(), log: StatusLog()).run(again)
            XCTFail("expected a throw")
        } catch {
            XCTAssertEqual("\(error)", "third")
        }
        XCTAssertEqual(again.purges, 0)
    }

    func testAnErrorWithTheGateClosedThrowsTheOriginalError() async throws {
        let prep = ScriptedPreparer([.failure("no service")])
        do {
            _ = try await healer(policy(gate: RepairGate(allowed: false)), log: StatusLog()).run(prep)
            XCTFail("expected a throw")
        } catch {
            XCTAssertEqual("\(error)", "no service")
        }
        XCTAssertEqual(prep.purges, 0)
    }

    func testAFailingRepairThrowsAndIsRemembered() async throws {
        // The slow encoder was released for the purge, so there is nothing to hand back when the second specialisation fails.
        let prep = ScriptedPreparer([.encoder(490), .failure("compiler service unavailable")])
        do {
            _ = try await healer(policy(), log: StatusLog()).run(prep)
            XCTFail("expected a throw")
        } catch {
            XCTAssertEqual("\(error)", "compiler service unavailable")
        }
        XCTAssertEqual(HealthMemory(file: memoryFile).record(for: "27A1|encoder.aimodel-1")?.outcome, .failed)
    }

    func testTheSlowEncoderIsReleasedBeforeTheCacheIsDeleted() async throws {
        // Core AI: "Deletion could not be completed, assets still in use" while a model made from the entry is alive.
        let prep = ScriptedPreparer([.encoder(490), .encoder(30)])
        let (_, health) = try await healer(policy(), log: StatusLog()).run(prep)
        XCTAssertEqual(health.state, .repaired)
        XCTAssertEqual(prep.rejectedForLiveEncoder, 0)
        XCTAssertEqual(prep.purges, 1)
    }

    func testThePurgeIsRetriedUntilTheReleaseCompletes() async throws {
        let prep = ScriptedPreparer([.encoder(490), .encoder(30)])
        prep.rejectFirstPurges = 2
        let (_, health) = try await healer(policy(), log: StatusLog()).run(prep)
        XCTAssertEqual(health.state, .repaired)
        XCTAssertEqual(prep.purges, 3)
        XCTAssertEqual(prep.prepares, 2)
    }

    func testAPurgeThatNeverWorksHandsBackAWorkingEncoderAndRemembersNothing() async throws {
        // Specialising again would only load the same cached plan, so it is not done; the encoder comes from one more load.
        let prep = ScriptedPreparer([.encoder(490), .encoder(488)])
        prep.purgeError = ScriptedPreparer.Failure(description: "no entry")
        let log = StatusLog()
        var p = policy()
        p.purgeAttempts = 3
        let (encoder, health) = try await healer(p, log: log).run(prep)
        XCTAssertEqual((encoder as! TimedEncoder).milliseconds, 488)
        XCTAssertEqual(health.state, .notOnNeuralEngine)
        XCTAssertEqual(prep.purges, 3)
        XCTAssertEqual(prep.prepares, 2)
        XCTAssertEqual(log.shape.last, "notOnNeuralEngine-untried")
        XCTAssertTrue(health.detail.contains("no entry"))
        XCTAssertNil(HealthMemory(file: memoryFile).record(for: "27A1|encoder.aimodel-1"), "an unfinished repair is not a failed one")
    }

    func testAPurgeErrorAfterASpecialisationFailureDoesNotStopTheRepair() async throws {
        // Nothing was loaded, so there may be no cache entry to delete: one delete attempt, then specialise again anyway.
        let prep = ScriptedPreparer([.failure("no service"), .encoder(30)])
        prep.purgeError = ScriptedPreparer.Failure(description: "no entry")
        let (_, health) = try await healer(policy(), log: StatusLog()).run(prep)
        XCTAssertEqual(health.state, .repaired)
        XCTAssertEqual(prep.purges, 1)
    }

    func testCancellationIsNotARepairTrigger() async throws {
        let prep = ScriptedPreparer([.cancel])
        do {
            _ = try await healer(policy(), log: StatusLog()).run(prep)
            XCTFail("expected a throw")
        } catch is CancellationError {
        }
        XCTAssertEqual(prep.purges, 0)
        XCTAssertEqual(prep.prepares, 1)
    }

    // MARK: the probe

    func testProbeUsesTheSmallestWindowWithWarmupAndReturnsTheFastestRun() async throws {
        let encoder = FakeEncoder(windows: [192, 64, 384])
        var p = EncoderHealthPolicy()
        p.warmupRuns = 2; p.measuredRuns = 3
        encoder.delay = 0.001
        let ms = try await probeEncoder(encoder, nMels: 128, policy: p)
        XCTAssertEqual(encoder.calls.count, 5)
        XCTAssertTrue(encoder.calls.allSatisfy { $0.window == 64 })
        XCTAssertEqual(Set(encoder.calls.map { $0.inputs }).count, 1, "one set of input buffers for the whole probe")
        XCTAssertGreaterThan(ms, 0)
        XCTAssertLessThan(ms, 150)
    }

    func testProbeSeesASlowEncoder() async throws {
        let encoder = FakeEncoder(windows: [64, 192])
        encoder.delay = 0.2
        var p = EncoderHealthPolicy()
        p.warmupRuns = 0; p.measuredRuns = 1
        let ms = try await probeEncoder(encoder, nMels: 128, policy: p)
        XCTAssertGreaterThan(ms, p.slowThresholdMilliseconds)
    }

    func testProbeStopsOnCancellation() async throws {
        let encoder = FakeEncoder(windows: [64])
        encoder.delay = 0.05
        let task = Task { try await probeEncoder(encoder, nMels: 128, policy: EncoderHealthPolicy()) }
        task.cancel()
        do { _ = try await task.value; XCTFail("expected a cancellation") } catch is CancellationError {}
    }

    // MARK: the memory file

    func testMemoryRoundTripPruningAndCorruption() throws {
        let memory = HealthMemory(file: memoryFile)
        XCTAssertNil(memory.record(for: "a"))
        for i in 0..<(HealthMemory.maxEntries + 3) {
            memory.store(HealthRecord(outcome: .stillSlow, probeMilliseconds: Double(i), date: Date(timeIntervalSince1970: Double(i))), for: "k\(i)")
        }
        XCTAssertNil(memory.record(for: "k0"), "the oldest entries are pruned")
        XCTAssertEqual(memory.record(for: "k\(HealthMemory.maxEntries + 2)")?.outcome, .stillSlow)
        try Data("not json".utf8).write(to: memoryFile)
        XCTAssertNil(memory.record(for: "k5"), "a damaged file reads as empty")
        memory.store(HealthRecord(outcome: .healthy, probeMilliseconds: 20, date: Date()), for: "fresh")
        XCTAssertEqual(memory.record(for: "fresh")?.outcome, .healthy)
        XCTAssertNil(HealthMemory(file: nil).record(for: "fresh"))
    }

    func testRememberOutcomeOffWritesNothing() async throws {
        _ = try await healer(policy(remember: false), log: StatusLog()).run(ScriptedPreparer([.encoder(490), .encoder(480)]))
        XCTAssertFalse(FileManager.default.fileExists(atPath: memoryFile.path))
    }

    func testOSBuildIsNotEmpty() {
        XCTAssertFalse(currentOSBuild().isEmpty)
    }

    // MARK: compatibility

    func testAnEncoderWithoutAHealthVerdictLoadsAsBefore() async throws {
        let root = try TestPackage.makeTemporaryDirectory("compat")
        defer { try? FileManager.default.removeItem(at: root) }
        try TestPackage.write(to: root)
        let model = try TestPackage.model(root: root)
        XCTAssertNil(model.encoderHealth)
    }
}
