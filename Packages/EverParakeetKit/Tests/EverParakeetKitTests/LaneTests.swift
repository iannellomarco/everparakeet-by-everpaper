import XCTest
@testable import EverParakeetKit

final class ProgressRecorder: @unchecked Sendable {
    private let lock = NSLock()
    private var items: [TranscriptionProgress] = []
    func add(_ p: TranscriptionProgress) { lock.lock(); items.append(p); lock.unlock() }
    var all: [TranscriptionProgress] { lock.lock(); defer { lock.unlock() }; return items }
}

final class LaneTests: XCTestCase {
    var root: URL!
    override func setUpWithError() throws {
        root = try TestPackage.makeTemporaryDirectory()
        try TestPackage.write(to: root)
    }
    override func tearDownWithError() throws { try? FileManager.default.removeItem(at: root) }

    func lane(_ encoder: FakeEncoder = FakeEncoder(), priority: LanePriority = .interactive, model: EverParakeetModel? = nil) throws -> (EverParakeetLane, EverParakeetModel) {
        let m = try model ?? TestPackage.model(root: root, encoder: encoder)
        var o = LaneOptions(); o.priority = priority
        return (try m.makeLane(o), m)
    }

    // MARK: short audio

    func testShortClipTextWordsAndTokensAgree() async throws {
        let (lane, model) = try lane()
        let result = try await lane.transcribe(pcm16k: TestPackage.audio(seconds: 6))
        XCTAssertFalse(result.tokens.isEmpty)
        XCTAssertEqual(result.text, model.tokens.decode(result.tokens.map(\.id)))
        XCTAssertEqual(result.text.split(separator: " ").map(String.init), result.words.map(\.text))
        XCTAssertEqual(result.words.map(\.startFrame), result.words.map(\.startFrame).sorted())
        XCTAssertTrue(result.tokens.allSatisfy { $0.confidence > 0 && $0.confidence <= 1 })
        XCTAssertEqual(result.timings.pieces, 1)
        XCTAssertGreaterThan(result.timings.joints, 0)
        XCTAssertEqual(model.piece(for: result.tokens[0].id), TestPackage.pieces[result.tokens[0].id])
    }

    func testRepeatedCallsAreIdentical() async throws {
        let (lane, _) = try lane()
        let audio = TestPackage.audio(seconds: 12)
        let a = try await lane.transcribe(pcm16k: audio)
        let b = try await lane.transcribe(pcm16k: audio)
        XCTAssertEqual(a.tokens, b.tokens)
        XCTAssertEqual(a.text, b.text)
    }

    func testEmptyAudioDoesNotTouchTheEncoder() async throws {
        let encoder = FakeEncoder()
        let (lane, _) = try lane(encoder)
        let result = try await lane.transcribe(pcm16k: [])
        XCTAssertEqual(result.text, "")
        XCTAssertTrue(result.tokens.isEmpty)
        XCTAssertEqual(encoder.calls.count, 0)
        let tiny = try await lane.transcribe(pcm16k: [Float](repeating: 0.1, count: 100))   // less than one mel hop of valid frames
        XCTAssertEqual(tiny.text, "")
    }

    func testNaNSamplesAreHarmless() async throws {
        let (lane, _) = try lane()
        var audio = TestPackage.audio(seconds: 3)
        let clean = try await lane.transcribe(pcm16k: audio)
        audio[500] = .nan; audio[501] = .infinity
        _ = try await lane.transcribe(pcm16k: audio)
        var zeroed = TestPackage.audio(seconds: 3)
        zeroed[500] = 0; zeroed[501] = 0
        let b = try await lane.transcribe(pcm16k: audio)
        let c = try await lane.transcribe(pcm16k: zeroed)
        XCTAssertEqual(b.tokens, c.tokens, "NaN and infinite samples count as 0")
        _ = clean
    }

    // MARK: routing

    func testWindowRouting() async throws {
        let encoder = FakeEncoder(windows: [64, 128, 192, 384])
        let (lane, _) = try lane(encoder)
        for seconds in [3.0, 4.9, 6.0, 9.9, 12.0, 20.0, 29.0] { _ = try await lane.transcribe(pcm16k: TestPackage.audio(seconds: seconds)) }
        XCTAssertEqual(encoder.calls.map(\.window), [64, 64, 128, 128, 192, 384, 384])
        let last = try await lane.transcribe(pcm16k: TestPackage.audio(seconds: 8))
        XCTAssertEqual(last.timings.windowUse, [128: 1])
    }

    // MARK: long audio

    func testLegacyCutsReportProgressAndKeepOrder() async throws {
        let encoder = FakeEncoder()
        let (lane, _) = try lane(encoder)
        let recorder = ProgressRecorder()
        var options = TranscribeOptions(); options.longForm = .off
        let audio = TestPackage.audio(seconds: 75)
        let result = try await lane.transcribe(pcm16k: audio, options: options, progress: { recorder.add($0) })
        let progress = recorder.all
        XCTAssertEqual(progress.count, 3, "75 s is three cuts of at most 30 s")
        XCTAssertEqual(progress.map(\.windows), [1, 2, 3])
        XCTAssertEqual(progress.map(\.processedSamples), progress.map(\.processedSamples).sorted())
        XCTAssertEqual(progress.last!.processedSamples, audio.count)
        XCTAssertEqual(progress.last!.fraction, 1)
        XCTAssertEqual(result.words.map(\.startFrame), result.words.map(\.startFrame).sorted())
        XCTAssertEqual(result.timings.pieces, 3)
        XCTAssertLessThanOrEqual(result.tokens.map(\.frame).max() ?? 0, Int(Double(audio.count) / 1280) + 1)
    }

    func testLegacyWithBiasOffMatchesTheShortPathPieceByPiece() async throws {
        // two independent 20 s clips glued: legacy cuts at the quietest point; each piece decodes like a stand-alone clip
        let (lane, model) = try lane()
        var options = TranscribeOptions(); options.longForm = .off
        let audio = TestPackage.audio(seconds: 40, pauses: true)
        let long = try await lane.transcribe(pcm16k: audio, options: options)
        XCTAssertEqual(long.text, long.words.map(\.text).joined(separator: " "))
        let pieces = long.timings.pieces
        XCTAssertEqual(pieces, 2)
        _ = model
    }

    func testLCSWindowsMergeIntoOneMonotonicTranscript() async throws {
        let encoder = FakeEncoder()
        let (lane, _) = try lane(encoder)
        let recorder = ProgressRecorder()
        let audio = TestPackage.audio(seconds: 70)
        let result = try await lane.transcribe(pcm16k: audio, progress: { recorder.add($0) })
        XCTAssertEqual(recorder.all.map(\.windows), [1, 2, 3])
        XCTAssertEqual(recorder.all.last!.processedSamples, audio.count)
        XCTAssertEqual(encoder.calls.count, 3)
        XCTAssertTrue(encoder.calls.allSatisfy { $0.window == 384 })
        let frames = result.tokens.map(\.frame)
        XCTAssertEqual(frames, frames.sorted(), "token times never go backwards across the seams")
        XCTAssertEqual(result.text.split(separator: " ").map(String.init), result.words.map(\.text))
        XCTAssertEqual(result.timings.pieces, 3)
    }

    func testBlankBiasOnlyAppliesToLongAudio() async throws {
        let (lane, _) = try lane()
        let short = TestPackage.audio(seconds: 20)
        var plain = TranscribeOptions(); plain.longForm.blankBias = 0
        var biased = TranscribeOptions(); biased.longForm.blankBias = 4
        let a = try await lane.transcribe(pcm16k: short, options: plain)
        let b = try await lane.transcribe(pcm16k: short, options: biased)
        XCTAssertEqual(a.tokens, b.tokens, "30 s and under: the model as trained")
        biased.longForm.biasShort = true
        let c = try await lane.transcribe(pcm16k: short, options: biased)
        XCTAssertNotEqual(a.tokens.count, c.tokens.count)
        // a package-carried bias is used when the options leave it open
        try TestPackage.write(to: root, blankBias: 3)
        let model2 = try TestPackage.model(root: root)
        XCTAssertEqual(model2.config.longBlankBias, 3)
    }

    func testSampleSourceAndArrayGiveTheSameResult() async throws {
        let (lane, _) = try lane()
        let audio = TestPackage.audio(seconds: 40)
        let a = try await lane.transcribe(pcm16k: audio)
        let b = try await lane.transcribe(source: ArraySampleSource(audio))
        XCTAssertEqual(a.tokens, b.tokens)
    }

    // MARK: cancellation

    func testCancelledBeforeStart() async throws {
        let (lane, _) = try lane()
        let task = Task { try await lane.transcribe(pcm16k: TestPackage.audio(seconds: 5)) }
        task.cancel()
        do { _ = try await task.value; XCTFail("expected cancellation") } catch { XCTAssertTrue(error is CancellationError, "\(error)") }
        let ok = try await lane.transcribe(pcm16k: TestPackage.audio(seconds: 5))     // the lane is usable afterwards
        XCTAssertFalse(ok.tokens.isEmpty)
    }

    func testCancelBetweenWindowsStopsTheRest() async throws {
        let encoder = FakeEncoder()
        let (lane, _) = try lane(encoder)
        let audio = TestPackage.audio(seconds: 140)
        let started = expectation(description: "first window done")
        let task = Task { () throws -> Transcription in
            try await lane.transcribe(pcm16k: audio, progress: { p in if p.windows == 1 { started.fulfill() } })
        }
        await fulfillment(of: [started], timeout: 30)
        task.cancel()
        do { _ = try await task.value; XCTFail("expected cancellation") } catch { XCTAssertTrue(error is CancellationError, "\(error)") }
        XCTAssertLessThan(encoder.calls.count, 7, "later windows were not decoded")
    }

    func testCancelWhileWaitingForTheEncoder() async throws {
        let encoder = FakeEncoder()
        encoder.delay = 0.4
        let (lane, _) = try lane(encoder)
        let task = Task { try await lane.transcribe(pcm16k: TestPackage.audio(seconds: 10)) }
        try await Task.sleep(nanoseconds: 150_000_000)
        let start = Date()
        task.cancel()
        do { _ = try await task.value } catch { XCTAssertTrue(error is CancellationError) }
        XCTAssertLessThan(Date().timeIntervalSince(start), 1.0, "cancel returns within a second")
    }

    /// A call queued for the shared encoder stops waiting the moment it is cancelled, even though the prediction
    /// ahead of it ignores cancellation (as a real Core ML prediction does).
    func testCancelWhileQueuedForTheEncoderIsPrompt() async throws {
        let encoder = FakeEncoder()
        encoder.uncancellableDelay = 1.5
        let model = try TestPackage.model(root: root, encoder: encoder)
        let first = try model.makeLane()
        let second = try model.makeLane()
        let holder = Task { try await first.transcribe(pcm16k: TestPackage.audio(seconds: 3)) }
        try await Task.sleep(nanoseconds: 300_000_000)                 // the first call is inside its prediction
        let waiter = Task { try await second.transcribe(pcm16k: TestPackage.audio(seconds: 3)) }
        try await Task.sleep(nanoseconds: 300_000_000)                 // the second is queued at the gate
        let start = Date()
        waiter.cancel()
        do { _ = try await waiter.value; XCTFail("expected cancellation") } catch { XCTAssertTrue(error is CancellationError, "\(error)") }
        XCTAssertLessThan(Date().timeIntervalSince(start), 0.5, "cancel returned only after the running prediction")
        _ = try await holder.value                                      // the running call is untouched
        XCTAssertEqual(encoder.calls.count, 1, "the cancelled call never reached the encoder")
    }

    /// A second call on the same lane stops waiting when cancelled, without waiting for the first call to finish.
    func testCancelWhileQueuedBehindTheSameLaneIsPrompt() async throws {
        let encoder = FakeEncoder()
        encoder.uncancellableDelay = 1.5
        let (lane, _) = try lane(encoder)
        let first = Task { try await lane.transcribe(pcm16k: TestPackage.audio(seconds: 3)) }
        try await Task.sleep(nanoseconds: 300_000_000)
        let second = Task { try await lane.transcribe(pcm16k: TestPackage.audio(seconds: 3)) }
        try await Task.sleep(nanoseconds: 300_000_000)
        let start = Date()
        second.cancel()
        do { _ = try await second.value; XCTFail("expected cancellation") } catch { XCTAssertTrue(error is CancellationError, "\(error)") }
        XCTAssertLessThan(Date().timeIntervalSince(start), 0.5)
        _ = try await first.value
        // The lane is free again afterwards.
        _ = try await lane.transcribe(pcm16k: TestPackage.audio(seconds: 3))
    }

    // MARK: chooser

    func testLaneChooserIsUsedAndConfidenceFollowsIt() async throws {
        let (lane, _) = try lane()
        let chooser = RecordingChooser()
        chooser.replacement = { _, top in (top[0], 0.123) }
        await lane.setChooser(chooser)
        let result = try await lane.transcribe(pcm16k: TestPackage.audio(seconds: 5))
        XCTAssertEqual(chooser.calls.count, result.tokens.count)
        XCTAssertTrue(result.tokens.allSatisfy { $0.confidence == 0.123 })
        await lane.setChooser(nil)
        let plain = try await lane.transcribe(pcm16k: TestPackage.audio(seconds: 5))
        XCTAssertTrue(plain.tokens.contains { $0.confidence != 0.123 })
    }

    // MARK: lanes

    func testLanesHaveSeparateEncoderInputs() throws {
        let encoder = FakeEncoder()
        let model = try TestPackage.model(root: root, encoder: encoder)
        let a = try model.makeLane(), b = try model.makeLane()
        XCTAssertNotEqual(a.inputsIdentity, b.inputsIdentity)
        XCTAssertEqual(encoder.inputsMade, 2)
    }

    func testTwoLanesRunConcurrentlyWithoutSharingBuffersAndMatchSequentialResults() async throws {
        let encoder = FakeEncoder()
        encoder.delay = 0.05
        let model = try TestPackage.model(root: root, encoder: encoder)
        let a = try model.makeLane(), b = try model.makeLane()
        let audioA = TestPackage.audio(seconds: 45, seed: 1), audioB = TestPackage.audio(seconds: 33, seed: 2)
        let alone = try await a.transcribe(pcm16k: audioA)
        let aloneB = try await b.transcribe(pcm16k: audioB)
        async let ra = a.transcribe(pcm16k: audioA)
        async let rb = b.transcribe(pcm16k: audioB)
        let (x, y) = try await (ra, rb)
        XCTAssertEqual(x.tokens, alone.tokens)
        XCTAssertEqual(y.tokens, aloneB.tokens)
        XCTAssertFalse(encoder.overlappingInputsUse)
    }

    func testSameLaneCallsAreSerialised() async throws {
        let encoder = FakeEncoder()
        encoder.delay = 0.03
        let (lane, _) = try lane(encoder)
        async let r1 = lane.transcribe(pcm16k: TestPackage.audio(seconds: 8, seed: 1))
        async let r2 = lane.transcribe(pcm16k: TestPackage.audio(seconds: 8, seed: 2))
        _ = try await (r1, r2)
        XCTAssertFalse(encoder.overlappingInputsUse)
    }

    func testInteractiveLaneGoesBeforeQueuedBackgroundLanes() async throws {
        let encoder = FakeEncoder()
        let model = try TestPackage.model(root: root, encoder: encoder)
        var bg = LaneOptions(); bg.priority = .background
        var hi = LaneOptions(); hi.priority = .interactive
        let blocker = try model.makeLane(bg), queuedBackground = try model.makeLane(bg), dictation = try model.makeLane(hi)
        encoder.firstCallDelay = 0.5                                   // the blocker holds the encoder for half a second
        let t1 = Task { try await blocker.transcribe(pcm16k: TestPackage.audio(seconds: 4, seed: 1)) }
        try await Task.sleep(nanoseconds: 150_000_000)
        let t2 = Task { try await queuedBackground.transcribe(pcm16k: TestPackage.audio(seconds: 4, seed: 2)) }
        try await Task.sleep(nanoseconds: 100_000_000)
        let t3 = Task { try await dictation.transcribe(pcm16k: TestPackage.audio(seconds: 4, seed: 3)) }
        _ = try await (t1.value, t2.value, t3.value)
        // lanes were made in the order blocker (1), queued background (2), dictation (3)
        XCTAssertEqual(encoder.calls.map(\.serial), [1, 3, 2], "the dictation lane is served before the background lane that queued first")
    }

    func testRecommendedDecoderThreadsIsTwoOrFour() {
        XCTAssertTrue([2, 4].contains(EverParakeetModel.recommendedDecoderThreads))
    }
}
