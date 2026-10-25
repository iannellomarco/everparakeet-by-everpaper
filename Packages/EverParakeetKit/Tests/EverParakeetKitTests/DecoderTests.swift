import Accelerate
import XCTest
@testable import EverParakeetKit

final class SoftmaxTests: XCTestCase {
    func testProbabilityMatchesDoubleReference() {
        var rng = SeededRandom(11)
        let x = (0..<500).map { _ in Float.random(in: -12...9, using: &rng) }
        var scratch = [Float](repeating: 0, count: x.count)
        let peak = Double(x.max()!)
        let denominator = x.reduce(0.0) { $0 + exp(Double($1) - peak) }
        for index in [0, 17, 250, 499, x.firstIndex(of: x.max()!)!] {
            let expected = exp(Double(x[index]) - peak) / denominator
            let got = x.withUnsafeBufferPointer { p in scratch.withUnsafeMutableBufferPointer { s in
                Softmax.probability(of: index, in: p.baseAddress!, count: x.count, scratch: s.baseAddress!) } }
            XCTAssertEqual(Double(got), expected, accuracy: 1e-5)
        }
    }

    func testProbabilitiesSumToOne() {
        var rng = SeededRandom(5)
        let x = (0..<64).map { _ in Float.random(in: -5...5, using: &rng) }
        var scratch = [Float](repeating: 0, count: x.count)
        var sum: Float = 0
        for i in 0..<x.count {
            sum += x.withUnsafeBufferPointer { p in scratch.withUnsafeMutableBufferPointer { s in
                Softmax.probability(of: i, in: p.baseAddress!, count: x.count, scratch: s.baseAddress!) } }
        }
        XCTAssertEqual(sum, 1, accuracy: 1e-4)
    }

    func testTopKIsSortedAndComplete() {
        var rng = SeededRandom(2)
        let x = (0..<300).map { _ in Float.random(in: -3...3, using: &rng) }
        let top = x.withUnsafeBufferPointer { Softmax.topK($0.baseAddress!, count: x.count, k: 8) }
        let expected = x.enumerated().sorted { $0.element != $1.element ? $0.element > $1.element : $0.offset < $1.offset }.prefix(8)
        XCTAssertEqual(top.ids, expected.map(\.offset))
        XCTAssertEqual(top.values, expected.map(\.element))
        let none = x.withUnsafeBufferPointer { Softmax.topK($0.baseAddress!, count: x.count, k: 0) }
        XCTAssertTrue(none.ids.isEmpty && none.values.isEmpty)
        let few = x.prefix(3).withUnsafeBufferPointer { Softmax.topK($0.baseAddress!, count: 3, k: 8) }
        XCTAssertEqual(few.ids.count, 3)
    }
}

/// A chooser that records what the decoder hands it.
final class RecordingChooser: TokenChooser {
    var resets = 0
    var observed: [String] = []
    var calls: [(original: Int, topK: [Int], logits: [Float], prob: Float)] = []
    var replacement: ((Int, [Int]) -> (Int, Float)?)?
    var onChoose: (() -> Void)?
    func reset() { resets += 1 }
    func observe(piece: String) { observed.append(piece) }
    func choose(original: Int, topK: [Int], logits: [Float], blankId: Int, originalProb: Float) -> (id: Int, confidence: Float) {
        calls.append((original, topK, logits, originalProb))
        onChoose?()
        if let r = replacement?(calls.count, topK) { return (r.0, r.1) }
        return (original, originalProb)
    }
}

final class DecoderTests: XCTestCase {
    var root: URL!
    var weights: DecoderWeights!
    var encoder: [Float] = []
    let frames = 120

    override func setUpWithError() throws {
        root = try TestPackage.makeTemporaryDirectory()
        try TestPackage.write(to: root)
        weights = try DecoderWeights(root: root, config: try EverParakeetConfig(root: root))
        encoder = (0..<(frames * TestPackage.hidden)).map { 1.8 * sin(Float($0 / TestPackage.hidden) * 0.37 + Float($0 % TestPackage.hidden) * 0.9) }
    }

    override func tearDownWithError() throws { try? FileManager.default.removeItem(at: root) }

    func decode(_ decoder: CPUTDTDecoder, offset: Int = 0, params: CPUTDTDecoder.Params = .init(), chooser: TokenChooser? = nil,
                cancel: CancelFlag? = nil) throws -> [TimedToken] {
        try encoder.withUnsafeBufferPointer {
            try decoder.decode(encoder: $0.baseAddress!, validFrames: frames, frameOffset: offset, params: params, chooser: chooser,
                               piece: { TestPackage.pieces.indices.contains($0) ? TestPackage.pieces[$0] : "" }, cancel: cancel)
        }
    }

    func testEmitsTimedTokensWithConfidence() throws {
        let tokens = try decode(CPUTDTDecoder(weights: weights, threads: 1))
        XCTAssertFalse(tokens.isEmpty)
        var previous = -1
        for t in tokens {
            XCTAssertGreaterThanOrEqual(t.frame, previous)
            XCTAssertLessThan(t.frame, frames)
            XCTAssertTrue((0...4).contains(t.durationFrames))
            XCTAssertTrue(t.confidence > 0 && t.confidence <= 1, "confidence \(t.confidence)")
            XCTAssertTrue((0..<TestPackage.blank).contains(t.id))
            previous = t.frame
        }
        XCTAssertTrue(Set(tokens.map(\.confidence)).count > 1, "confidence is not constant")
    }

    func testFrameOffsetShiftsEveryFrameAndNothingElse() throws {
        let a = try decode(CPUTDTDecoder(weights: weights, threads: 1))
        let b = try decode(CPUTDTDecoder(weights: weights, threads: 1), offset: 375)
        XCTAssertEqual(a.map(\.id), b.map(\.id))
        XCTAssertEqual(a.map { $0.frame + 375 }, b.map(\.frame))
        XCTAssertEqual(a.map(\.confidence), b.map(\.confidence))
    }

    func testThreadCountDoesNotChangeTheResult() throws {
        let one = try decode(CPUTDTDecoder(weights: weights, threads: 1))
        let four = try decode(CPUTDTDecoder(weights: weights, threads: 4))
        XCTAssertEqual(one, four)
    }

    func testDecoderIsReusableAndDeterministic() throws {
        let decoder = CPUTDTDecoder(weights: weights, threads: 1)
        let first = try decode(decoder)
        let second = try decode(decoder)
        XCTAssertEqual(first, second)
    }

    func testBlankBiasShiftsTheDecisionAndNeverChangesZeroBiasOutput() throws {
        let decoder = CPUTDTDecoder(weights: weights, threads: 1)
        let plain = try decode(decoder)
        var never = CPUTDTDecoder.Params(); never.blankBias = -60       // blank always wins
        XCTAssertEqual(try decode(decoder, params: never).count, 0)
        var more = CPUTDTDecoder.Params(); more.blankBias = 3
        XCTAssertGreaterThan(try decode(decoder, params: more).count, plain.count)
        XCTAssertEqual(try decode(decoder), plain)                      // state of the decoder was fully reset
    }

    func testConfidenceIgnoresTheBias() throws {
        // with a bias the emitted token's confidence is still its probability under the model's own logits
        let decoder = CPUTDTDecoder(weights: weights, threads: 1)
        var biased = CPUTDTDecoder.Params(); biased.blankBias = 2; biased.loopRepeats = 0; biased.loopRepeats1 = 0
        let chooser = RecordingChooser()
        let tokens = try decode(decoder, params: biased, chooser: chooser)
        XCTAssertEqual(tokens.count, chooser.calls.count)
        for (t, call) in zip(tokens, chooser.calls) { XCTAssertEqual(t.confidence, call.prob, accuracy: 1e-7) }
    }

    func testChooserGetsTheOriginalTokenEvenWhenTheBiasMadeItWin() throws {
        // A strong blank bias is what lets a non-blank token win over a higher raw blank: then topK[0] is the blank
        // and `original` is the token the decoder is about to emit.
        let decoder = CPUTDTDecoder(weights: weights, threads: 1)
        var biased = CPUTDTDecoder.Params(); biased.blankBias = 3; biased.loopRepeats = 0; biased.loopRepeats1 = 0
        let chooser = RecordingChooser()
        let tokens = try decode(decoder, params: biased, chooser: chooser)
        XCTAssertEqual(tokens.count, chooser.calls.count)
        XCTAssertEqual(tokens.map(\.id), chooser.calls.map(\.original))   // the default answer leaves the decision alone
        XCTAssertTrue(chooser.calls.allSatisfy { $0.original != TestPackage.blank })
        XCTAssertTrue(chooser.calls.contains { $0.topK[0] == TestPackage.blank }, "no call had the blank on top; raise the bias")
    }

    func testRepetitionGuardStopsLoops() throws {
        var forced = CPUTDTDecoder.Params(); forced.blankBias = 60      // blank never wins: the decoder would emit on every frame
        let decoder = CPUTDTDecoder(weights: weights, threads: 1)
        var unguarded = forced; unguarded.loopRepeats = 0; unguarded.loopRepeats1 = 0
        let free = try decode(decoder, params: unguarded)
        let guarded = try decode(decoder, params: forced)
        XCTAssertGreaterThan(free.count, 0)
        // no token repeats 8 times in a row in the guarded output
        var run = 1
        for i in 1..<max(1, guarded.count) { run = guarded[i].id == guarded[i - 1].id ? run + 1 : 1; XCTAssertLessThan(run, 8) }
    }

    func testChooserSeesRawTopKAndMayReplace() throws {
        let decoder = CPUTDTDecoder(weights: weights, threads: 1)
        let plain = try decode(decoder)
        let chooser = RecordingChooser()
        let same = try decode(decoder, chooser: chooser)
        XCTAssertEqual(same, plain, "a chooser that keeps the original must not change anything")
        XCTAssertEqual(chooser.calls.count, plain.count)
        XCTAssertEqual(chooser.resets, 1)
        for (t, call) in zip(plain, chooser.calls) {
            XCTAssertEqual(call.topK.count, 16)
            XCTAssertEqual(call.topK[0], t.id)
            XCTAssertEqual(call.logits, call.logits.sorted(by: >))
            XCTAssertEqual(call.topK.count, call.logits.count)
        }
        // observe() got the piece of every emitted token, in order
        XCTAssertEqual(chooser.observed, plain.map { TestPackage.pieces[$0.id] })

        let replacing = RecordingChooser()
        replacing.replacement = { n, top in n == 1 ? (top.dropFirst().first(where: { $0 != TestPackage.blank })!, 0.5) : nil }
        let changed = try decode(decoder, chooser: replacing)
        XCTAssertEqual(changed[0].id, replacing.calls[0].topK.dropFirst().first(where: { $0 != TestPackage.blank }))
        XCTAssertNotEqual(changed[0].id, plain[0].id)
        XCTAssertEqual(changed[0].confidence, 0.5)
    }

    func testCancelFlagStopsTheDecoder() throws {
        let decoder = CPUTDTDecoder(weights: weights, threads: 1)
        let flag = CancelFlag()
        flag.cancel()
        XCTAssertThrowsError(try decode(decoder, cancel: flag)) { XCTAssertTrue($0 is CancellationError) }

        // raised from inside a decode: stops within `cancelStride` joint calls
        let mid = CancelFlag()
        let chooser = RecordingChooser()
        chooser.onChoose = { if chooser.calls.count == 3 { mid.cancel() } }
        var params = CPUTDTDecoder.Params(); params.cancelStride = 1
        XCTAssertThrowsError(try decode(decoder, params: params, chooser: chooser, cancel: mid)) { XCTAssertTrue($0 is CancellationError) }
        XCTAssertLessThanOrEqual(chooser.calls.count, 5)
        XCTAssertEqual(try decode(decoder), try decode(CPUTDTDecoder(weights: weights, threads: 1)), "a decoder is usable after a cancel")
    }

    func testWeightsRejectMismatchedConfig() throws {
        var config = try EverParakeetConfig(root: root)
        _ = config
        let bad = try TestPackage.makeTemporaryDirectory("bad")
        defer { try? FileManager.default.removeItem(at: bad) }
        try TestPackage.write(to: bad)
        // a config that claims a bigger vocabulary than the weights hold
        let url = bad.appendingPathComponent("models/config.json")
        var json = try JSONSerialization.jsonObject(with: Data(contentsOf: url)) as! [String: Any]
        json["vocab_size"] = 40
        try JSONSerialization.data(withJSONObject: json).write(to: url)
        config = try EverParakeetConfig(root: bad)
        XCTAssertThrowsError(try DecoderWeights(root: bad, config: config))
    }
}
