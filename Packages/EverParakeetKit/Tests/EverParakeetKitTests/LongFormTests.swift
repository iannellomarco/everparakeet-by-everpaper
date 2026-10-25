import XCTest
@testable import EverParakeetKit

final class LongFormTests: XCTestCase {
    func testLCSPlanCoversTheAudioWithTheRequestedOverlap() {
        // 70 s, 30 s windows (375 frames), 8 s overlap (100 frames): stride 275 frames = 22 s
        let fs = 1280
        let total = 70 * 16000
        let plan = LongFormPlan.lcsWindows(totalSamples: total, samplesPerFrame: fs, windowFrames: 375, overlapFrames: 100)
        XCTAssertEqual(plan.map(\.startFrame), [0, 275, 550])
        XCTAssertEqual(plan[0].endSample, 375 * fs)
        XCTAssertEqual(plan.last!.endSample, total)
        XCTAssertEqual(plan.map(\.previousEndFrame), [0, 375, 650])
        for (a, b) in zip(plan, plan.dropFirst()) {
            XCTAssertEqual((a.endSample - b.startSample) / fs, 100, "overlap in frames")
            XCTAssertLessThanOrEqual(b.endSample - b.startSample, 375 * fs)
        }
    }

    func testPlanForExactlyOneWindowAndTinyTail() {
        let fs = 1280
        XCTAssertEqual(LongFormPlan.lcsWindows(totalSamples: 375 * fs, samplesPerFrame: fs, windowFrames: 375, overlapFrames: 100).count, 1)
        // a tail shorter than the overlap is covered by the previous window: no extra window for it
        let plan = LongFormPlan.lcsWindows(totalSamples: 375 * fs + 5 * fs, samplesPerFrame: fs, windowFrames: 375, overlapFrames: 100)
        XCTAssertEqual(plan.count, 2)
        XCTAssertEqual(plan.last!.endSample, 380 * fs)
    }

    func testWindowFramesIsClamped() {
        var o = LongFormOptions()
        XCTAssertEqual(LongFormPlan.windowFrames(o, framesPerSecond: 12.5), 375)
        o.window = 60
        XCTAssertEqual(LongFormPlan.windowFrames(o, framesPerSecond: 12.5), 375)
        o.window = 1
        XCTAssertEqual(LongFormPlan.windowFrames(o, framesPerSecond: 12.5), 24)
    }

    func testQuietestCentreFindsThePause() {
        var x = [Float](repeating: 0.5, count: 16000 * 4)
        for i in 30000..<(30000 + 4000) { x[i] = 0 }          // 0.25 s of silence
        let centre = LongFormPlan.quietestCentre(x, width: 0.2, rate: 16000)
        XCTAssertTrue((30000...34000).contains(centre), "centre \(centre)")
        // no room for the window: the end of the region
        XCTAssertEqual(LongFormPlan.quietestCentre([0.1, 0.2], width: 0.2, rate: 16000), 2)
    }

    func testOptionDefaults() {
        let o = LongFormOptions()
        XCTAssertEqual(o.mode, .lcs)
        XCTAssertEqual(o.overlap, 8)
        XCTAssertNil(o.blankBias)
        XCTAssertEqual(LongFormOptions.defaultBlankBias, 1.25)
        XCTAssertEqual(o.loopRepeats, 5)
        XCTAssertEqual(o.loopRepeats1, 8)
        XCTAssertEqual(o.durationCap, 4)
        XCTAssertFalse(o.biasShort)
        XCTAssertEqual(LongFormOptions.off.mode, .legacy)
        XCTAssertEqual(LongFormOptions.off.blankBias, 0)
    }
}

final class LCSMergeTests: XCTestCase {
    // one piece per word: ▁a ▁b ▁c ▁the ▁x ...
    let table = ["<unk>", "\u{2581}a", "\u{2581}b", "\u{2581}c", "\u{2581}d", "\u{2581}e", "\u{2581}f", "\u{2581}g", "\u{2581}h"]
    func piece(_ id: Int) -> String { table[id] }
    func tok(_ id: Int, _ frame: Int) -> TimedToken { TimedToken(id: id, frame: frame, durationFrames: 1, confidence: 0.9) }

    func testSeamHasNoDuplicatedOrMissingWords() {
        // left window frames 0..<100 saw a b c d e (e is cut mid-word at the edge); right starts at 60 and saw d e f g h
        let left = [tok(1, 5), tok(2, 25), tok(3, 45), tok(4, 65), tok(5, 85)]
        let right = [tok(4, 66), tok(5, 86), tok(6, 105), tok(7, 125), tok(8, 145)]
        let merged = LongFormMerge.merge(left, right, overlapStart: 60, overlapEnd: 100, piece: piece)
        XCTAssertEqual(merged.map(\.id), [1, 2, 3, 4, 5, 6, 7, 8])
        XCTAssertEqual(merged.map(\.frame), merged.map(\.frame).sorted())
    }

    func testJunctionIsTheCommonWordNearestTheMiddleOfTheOverlap() {
        let left = [tok(1, 5), tok(4, 65), tok(5, 85)]
        let right = [tok(4, 64), tok(5, 88), tok(6, 105)]
        let merged = LongFormMerge.merge(left, right, overlapStart: 60, overlapEnd: 100, piece: piece)
        // middle of the overlap is frame 80: word 5 (frame 85) is nearer than word 4 (frame 65), so the junction is at 5
        XCTAssertEqual(merged.map(\.id), [1, 4, 5, 6])
        XCTAssertEqual(merged[2].frame, 85, "the left window supplies the junction word")
    }

    func testTimesNeverGoBackwardsAtTheJunction() {
        // the junction word sits 17 frames earlier in the right window than in the left one
        let left = [tok(1, 70), tok(2, 85), tok(3, 95)]
        let right = [tok(2, 68), tok(3, 75), tok(4, 105)]
        let merged = LongFormMerge.merge(left, right, overlapStart: 60, overlapEnd: 100, piece: piece)
        XCTAssertEqual(merged.map(\.id), [1, 2, 3, 4])
        XCTAssertEqual(merged.map(\.frame), [70, 85, 85, 105])
    }

    func testNoCommonWordCutsAtTheMiddleInTime() {
        let left = [tok(1, 10), tok(2, 70), tok(3, 90)]
        let right = [tok(4, 62), tok(5, 85), tok(6, 105)]
        let merged = LongFormMerge.merge(left, right, overlapStart: 60, overlapEnd: 100, piece: piece)
        // middle = 80: left keeps frames < 80, right keeps frames >= 80
        XCTAssertEqual(merged.map(\.id), [1, 2, 5, 6])
    }

    func testNoCommonWordCutNeverSplitsAWord() {
        // pieces: 1 "▁he", 2 "llo", 3 "▁wor", 4 "ld", 5 "▁x", 6 "▁y": "world" starts before the middle (80) but its
        // last piece lies after it; the right window has no word in common.
        let pieces = ["<unk>", "\u{2581}he", "llo", "\u{2581}wor", "ld", "\u{2581}x", "\u{2581}y"]
        let left = [tok(1, 10), tok(2, 12), tok(3, 75), tok(4, 83)]
        let right = [tok(5, 90), tok(6, 105)]
        let merged = LongFormMerge.merge(left, right, overlapStart: 60, overlapEnd: 100, piece: { pieces[$0] })
        XCTAssertEqual(merged.map(\.id), [1, 2, 3, 4, 5, 6], "the word that starts before the middle stays whole")
        // A right word that starts before the middle is dropped whole, not from its second piece on.
        let right2 = [tok(3, 70), tok(4, 78), tok(5, 90)]
        let merged2 = LongFormMerge.merge([tok(1, 10), tok(2, 12), tok(6, 65)], right2, overlapStart: 60, overlapEnd: 100,
                                          piece: { pieces[$0] })
        XCTAssertEqual(merged2.map(\.id), [1, 2, 6, 5])
    }

    func testEmptySidesAndDistantMatchesDoNotDuplicate() {
        XCTAssertEqual(LongFormMerge.merge([tok(1, 5)], [], overlapStart: 60, overlapEnd: 100, piece: piece).map(\.id), [1])
        // same word far apart in time (beyond the 1.6 s tolerance) is not a match
        let merged = LongFormMerge.merge([tok(1, 5), tok(4, 62)], [tok(4, 99), tok(5, 120)], overlapStart: 60, overlapEnd: 100, piece: piece)
        XCTAssertEqual(merged.filter { $0.id == 4 }.count, 1 + 1, "no pair: time cut, both survive their side of the cut or one is dropped by the cut")
    }

    func testLongListsMergeInLinearWork() {
        var left: [TimedToken] = [], right: [TimedToken] = []
        for i in 0..<4000 { left.append(tok(1 + i % 8, i)) }                     // a long history before the overlap
        for i in 0..<30 { left.append(tok(1 + i % 8, 4000 + i * 3)) }
        for i in 0..<30 { right.append(tok(1 + (i + 0) % 8, 4000 + i * 3 + 1)) }
        let merged = LongFormMerge.merge(left, right, overlapStart: 4000, overlapEnd: 4100, piece: piece)
        XCTAssertGreaterThan(merged.count, 4000)
        XCTAssertEqual(Array(merged[0..<4000]).map(\.id), Array(left[0..<4000]).map(\.id))
    }
}
