import XCTest
@testable import EverParakeetKit

final class WordsTests: XCTestCase {
    let decoder = TokenDecoder(pieces: TestPackage.pieces, skip: [20, 21], unkId: 0)

    func tokens(_ ids: [Int], frames: [Int]? = nil, durations: [Int]? = nil, confidence: [Float]? = nil) -> [TimedToken] {
        ids.enumerated().map { i, id in
            TimedToken(id: id, frame: frames?[i] ?? i * 2, durationFrames: durations?[i] ?? 1, confidence: confidence?[i] ?? 0.5)
        }
    }

    func testPieceForIds() {
        XCTAssertEqual(decoder.piece(for: 1), "\u{2581}a")
        XCTAssertEqual(decoder.piece(for: TestPackage.blank), "")
        XCTAssertEqual(decoder.piece(for: -3), "")
        XCTAssertEqual(decoder.piece(for: 20), "", "pad is skipped")
        XCTAssertEqual(decoder.piece(for: 0), String(Character(GreekFinalSigma.unkMarker)))
    }

    func testWordsGroupPiecesAndMeanConfidence() {
        // ▁a b ▁c d  ->  "ab" "cd"
        let words = decoder.words(from: tokens([1, 2, 3, 4], frames: [10, 11, 14, 16], durations: [1, 2, 1, 3], confidence: [0.9, 0.5, 0.2, 0.4]))
        XCTAssertEqual(words.map(\.text), ["ab", "cd"])
        XCTAssertEqual(words[0].startFrame, 10)
        XCTAssertEqual(words[0].endFrame, 13)                       // last token 11 + 2
        XCTAssertEqual(words[0].confidence, 0.7, accuracy: 1e-6)
        XCTAssertEqual(words[1].startFrame, 14)
        XCTAssertEqual(words[1].endFrame, 19)
        XCTAssertEqual(words[1].tokenCount, 2)
    }

    func testDurationZeroTokensStillCountAsOneFrame() {
        let words = decoder.words(from: tokens([1, 2], frames: [5, 5], durations: [0, 0]))
        XCTAssertEqual(words.count, 1)
        XCTAssertEqual(words[0].startFrame, 5)
        XCTAssertEqual(words[0].endFrame, 6)
    }

    func testLoneWordMarkerEndsTheOpenWord() {
        // ▁a  ▁  b  ->  "a" then "b" (the bare marker is a boundary, and is not a token of any word)
        let words = decoder.words(from: tokens([1, 5, 2]))
        XCTAssertEqual(words.map(\.text), ["a", "b"])
        XCTAssertEqual(words[0].tokenCount, 1)
    }

    func testDigitsGlueIntoOneWord() {
        // ▁1 9 6 6 -> "1966"
        XCTAssertEqual(decoder.words(from: tokens([13, 14, 15, 15])).map(\.text), ["1966"])
    }

    func testGreekFinalSigmaAtWordLevel() {
        // ▁ό ρ ο σ -> όροσ -> όρος ; ▁και stays ; ▁σ alone stays σ
        let words = decoder.words(from: tokens([8, 9, 10, 11, 12, 16]))
        XCTAssertEqual(words.map(\.text), ["\u{03CC}\u{03C1}\u{03BF}\u{03C2}", "\u{03BA}\u{03B1}\u{03B9}", "\u{03C3}"])
        // an <unk> that ends a Greek word is the final sigma; a lone <unk> word disappears
        let unk = decoder.words(from: tokens([8, 9, 10, 0, 18, 0]))
        XCTAssertEqual(unk.map(\.text), ["\u{03CC}\u{03C1}\u{03BF}\u{03C2}", "x"])
    }

    func testWordsAgreeWithDecodedText() {
        var rng = SeededRandom(99)
        for _ in 0..<300 {
            let n = Int.random(in: 0...25, using: &rng)
            let ids = (0..<n).map { _ in Int.random(in: 0...TestPackage.blank, using: &rng) }
            let text = decoder.decode(ids)
            let words = decoder.words(from: tokens(ids))
            XCTAssertEqual(text.split(whereSeparator: \.isWhitespace).map(String.init), words.map(\.text), "ids \(ids)")
        }
    }

    func testBlanksAndSkippedIdsProduceNoWords() {
        XCTAssertTrue(decoder.words(from: tokens([TestPackage.blank, 20, 21])).isEmpty)
        XCTAssertTrue(decoder.words(from: []).isEmpty)
    }
}

final class MaskAndRoutingTests: XCTestCase {
    func testMaskLengths() {
        XCTAssertEqual(maskLengths(3000), [1500, 750, 375])
        XCTAssertEqual(maskLengths(0), [0, 0, 0])
        XCTAssertEqual(maskLengths(1), [1, 1, 1])
        XCTAssertEqual(maskLengths(513), [257, 129, 65])
    }

    func testRouteWindow() {
        let standard = [64, 192, 384]
        XCTAssertEqual(routeWindow(melFrames: 1, windows: standard), 64)
        XCTAssertEqual(routeWindow(melFrames: 512, windows: standard), 64)
        XCTAssertEqual(routeWindow(melFrames: 513, windows: standard), 192)
        XCTAssertEqual(routeWindow(melFrames: 1536, windows: standard), 192)
        XCTAssertEqual(routeWindow(melFrames: 1537, windows: standard), 384)
        XCTAssertEqual(routeWindow(melFrames: 3001, windows: standard), 384)
        XCTAssertNil(routeWindow(melFrames: 3073, windows: standard))
    }

    func testMain128OnlyServesFiveToTenSecondClips() {
        let withTen = [64, 128, 192, 384]
        XCTAssertEqual(routeWindow(melFrames: 512, windows: withTen), 64)          // 5 s and under: unchanged
        XCTAssertEqual(routeWindow(melFrames: 513, windows: withTen), 128)
        XCTAssertEqual(routeWindow(melFrames: 1024, windows: withTen), 128)
        XCTAssertEqual(routeWindow(melFrames: 1025, windows: withTen), 192)        // over 10 s: unchanged
    }

    func testConfigWindowsAndBlankBias() throws {
        let root = try TestPackage.makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        try TestPackage.write(to: root)
        XCTAssertNil(try EverParakeetConfig(root: root).longBlankBias)
        XCTAssertEqual(try EverParakeetConfig(root: root).windows, [64, 192, 384])
        try TestPackage.write(to: root, blankBias: 1.5)
        XCTAssertEqual(try EverParakeetConfig(root: root).longBlankBias, 1.5)
        XCTAssertEqual(try EverParakeetConfig(root: root).secondsPerFrame, 0.08, accuracy: 1e-9)
    }
}
