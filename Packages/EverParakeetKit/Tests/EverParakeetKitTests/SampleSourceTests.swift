import XCTest
@testable import EverParakeetKit

final class SampleSourceTests: XCTestCase {
    var dir: URL!
    override func setUpWithError() throws { dir = try TestPackage.makeTemporaryDirectory("wav") }
    override func tearDownWithError() throws { try? FileManager.default.removeItem(at: dir) }

    func le32(_ v: UInt32) -> [UInt8] { (0..<4).map { UInt8((v >> (8 * UInt32($0))) & 0xFF) } }
    func le16(_ v: UInt16) -> [UInt8] { [UInt8(v & 0xFF), UInt8(v >> 8)] }

    /// RIFF/WAVE with optional extra chunk before `data` and an explicit data-length field.
    func wav(samples: [Float], float32: Bool = true, channels: Int = 1, rate: UInt32 = 16000, lengthField: UInt32? = nil,
             extraChunk: Bool = false, trailingBytes: [UInt8] = [], name: String = "a.wav") throws -> URL {
        var payload: [UInt8] = []
        for s in samples {
            if float32 { payload += le32(s.bitPattern) } else { payload += le16(UInt16(bitPattern: Int16(max(-1, min(1, s)) * 32767))) }
        }
        payload += trailingBytes
        let width = float32 ? 4 : 2
        var fmt: [UInt8] = le16(float32 ? 3 : 1) + le16(UInt16(channels)) + le32(rate) + le32(rate * UInt32(channels * width))
        fmt += le16(UInt16(channels * width)) + le16(UInt16(width * 8))
        var bytes = Array("RIFF".utf8) + le32(0) + Array("WAVE".utf8)
        bytes += Array("fmt ".utf8) + le32(UInt32(fmt.count)) + fmt
        if extraChunk { bytes += Array("LIST".utf8) + le32(5) + [1, 2, 3, 4, 5, 0] }       // odd size: padded to even
        bytes += Array("data".utf8) + le32(lengthField ?? UInt32(payload.count)) + payload
        let url = dir.appendingPathComponent(name)
        try Data(bytes).write(to: url)
        return url
    }

    let ramp: [Float] = (0..<1000).map { Float($0) / 1000 - 0.5 }

    func testArraySourceClampsAndSanitises() throws {
        let source = ArraySampleSource([0.1, .nan, 0.3, .infinity, 0.5])
        XCTAssertEqual(source.sampleCount, 5)
        XCTAssertEqual(try source.read(1..<4), [0, 0.3, 0])
        XCTAssertEqual(try source.read(-5..<2), [0.1, 0])
        XCTAssertEqual(try source.read(4..<99), [0.5])
        XCTAssertEqual(try source.read(7..<9), [])
    }

    func testFloat32WithChunkBeforeData() throws {
        let source = try WAVSampleSource(url: try wav(samples: ramp, extraChunk: true))
        XCTAssertEqual(source.sampleCount, 1000)
        XCTAssertEqual(try source.read(0..<1000), ramp)
        XCTAssertEqual(try source.read(250..<260), Array(ramp[250..<260]))
    }

    func testStaleLengthFieldsAreClamped() throws {
        for field in [UInt32(0), 0xFFFF_FFFF, 3_000_000] {
            let source = try WAVSampleSource(url: try wav(samples: ramp, lengthField: field, name: "stale\(field).wav"))
            XCTAssertEqual(source.sampleCount, 1000, "length field \(field)")
            XCTAssertEqual(try source.read(990..<1000), Array(ramp[990..<1000]))
        }
    }

    func testPartialTrailingFrameIsDropped() throws {
        let source = try WAVSampleSource(url: try wav(samples: ramp, lengthField: 0, trailingBytes: [9, 9, 9], name: "partial.wav"))
        XCTAssertEqual(source.sampleCount, 1000)
    }

    func testShorterLengthFieldIsHonoured() throws {
        let source = try WAVSampleSource(url: try wav(samples: ramp, lengthField: 400, name: "short.wav"))
        XCTAssertEqual(source.sampleCount, 100)
    }

    func testPCM16AndStereoAverage() throws {
        let pcm = try WAVSampleSource(url: try wav(samples: [0, 0.5, -0.5, 1], float32: false, name: "pcm.wav"))
        XCTAssertEqual(pcm.sampleCount, 4)
        let got = try pcm.read(0..<4)
        XCTAssertEqual(got[1], 0.5, accuracy: 1e-3)
        XCTAssertEqual(got[2], -0.5, accuracy: 1e-3)
        let stereo = try WAVSampleSource(url: try wav(samples: [0.2, 0.6, -0.4, 0.0], channels: 2, name: "st.wav"))
        XCTAssertEqual(stereo.sampleCount, 2)
        XCTAssertEqual(try stereo.read(0..<2), [0.4, -0.2])
    }

    func testNaNSamplesBecomeZero() throws {
        let source = try WAVSampleSource(url: try wav(samples: [0.5, .nan, 0.25], name: "nan.wav"))
        XCTAssertEqual(try source.read(0..<3), [0.5, 0, 0.25])
    }

    func testRejectsOtherRatesAndGarbage() throws {
        XCTAssertThrowsError(try WAVSampleSource(url: try wav(samples: ramp, rate: 44100, name: "r.wav")))
        let junk = dir.appendingPathComponent("junk.wav")
        try Data("not a wav file at all".utf8).write(to: junk)
        XCTAssertThrowsError(try WAVSampleSource(url: junk))
        XCTAssertThrowsError(try WAVSampleSource(url: dir.appendingPathComponent("missing.wav")))
    }

    func testRandomAccessMatchesSequentialReads() throws {
        let long: [Float] = (0..<48_000).map { sin(Float($0) * 0.01) }
        let source = try WAVSampleSource(url: try wav(samples: long, name: "long.wav"))
        var joined: [Float] = []
        var start = 0
        while start < source.sampleCount { joined += try source.read(start..<(start + 7777)); start += 7777 }
        XCTAssertEqual(joined, long)
    }
}
