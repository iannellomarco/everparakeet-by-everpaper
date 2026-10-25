import Foundation

/// Random-access mono 16 kHz audio. Long recordings are read window by window through this protocol, so the whole
/// track never has to sit in memory as a `[Float]` (four hours of 16 kHz float audio is 0.92 GB).
public protocol SampleSource: Sendable {
    /// Total number of samples.
    var sampleCount: Int { get }
    /// Samples `range` (clamped to `0..<sampleCount`). NaN and infinite samples are returned as 0.
    func read(_ range: Range<Int>) throws -> [Float]
}

/// An in-memory `[Float]`.
public struct ArraySampleSource: SampleSource {
    private let samples: [Float]
    public init(_ samples: [Float]) { self.samples = samples }
    public var sampleCount: Int { samples.count }
    public func read(_ range: Range<Int>) throws -> [Float] {
        let lo = max(0, range.lowerBound), hi = min(samples.count, range.upperBound)
        guard lo < hi else { return [] }
        return EverParakeetAudio.sanitized(Array(samples[lo..<hi]))
    }
}

/// A WAV file read on demand: mono (or averaged to mono) 16 kHz, 32-bit float or 16-bit PCM.
///
/// The `data` chunk is located by walking the chunk list (a `LIST` or `FACT` chunk before it is fine). Files that were
/// recovered after a crash often carry a stale length: a data length of 0 or 0xFFFFFFFF means "to the end of the
/// file", and a length larger than what is on disk is clamped to what is on disk. A trailing partial frame is dropped.
public final class WAVSampleSource: SampleSource, @unchecked Sendable {
    public let sampleCount: Int
    private let fd: Int32
    private let dataStart: Int64
    private let channels: Int
    private let isFloat: Bool

    public init(url: URL) throws {
        let fd = open(url.path, O_RDONLY)
        guard fd >= 0 else { throw EverParakeetError.missing("audio file \(url.path)") }
        self.fd = fd
        func fail(_ message: String) -> EverParakeetError { EverParakeetError.invalid("\(url.lastPathComponent): \(message)") }
        var st = stat()
        guard fstat(fd, &st) == 0 else { close(fd); throw fail("cannot stat") }
        let fileSize = Int64(st.st_size)
        func bytes(_ offset: Int64, _ count: Int) -> [UInt8]? {
            var buf = [UInt8](repeating: 0, count: count)
            let n = buf.withUnsafeMutableBytes { pread(fd, $0.baseAddress, count, off_t(offset)) }
            return n == count ? buf : nil
        }
        func u16(_ b: [UInt8], _ o: Int) -> Int { Int(b[o]) | Int(b[o + 1]) << 8 }
        func u32(_ b: [UInt8], _ o: Int) -> Int64 { Int64(b[o]) | Int64(b[o + 1]) << 8 | Int64(b[o + 2]) << 16 | Int64(b[o + 3]) << 24 }
        guard let head = bytes(0, 12), String(decoding: head[0..<4], as: UTF8.self) == "RIFF",
              String(decoding: head[8..<12], as: UTF8.self) == "WAVE" else { close(fd); throw fail("not a RIFF/WAVE file") }
        var offset: Int64 = 12
        var format: (tag: Int, channels: Int, rate: Int64, bits: Int)? = nil
        var start: Int64 = -1, length: Int64 = 0
        while offset + 8 <= fileSize {
            guard let h = bytes(offset, 8) else { break }
            let id = String(decoding: h[0..<4], as: UTF8.self)
            let size = u32(h, 4)
            let body = offset + 8
            if id == "fmt " {
                guard size >= 16, let f = bytes(body, 16) else { close(fd); throw fail("short fmt chunk") }
                var tag = u16(f, 0)
                if tag == 0xFFFE, size >= 26, let ext = bytes(body + 24, 2) { tag = u16(ext, 0) }   // WAVE_FORMAT_EXTENSIBLE
                format = (tag, u16(f, 2), u32(f, 4), u16(f, 14))
            } else if id == "data" {
                start = body
                let remaining = fileSize - body
                length = (size == 0 || size == 0xFFFF_FFFF || size > remaining) ? remaining : size
                break
            }
            offset = body + size + (size & 1)
        }
        guard let fmt = format else { close(fd); throw fail("no fmt chunk") }
        guard start >= 0 else { close(fd); throw fail("no data chunk") }
        guard fmt.rate == 16000 else { close(fd); throw fail("sample rate \(fmt.rate) Hz, expected 16000") }
        guard fmt.channels >= 1 else { close(fd); throw fail("no channels") }
        switch (fmt.tag, fmt.bits) {
        case (3, 32): isFloat = true
        case (1, 16): isFloat = false
        default: close(fd); throw fail("unsupported sample format (tag \(fmt.tag), \(fmt.bits) bits); use 32-bit float or 16-bit PCM")
        }
        channels = fmt.channels
        dataStart = start
        let frameBytes = Int64(fmt.channels * fmt.bits / 8)
        sampleCount = Int(length / frameBytes)
    }

    deinit { close(fd) }

    public func read(_ range: Range<Int>) throws -> [Float] {
        let lo = max(0, range.lowerBound), hi = min(sampleCount, range.upperBound)
        guard lo < hi else { return [] }
        let n = hi - lo
        let width = isFloat ? 4 : 2
        let frameBytes = width * channels
        var raw = [UInt8](repeating: 0, count: n * frameBytes)
        let got = raw.withUnsafeMutableBytes { pread(fd, $0.baseAddress, n * frameBytes, off_t(dataStart + Int64(lo * frameBytes))) }
        guard got == n * frameBytes else { throw EverParakeetError.invalid("short read from the WAV file") }
        var out = [Float](repeating: 0, count: n)
        raw.withUnsafeBytes { p in
            for i in 0..<n {
                var sum: Float = 0
                for c in 0..<channels {
                    let o = (i * channels + c) * width
                    if isFloat {
                        sum += Float(bitPattern: UInt32(littleEndian: p.loadUnaligned(fromByteOffset: o, as: UInt32.self)))
                    } else {
                        sum += Float(Int16(littleEndian: p.loadUnaligned(fromByteOffset: o, as: Int16.self))) / 32768
                    }
                }
                out[i] = sum / Float(channels)
            }
        }
        return EverParakeetAudio.sanitized(out)
    }
}
