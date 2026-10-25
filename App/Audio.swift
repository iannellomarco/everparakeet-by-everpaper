import AVFoundation
import Foundation

/// Thread-safe growing buffer of 16 kHz mono samples (written by the audio tap, read by the live decoder).
final class SampleBox: @unchecked Sendable {
    private let lock = NSLock()
    private var samples: [Float] = []
    private var rms: Float = 0
    var count: Int { lock.lock(); defer { lock.unlock() }; return samples.count }
    var level: Float { lock.lock(); defer { lock.unlock() }; return rms }
    func append(_ p: UnsafeBufferPointer<Float>) {
        var e: Float = 0
        for v in p { e += v * v }
        lock.lock(); samples.append(contentsOf: p); rms = (e / Float(max(1, p.count))).squareRoot(); lock.unlock()
    }
    func append(_ a: [Float]) { a.withUnsafeBufferPointer { append($0) } }
    func snapshot(_ r: Range<Int>) -> [Float] {
        lock.lock(); defer { lock.unlock() }
        let hi = min(r.upperBound, samples.count), lo = min(r.lowerBound, hi)
        return Array(samples[lo..<hi])
    }
}

enum AudioError: LocalizedError {
    case noAudio, unreadable(String)
    var errorDescription: String? {
        switch self {
        case .noAudio: return "This file has no audio track."
        case .unreadable(let s): return "The audio could not be read (\(s))."
        }
    }
}

enum AudioFile {
    /// Decodes any audio or video file (wav, m4a, mp3, mp4, mov, ...) to 16 kHz mono Float32.
    static func load(_ url: URL) async throws -> [Float] {
        let asset = AVURLAsset(url: url)
        let tracks = try await asset.loadTracks(withMediaType: .audio)
        guard let track = tracks.first else { throw AudioError.noAudio }
        let reader = try AVAssetReader(asset: asset)
        let out = AVAssetReaderTrackOutput(track: track, outputSettings: [
            AVFormatIDKey: kAudioFormatLinearPCM, AVSampleRateKey: 16000, AVNumberOfChannelsKey: 1,
            AVLinearPCMBitDepthKey: 32, AVLinearPCMIsFloatKey: true,
            AVLinearPCMIsBigEndianKey: false, AVLinearPCMIsNonInterleaved: false,
        ])
        guard reader.canAdd(out) else { throw AudioError.unreadable("unsupported track") }
        reader.add(out)
        guard reader.startReading() else { throw AudioError.unreadable(reader.error?.localizedDescription ?? "start") }
        var pcm: [Float] = []
        while let buffer = out.copyNextSampleBuffer() {
            guard let block = CMSampleBufferGetDataBuffer(buffer) else { continue }
            let length = CMBlockBufferGetDataLength(block)
            var chunk = [Float](repeating: 0, count: length / 4)
            chunk.withUnsafeMutableBytes { _ = CMBlockBufferCopyDataBytes(block, atOffset: 0, dataLength: length, destination: $0.baseAddress!) }
            pcm.append(contentsOf: chunk)
        }
        if reader.status == .failed { throw AudioError.unreadable(reader.error?.localizedDescription ?? "read") }
        return pcm.map { $0.isFinite ? $0 : 0 }
    }
}

/// Microphone capture to 16 kHz mono, straight into a `SampleBox`.
final class MicRecorder {
    private let engine = AVAudioEngine()
    private(set) var running = false

    static func requestPermission() async -> Bool { await AVAudioApplication.requestRecordPermission() }

    func start(into box: SampleBox) throws {
        let session = AVAudioSession.sharedInstance()
        try session.setCategory(.record, mode: .default)
        try session.setActive(true)
        let input = engine.inputNode
        let inFormat = input.outputFormat(forBus: 0)
        let outFormat = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: 16000, channels: 1, interleaved: false)!
        guard let converter = AVAudioConverter(from: inFormat, to: outFormat) else { throw AudioError.unreadable("converter") }
        let ratio = 16000 / inFormat.sampleRate
        input.installTap(onBus: 0, bufferSize: 4096, format: inFormat) { buffer, _ in
            let capacity = AVAudioFrameCount(Double(buffer.frameLength) * ratio) + 64
            guard let out = AVAudioPCMBuffer(pcmFormat: outFormat, frameCapacity: capacity) else { return }
            var supplied = false
            var error: NSError?
            converter.convert(to: out, error: &error) { _, status in
                if supplied { status.pointee = .noDataNow; return nil }
                supplied = true; status.pointee = .haveData; return buffer
            }
            if out.frameLength > 0, let p = out.floatChannelData?[0] {
                box.append(UnsafeBufferPointer(start: p, count: Int(out.frameLength)))
            }
        }
        engine.prepare()
        try engine.start()
        running = true
    }

    func stop() {
        guard running else { return }
        engine.inputNode.removeTap(onBus: 0)
        engine.stop()
        running = false
        try? AVAudioSession.sharedInstance().setActive(false, options: .notifyOthersOnDeactivation)
    }
}
