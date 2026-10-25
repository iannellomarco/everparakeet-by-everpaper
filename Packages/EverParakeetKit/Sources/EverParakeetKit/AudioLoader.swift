// Portions Copyright 2026 Apple Inc. Modifications Copyright 2026 Everpaper.
//
// The chunked file reader is adapted from MelSpectrogram.readAllSamples (Sources/CoreAISpeech/MelSpectrogram.swift
// in Apple's coreai-models repository).
//
// This whole file is distributed under Apple's BSD 3-clause licence (full text in
// swift/THIRD_PARTY_NOTICES):
//
// Redistribution and use in source and binary forms, with or without modification, are permitted provided
// that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the
//    following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and
//    the following disclaimer in the documentation and/or other materials provided with the distribution.
// 3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or
//    promote products derived from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED
// WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
// PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
// ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR
// TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
// ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import AVFoundation
import Foundation

public enum EverParakeetAudio {
    /// Reads an audio file, averages the channels to mono, replaces NaN and infinite samples with 0 and
    /// resamples linearly to `targetRate` (16 kHz), the same conversion as the Python runner.
    public static func loadAudio(url: URL, targetRate: Int = 16000) throws -> [Float] {
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw EverParakeetError.missing("audio file \(url.path)")
        }
        let file: AVAudioFile
        do { file = try AVAudioFile(forReading: url) } catch {
            throw EverParakeetError.invalid("\(url.lastPathComponent) cannot be read as an audio file (\((error as NSError).localizedDescription))")
        }
        let format = file.processingFormat
        let channels = Int(format.channelCount)
        let chunk: AVAudioFrameCount = 1 << 16
        guard file.length > 0, let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: chunk) else { return [] }
        var mono: [Float] = []
        mono.reserveCapacity(Int(file.length))
        // AVAudioFile.read(into:) may return fewer frames than asked (it stops at packet boundaries),
        // so read in a loop until the file is exhausted.
        while file.framePosition < file.length {
            let want = AVAudioFrameCount(min(Int64(chunk), file.length - file.framePosition))
            try file.read(into: buffer, frameCount: want)
            let n = Int(buffer.frameLength)
            if n == 0 { break }
            guard let data = buffer.floatChannelData else { return [] }
            if channels == 1 {
                mono.append(contentsOf: UnsafeBufferPointer(start: data[0], count: n))
            } else {
                for i in 0..<n { var s: Float = 0; for c in 0..<channels { s += data[c][i] }; mono.append(s / Float(channels)) }
            }
        }
        return resample(sanitized(mono), from: Int(format.sampleRate), to: targetRate)
    }

    /// Replaces NaN and infinite samples with 0 (returns the input unchanged when all samples are finite).
    public static func sanitized(_ x: [Float]) -> [Float] {
        guard x.contains(where: { !$0.isFinite }) else { return x }
        return x.map { $0.isFinite ? $0 : 0 }
    }

    public static func resample(_ x: [Float], from source: Int, to target: Int) -> [Float] {
        if source == target || x.isEmpty { return x }
        let outLen = max(1, Int((Double(x.count) * Double(target) / Double(source)).rounded()))
        var out = [Float](repeating: 0, count: outLen)
        let last = Double(x.count - 1)
        for i in 0..<outLen {      // np.interp over linspace(0,1) grids
            let pos = outLen > 1 ? Double(i) / Double(outLen - 1) * last : 0
            let i0 = min(Int(pos), x.count - 1), i1 = min(i0 + 1, x.count - 1)
            let frac = Float(pos - Double(i0))
            out[i] = x[i0] + (x[i1] - x[i0]) * frac
        }
        return out
    }
}
