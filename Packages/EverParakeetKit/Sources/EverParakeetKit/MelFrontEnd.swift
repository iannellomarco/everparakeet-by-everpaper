// Portions Copyright 2026 Apple Inc. Modifications Copyright 2026 Everpaper.
//
// Adapted from CoreAISpeech.MelSpectrogram (Sources/CoreAISpeech/MelSpectrogram.swift in Apple's
// coreai-models repository): framing, pre-emphasis, filterbank and per-bin normalisation steps.
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

import Accelerate
import Foundation

/// Log-mel front end matching HF `ParakeetFeatureExtractor` (preemphasis 0.97, symmetric Hann 400 in a
/// 512 FFT, constant centre padding, slaney mel 128, natural log with 2^-24 guard, per-bin mean/std over
/// valid frames with Bessel correction, std + 1e-5). Structure follows Apple's CoreAISpeech.MelSpectrogram
/// (BSD-3) but computes in channel-major [128, T] (the encoder's native layout) with one sgemm for the
/// filterbank, like the reference's `mel_filters @ magnitudes`.
public final class MelFrontEnd {
    let nFFT: Int, win: Int, hop: Int, nMels: Int, nFreqs: Int
    let preemph: Float
    let logGuard: Float
    let window: [Float]
    let filterbank: [Float]          // [nMels, nFreqs] row-major
    private let fftSetup: FFTSetup
    private let log2n: vDSP_Length

    public init(config: EverParakeetConfig) {
        nFFT = config.nFFT; win = config.winLength; hop = config.hop; nMels = config.nMels
        nFreqs = nFFT / 2 + 1
        preemph = config.preemphasis
        logGuard = config.logGuard
        let denom = Double(win - 1)   // torch.hann_window(win, periodic=False)
        window = (0..<win).map { Float(0.5 * (1 - cos(2 * Double.pi * Double($0) / denom))) }
        filterbank = MelFrontEnd.slaneyFilterbank(sampleRate: Double(config.sampleRate), nFFT: nFFT, nMels: nMels)
        log2n = vDSP_Length(nFFT.trailingZeroBitCount)
        fftSetup = vDSP_create_fftsetup(log2n, FFTRadix(kFFTRadix2))!
    }

    deinit { vDSP_destroy_fftsetup(fftSetup) }

    /// librosa.filters.mel(sr, n_fft, n_mels, fmin=0, fmax=sr/2, norm="slaney") (float64 maths, float32 result).
    static func slaneyFilterbank(sampleRate: Double, nFFT: Int, nMels: Int) -> [Float] {
        let nFreqs = nFFT / 2 + 1
        let fSp = 200.0 / 3, minLogHz = 1000.0
        let minLogMel = minLogHz / fSp
        let logstep = log(6.4) / 27.0
        func h2m(_ f: Double) -> Double { f >= minLogHz ? minLogMel + log(f / minLogHz) / logstep : f / fSp }
        func m2h(_ m: Double) -> Double { m >= minLogMel ? minLogHz * exp(logstep * (m - minLogMel)) : fSp * m }
        let fMax = sampleRate / 2
        let lo = h2m(0), hi = h2m(fMax)
        let pts = (0..<nMels + 2).map { m2h(lo + (hi - lo) * Double($0) / Double(nMels + 1)) }
        let freqs = (0..<nFreqs).map { Double($0) * sampleRate / Double(nFFT) }
        var fb = [Float](repeating: 0, count: nMels * nFreqs)
        for m in 0..<nMels {
            let fl = pts[m], fc = pts[m + 1], fr = pts[m + 2]
            let enorm = 2.0 / (fr - fl)
            for k in 0..<nFreqs {
                let lower = (freqs[k] - fl) / (fc - fl)
                let upper = (fr - freqs[k]) / (fr - fc)
                fb[m * nFreqs + k] = Float(max(0, min(lower, upper)) * enorm)
            }
        }
        return fb
    }

    public struct Result {
        /// Channel-major [nMels, frames]; frames = 1 + N/hop, the trailing (non-valid) frames are zero.
        public var mel: [Float]
        public var frames: Int
        public var valid: Int
    }

    public func compute(_ pcm: [Float]) -> Result {
        let n = pcm.count
        let frames = 1 + n / hop
        let valid = n / hop      // floor((N + n_fft - n_fft) / hop)
        var mel = [Float](repeating: 0, count: nMels * frames)
        if valid == 0 { return Result(mel: mel, frames: frames, valid: valid) }

        // pre-emphasis: y[0] = x[0], y[i] = x[i] - 0.97 x[i-1]; centre zero padding of n_fft/2
        let pad = nFFT / 2
        var padded = [Float](repeating: 0, count: n + 2 * pad)
        padded.withUnsafeMutableBufferPointer { p in
            p[pad] = pcm[0]
            for i in 1..<max(n, 1) { p[pad + i] = pcm[i] - preemph * pcm[i - 1] }
        }

        let half = nFFT / 2
        let frameOffset = (nFFT - win) / 2
        var power = [Float](repeating: 0, count: nFreqs * valid)   // [nFreqs, valid]
        let fr = UnsafeMutablePointer<Float>.allocate(capacity: nFFT)
        let re = UnsafeMutablePointer<Float>.allocate(capacity: half)
        let im = UnsafeMutablePointer<Float>.allocate(capacity: half)
        defer { fr.deallocate(); re.deallocate(); im.deallocate() }
        var split = DSPSplitComplex(realp: re, imagp: im)
        padded.withUnsafeBufferPointer { pp in
            window.withUnsafeBufferPointer { wp in
                power.withUnsafeMutableBufferPointer { pw in
                    for t in 0..<valid {
                        for i in 0..<nFFT { fr[i] = 0 }
                        vDSP_vmul(pp.baseAddress! + t * hop + frameOffset, 1, wp.baseAddress!, 1, fr + frameOffset, 1, vDSP_Length(win))
                        fr.withMemoryRebound(to: DSPComplex.self, capacity: half) { c in
                            vDSP_ctoz(c, 2, &split, 1, vDSP_Length(half))
                        }
                        vDSP_fft_zrip(fftSetup, &split, 1, log2n, FFTDirection(FFT_FORWARD))
                        // zrip output is 2x the DFT; bin 0 / bin n/2 are packed into realp[0] / imagp[0].
                        // Reference: |X| = sqrt(re^2 + im^2) then squared.
                        func mag2(_ a: Float, _ b: Float) -> Float { let m = (a * a + b * b).squareRoot(); return m * m }
                        pw[0 * valid + t] = mag2(0.5 * re[0], 0)
                        pw[half * valid + t] = mag2(0.5 * im[0], 0)
                        for k in 1..<half { pw[k * valid + t] = mag2(0.5 * re[k], 0.5 * im[k]) }
                    }
                }
            }
        }
        // mel = filterbank [nMels, nFreqs] @ power [nFreqs, valid]  -> [nMels, valid] (written with row stride `frames`)
        var melValid = [Float](repeating: 0, count: nMels * valid)
        filterbank.withUnsafeBufferPointer { a in
            power.withUnsafeBufferPointer { b in
                melValid.withUnsafeMutableBufferPointer { c in
                    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, Int32(nMels), Int32(valid), Int32(nFreqs),
                                1, a.baseAddress!, Int32(nFreqs), b.baseAddress!, Int32(valid), 0, c.baseAddress!, Int32(valid))
                }
            }
        }
        // log, per-bin normalisation over valid frames (Bessel std, + 1e-5 outside the sqrt)
        let epsilon: Float = 1e-5
        var tmp = [Float](repeating: 0, count: valid)
        melValid.withUnsafeMutableBufferPointer { m in
            mel.withUnsafeMutableBufferPointer { o in
                for b in 0..<nMels {
                    let row = m.baseAddress! + b * valid
                    let out = o.baseAddress! + b * frames
                    for t in 0..<valid { row[t] = logf(row[t] + logGuard) }
                    var mean: Float = 0
                    vDSP_meanv(row, 1, &mean, vDSP_Length(valid))
                    var negMean = -mean
                    var sq: Float = 0
                    vDSP_vsadd(row, 1, &negMean, &tmp, 1, vDSP_Length(valid))   // x - mean
                    vDSP_svesq(tmp, 1, &sq, vDSP_Length(valid))
                    let std = valid > 1 ? (sq / Float(valid - 1)).squareRoot() : 0
                    let denom = std + epsilon
                    for t in 0..<valid { out[t] = (row[t] - mean) / denom }
                }
            }
        }
        return Result(mel: mel, frames: frames, valid: valid)
    }
}
