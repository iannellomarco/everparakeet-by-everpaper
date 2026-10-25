import Foundation

/// How audio longer than one encoder window (30 s) is cut, overlapped, stitched and decoded.
/// Audio of 30 s or less never uses these options (one piece, the decoder exactly as trained).
///
/// - `lcs` (default): fixed-stride windows overlapping by `overlap` seconds, each decoded completely on its own (fresh
///   prediction network, so no state leaks from one window to the next) and merged by the longest common subsequence of
///   words inside the overlap; the junction is the common word nearest the middle of the overlap, so every emitted word
///   sits at least `overlap / 2` seconds from the edge of the window that produced it.
/// - `legacy`: the 1.1.x behaviour, cut at the quietest 200 ms in the last 4 s of each 30 s piece, every piece decoded on
///   its own, pieces joined with a space. With `blankBias` 0 it reproduces 1.1.x bit for bit.
///
/// `blankBias` is subtracted from the blank logit before each greedy TDT decision. The compressed encoders are
/// under-confident on meeting audio and answer blank for whole stretches; a small bias removes most of that, at the price
/// of some insertions in overlapped speech. The repetition guard keeps a biased decode from falling into a loop
/// ("a little bit of a little bit of ...") that would flood the output. The value is calibrated per package: 1.25 is the
/// conservative default; a package can carry its own as `long_audio.blank_bias` in `models/config.json`.
public struct LongFormOptions: Sendable, Equatable {
    public enum Mode: String, Sendable { case legacy, lcs }
    public var mode: Mode = .lcs
    /// Length of one encoder window in seconds (at most 30).
    public var window = 30.0
    /// lcs: overlap in seconds between consecutive windows.
    public var overlap = 8.0
    /// Subtracted from the blank logit (long audio only unless `biasShort`); 0 = the model as trained. nil uses the
    /// package's `long_audio.blank_bias` when it has one, otherwise `defaultBlankBias`.
    public var blankBias: Float? = nil
    public static let defaultBlankBias: Float = 1.25
    /// Repetition guard: a token that would repeat an n-gram (n = 2...8) `loopRepeats` times in a row, or one token
    /// `loopRepeats1` times in a row, is dropped (0 or 1 turns the guard off).
    public var loopRepeats = 5
    public var loopRepeats1 = 8
    /// Cap on the TDT duration jump in frames (long audio only unless `biasShort`).
    public var durationCap = 4
    /// Apply `blankBias` / `durationCap` to short audio as well.
    public var biasShort = false
    /// Long-form off: audio over 30 s keeps the 1.1.x behaviour exactly (legacy cuts, no bias, no cap).
    public static var off: LongFormOptions {
        var o = LongFormOptions()
        o.mode = .legacy; o.blankBias = 0
        return o
    }
    public init() {}
}

/// One decoding window of the `lcs` plan.
struct LCSWindow: Equatable {
    var startFrame: Int          // first encoder frame of the window in the whole recording
    var startSample: Int
    var endSample: Int
    var previousEndFrame: Int    // end frame of the previous window (0 for the first)
}

enum LongFormPlan {
    /// Windows of `windowFrames` encoder frames starting every `windowFrames - overlapFrames` frames, until one reaches the
    /// end of the audio.
    static func lcsWindows(totalSamples: Int, samplesPerFrame fs: Int, windowFrames W: Int, overlapFrames ov: Int) -> [LCSWindow] {
        var out: [LCSWindow] = []
        var a = 0, prevEnd = 0
        while true {
            let startS = a * fs
            let endS = min(totalSamples, startS + W * fs)
            out.append(LCSWindow(startFrame: a, startSample: startS, endSample: endS, previousEndFrame: prevEnd))
            if endS >= totalSamples { break }
            prevEnd = a + W
            a += max(1, W - ov)
            if a * fs >= totalSamples { break }
        }
        return out
    }

    static func windowFrames(_ o: LongFormOptions, framesPerSecond fps: Double) -> Int {
        max(24, min(375, Int((o.window * fps).rounded())))
    }

    /// Centre sample (relative to `x`'s first sample) of the quietest `seconds`-wide stretch of `x`, 50 ms steps;
    /// `x.count` when there is no room for one stretch.
    static func quietestCentre(_ x: [Float], width seconds: Double, rate: Int) -> Int {
        let width = max(1, Int(seconds * Double(rate))), hop = max(1, Int(0.05 * Double(rate)))
        var best = x.count, bestEnergy = Float.infinity
        var position = 0
        while position + width <= x.count {
            var e: Float = 0
            x.withUnsafeBufferPointer { p in for i in position..<(position + width) { e += p[i] * p[i] } }
            if e < bestEnergy { best = position + width / 2; bestEnergy = e }
            position += hop
        }
        return best
    }
}

/// LCS merge of two overlapping windows' tokens.
enum LongFormMerge {
    struct Word { var first: Int; var end: Int; var frame: Int; var key: String }

    static func normalizedKey(_ s: String) -> String {
        String(s.lowercased().unicodeScalars.filter { CharacterSet.alphanumerics.contains($0) })
    }

    static func words(_ t: [TimedToken], piece: (Int) -> String) -> [Word] {
        var out: [Word] = []
        var buf = ""
        for (i, tk) in t.enumerated() {
            let p = piece(tk.id)
            if i == 0 || p.hasPrefix("\u{2581}") {
                if var w = out.popLast() { w.key = normalizedKey(buf); out.append(w) }
                out.append(Word(first: i, end: i + 1, frame: tk.frame, key: ""))
                buf = ""
            } else if var w = out.popLast() {
                w.end = i + 1; out.append(w)
            }
            buf += p
        }
        if var w = out.popLast() { w.key = normalizedKey(buf); out.append(w) }
        return out
    }

    /// Joins `right` (a window that starts at frame `overlapStart`) onto `left` (whose last window ends at frame
    /// `overlapEnd`) at the common word nearest the middle of the overlap; time cut at the middle if there is none.
    static func merge(_ left: [TimedToken], _ right: [TimedToken], overlapStart: Int, overlapEnd: Int,
                      piece: (Int) -> String) -> [TimedToken] {
        // Only the tail of `left` (from 2 s before the overlap) can take part in the merge; leaving the rest alone
        // keeps the cost of a recording linear in its length.
        var split = left.count
        while split > 0 && left[split - 1].frame >= overlapStart - 25 { split -= 1 }
        let head = Array(left[0..<split])
        var core = mergeCore(Array(left[split...]), right, overlapStart: overlapStart, overlapEnd: overlapEnd, piece: piece)
        // The same word can sit up to 20 frames earlier in the right window than in the left one, so the tokens after the
        // junction may start before the last left token. Times never go backwards: raise such a token to the running maximum.
        var last = head.last?.frame ?? 0
        for i in core.indices {
            if core[i].frame < last { core[i].frame = last } else { last = core[i].frame }
        }
        return head + core
    }

    static func mergeCore(_ left: [TimedToken], _ right: [TimedToken], overlapStart: Int, overlapEnd: Int,
                          piece: (Int) -> String) -> [TimedToken] {
        let lw = words(left, piece: piece), rw = words(right, piece: piece)
        let li = lw.indices.filter { lw[$0].frame >= overlapStart }
        let ri = rw.indices.filter { rw[$0].frame < overlapEnd }
        if li.isEmpty || ri.isEmpty { return left + right }
        let tol = 20                                        // frames (1.6 s)
        let n = li.count, m = ri.count
        var dp = [[Int]](repeating: [Int](repeating: 0, count: m + 1), count: n + 1)
        for i in 1...n {
            for j in 1...m {
                let a = lw[li[i - 1]], b = rw[ri[j - 1]]
                if !a.key.isEmpty, a.key == b.key, abs(a.frame - b.frame) <= tol {
                    dp[i][j] = dp[i - 1][j - 1] + 1
                } else {
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
                }
            }
        }
        var pairs: [(Int, Int)] = []
        var i = n, j = m
        while i > 0 && j > 0 {
            let a = lw[li[i - 1]], b = rw[ri[j - 1]]
            if !a.key.isEmpty, a.key == b.key, abs(a.frame - b.frame) <= tol, dp[i][j] == dp[i - 1][j - 1] + 1 {
                pairs.append((li[i - 1], ri[j - 1])); i -= 1; j -= 1
            } else if dp[i - 1][j] >= dp[i][j - 1] { i -= 1 } else { j -= 1 }
        }
        let mid = (overlapStart + overlapEnd) / 2
        if let best = pairs.min(by: { abs(lw[$0.0].frame - mid) < abs(lw[$1.0].frame - mid) }) {
            return Array(left[0..<lw[best.0].end]) + Array(right[rw[best.1].end...])
        }
        // No common word: cut at the middle of the overlap by time, but between words. Whole words that start before
        // the middle stay from the left window, whole words that start at or after it come from the right, so no
        // word is split between the two windows.
        let keepLeft = lw.last(where: { $0.frame < mid })?.end ?? 0
        let takeRight = rw.first(where: { $0.frame >= mid })?.first ?? right.count
        return Array(left[0..<keepLeft]) + Array(right[takeRight...])
    }
}
