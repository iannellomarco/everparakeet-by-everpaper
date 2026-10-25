import Foundation

// Pieces and words of the timed output.

extension TokenDecoder {
    /// Raw SentencePiece text of one id; a leading "\u{2581}" marks a word start. "" for blanks, ids outside the
    /// vocabulary and skipped ids (eos / pad / bos). The `<unk>` id returns `GreekFinalSigma.unkMarker`, so an app that
    /// assembles words itself can run `GreekFinalSigma.apply` on each finished word and get the same text as `decode`.
    public func piece(for id: Int) -> String {
        if id < 0 || id >= pieces.count || skip.contains(id) { return "" }
        if id == unkId { return String(Character(GreekFinalSigma.unkMarker)) }
        return pieces[id]
    }

    /// Assembles tokens into words. A piece that starts with "\u{2581}" starts a word; a piece that is only "\u{2581}"
    /// ends the open word; the other pieces continue it. The Greek final-sigma rule is applied to each finished word, so
    /// the words agree with `decode` (up to whitespace). Words that end up empty (a lone `<unk>`) are dropped.
    public func words(from tokens: [TimedToken]) -> [TimedWord] {
        var out: [TimedWord] = []
        var text = "", start = 0, end = 0, count = 0
        var confidenceSum: Float = 0
        var open = false
        func flush() {
            guard open else { return }
            open = false
            let word = GreekFinalSigma.apply(text).trimmingCharacters(in: .whitespacesAndNewlines)
            if !word.isEmpty {
                out.append(TimedWord(text: word, startFrame: start, endFrame: end,
                                     confidence: confidenceSum / Float(max(1, count)), tokenCount: count))
            }
            text = ""; count = 0; confidenceSum = 0
        }
        for t in tokens {
            let p = piece(for: t.id)
            if p.isEmpty { continue }
            let normalized = p.replacingOccurrences(of: "\u{2581}", with: " ")
            if normalized.trimmingCharacters(in: .whitespaces).isEmpty { flush(); continue }
            if normalized.first == " " || !open {
                flush()
                open = true
                start = t.frame
            }
            text += normalized.trimmingCharacters(in: .whitespaces)
            end = t.frame + max(1, t.durationFrames)
            count += 1
            confidenceSum += t.confidence
        }
        flush()
        return out
    }
}
