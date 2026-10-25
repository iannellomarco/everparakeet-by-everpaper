import Foundation

/// Greek final-sigma handling for decoded Parakeet text. Swift port of greek/postprocess.py (same character classes, same rules).
///
/// The 8192-piece tokenizer has no piece for the final sigma U+03C2: the model can only write it as the `<unk>` id (id 0), or,
/// after a convention change in training, as a plain σ.  Two deterministic rules, applied in this order:
///   R2  an `<unk>` that follows a Greek letter and ends the word is the final sigma  -> "ς"; every other `<unk>` is dropped.
///   R1  a lower-case σ that ends a Greek word (preceded by a Greek letter, not followed by a letter, apostrophe or hyphen) -> "ς".
/// One-letter tokens ("σ", "σ'"), σ after a digit or Latin letter and capital Σ are never touched; non-Greek text is unchanged.
public enum GreekFinalSigma {
    /// What `TokenDecoder` renders for the `<unk>` id. A private-use scalar: it can never occur in decoded text.
    public static let unkMarker: Unicode.Scalar = "\u{E000}"

    // G  U+0370...U+03FF, U+1F00...U+1FFF     M  U+0300...U+036F     CONT  G, M, ' U+2019 U+02BC - U+2010 U+2011
    @inline(__always) static func isGreek(_ s: Unicode.Scalar) -> Bool {
        let v = s.value
        return (0x0370...0x03FF).contains(v) || (0x1F00...0x1FFF).contains(v)
    }
    @inline(__always) static func isMark(_ s: Unicode.Scalar) -> Bool { (0x0300...0x036F).contains(s.value) }
    @inline(__always) static func continuesWord(_ s: Unicode.Scalar) -> Bool {
        if isGreek(s) || isMark(s) { return true }
        switch s.value {
        case 0x27, 0x2019, 0x02BC, 0x2D, 0x2010, 0x2011: return true
        default: return false
        }
    }

    /// R2 then R1.
    public static func apply(_ text: String, unk: Unicode.Scalar = unkMarker) -> String {
        // R2: the marker after a Greek letter (or mark), not followed by one, is the final sigma; any other marker is removed.
        let s = Array(text.unicodeScalars)
        var t = [Unicode.Scalar]()
        t.reserveCapacity(s.count)
        for i in s.indices {
            guard s[i] == unk else { t.append(s[i]); continue }
            let before = i > 0 && (isGreek(s[i - 1]) || isMark(s[i - 1]))
            let after = i + 1 < s.count && (isGreek(s[i + 1]) || isMark(s[i + 1]))
            if before && !after { t.append("\u{03C2}") }
        }
        // R1: σ -> ς at the end of a Greek word (looking back over at most two combining marks).
        var out = t
        for i in t.indices where t[i] == "\u{03C3}" {
            var j = i - 1, marks = 0
            while j >= 0, isMark(t[j]), marks < 2 { j -= 1; marks += 1 }
            let prevIsGreekLetter = j >= 0 && isGreek(t[j]) && !isMark(t[j])
            let next = i + 1 < t.count ? t[i + 1] : nil
            if prevIsGreekLetter && !(next.map(continuesWord) ?? false) { out[i] = "\u{03C2}" }
        }
        var view = String.UnicodeScalarView()
        view.append(contentsOf: out)
        return String(view)
    }
}
