import Foundation

/// Equivalent of `tokenizer.decode(tokens, skip_special_tokens=True, group_tokens=False).strip()`
/// for this tokenizer, with blanks removed first.
///
/// Behaviour of the HF ParakeetTokenizer decode:
///  - no grouping of repeated ids (the HF default groups them CTC-style, which is wrong for TDT);
///  - ids of the tokenizer's special tokens (eos/pad) are skipped; ids outside the vocab are dropped;
///  - the other `<|...|>` added tokens are NOT special for skip_special_tokens and stay in the text;
///  - SentencePiece "▁" becomes a space, the whole string is stripped;
///  - the `<unk>` id is NOT skipped any more: the tokenizer has no piece for the Greek final sigma (U+03C2), so the
///    model can only write it as id 0.  `GreekFinalSigma.apply` turns an `<unk>` that ends a Greek word into "ς" and drops
///    every other `<unk>` (what skipping did), and writes a word-final σ as "ς".  Text that is not Greek is unchanged.
public struct TokenDecoder: Sendable {
    let pieces: [String]      // index = id, "" for ids without a piece
    let skip: Set<Int>        // eos / pad / bos
    let unkId: Int?           // "<unk>"

    public init(root: URL) throws {
        let dir = root.appendingPathComponent("tokenizer")
        let tj = try JSONSerialization.jsonObject(with: Data(contentsOf: dir.appendingPathComponent("tokenizer.json"))) as? [String: Any]
        guard let model = tj?["model"] as? [String: Any], let vocab = model["vocab"] as? [String: NSNumber]
        else { throw EverParakeetError.invalid("tokenizer.json vocab") }
        var table = [String](repeating: "", count: (vocab.values.map { $0.intValue }.max() ?? 0) + 1)
        for (piece, id) in vocab { table[id.intValue] = piece }
        // added tokens carry ids too (they overlap the vocab here, but honour them if not)
        if let added = tj?["added_tokens"] as? [[String: Any]] {
            for a in added {
                if let id = (a["id"] as? NSNumber)?.intValue, let c = a["content"] as? String, id < table.count, table[id].isEmpty {
                    table[id] = c
                }
            }
        }
        pieces = table
        var skip = Set<Int>()
        if let cfg = try JSONSerialization.jsonObject(with: Data(contentsOf: dir.appendingPathComponent("tokenizer_config.json"))) as? [String: Any] {
            for key in ["eos_token", "pad_token", "bos_token"] {
                if let name = cfg[key] as? String, let id = vocab[name] { skip.insert(id.intValue) }
            }
            unkId = (cfg["unk_token"] as? String).flatMap { vocab[$0]?.intValue }
        } else {
            unkId = nil
        }
        self.skip = skip
    }

    /// Direct construction from a piece table (tests, tools).
    public init(pieces: [String], skip: Set<Int> = [], unkId: Int? = nil) {
        self.pieces = pieces; self.skip = skip; self.unkId = unkId
    }

    public func decode(_ tokens: [Int]) -> String {
        // No CTC-style grouping: TDT legitimately emits the same token twice in a row
        // ("1966" is 1, 9, 6, 6). Blanks lie outside the vocabulary and are dropped below.
        var out = ""
        for t in tokens {
            if t < 0 || t >= pieces.count || skip.contains(t) { continue }
            if t == unkId { out.unicodeScalars.append(GreekFinalSigma.unkMarker) } else { out += pieces[t] }
        }
        // apply before the strip: a dropped leading/trailing <unk> must not leave its neighbouring space behind
        return GreekFinalSigma.apply(out.replacingOccurrences(of: "\u{2581}", with: " "))
            .trimmingCharacters(in: .whitespacesAndNewlines)
    }
}
