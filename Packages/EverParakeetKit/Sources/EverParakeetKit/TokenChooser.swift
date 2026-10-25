import Accelerate
import Foundation

/// A lane-local hook that may replace a non-blank token the decoder was about to emit (vocabulary / custom-term
/// biasing). The decoder computes the top-k tokens on the raw joint logits and asks the chooser; blank emissions never
/// reach it. The lane calls a chooser only from its decoder queue, one decode at a time.
public protocol TokenChooser: AnyObject {
    /// Called at the start of every decoded window.
    func reset()
    /// Called for the piece of every token that was emitted (after the choice), in order. Raw SentencePiece text.
    func observe(piece: String)
    /// - Parameters:
    ///   - original: the token the decoder would emit (never blank). With a blank bias on, the bias is what made it
    ///     win over a blank that is higher on the raw logits, so `original` is not always `topK[0]`.
    ///   - topK: the `k` best token ids by raw joint logit (the model's own, before any blank bias), best first. Blank
    ///     may be among them, and `original` may not be.
    ///   - logits: the raw logits of `topK`, same order.
    ///   - blankId: the blank id.
    ///   - originalProb: softmax probability (over the whole vocabulary, blank included, raw logits) of `original`.
    /// - Returns: the token to emit and its confidence. Return `(original, originalProb)` to leave the decision alone.
    func choose(original: Int, topK: [Int], logits: [Float], blankId: Int, originalProb: Float) -> (id: Int, confidence: Float)
}

extension TokenChooser {
    public func reset() {}
    public func observe(piece: String) {}
}

enum Softmax {
    /// exp(x[index] - max) / sum(exp(x - max)) over `count` logits; `scratch` holds `count` floats.
    static func probability(of index: Int, in x: UnsafePointer<Float>, count: Int, scratch: UnsafeMutablePointer<Float>) -> Float {
        var peak: Float = 0
        vDSP_maxv(x, 1, &peak, vDSP_Length(count))
        var shift = -peak
        vDSP_vsadd(x, 1, &shift, scratch, 1, vDSP_Length(count))
        var n = Int32(count)
        vvexpf(scratch, scratch, &n)
        var sum: Float = 0
        vDSP_sve(scratch, 1, &sum, vDSP_Length(count))
        return sum > 0 ? scratch[index] / sum : 0
    }

    /// The `k` largest values with their indices, best first (ties: lower index first).
    static func topK(_ x: UnsafePointer<Float>, count: Int, k: Int) -> (ids: [Int], values: [Float]) {
        let k = min(k, count)
        guard k > 0 else { return ([], []) }
        var ids = [Int](), values = [Float]()
        ids.reserveCapacity(k + 1); values.reserveCapacity(k + 1)
        for i in 0..<count {
            let v = x[i]
            if ids.count == k, v <= values[k - 1] { continue }
            var j = ids.count
            while j > 0, values[j - 1] < v { j -= 1 }
            ids.insert(i, at: j); values.insert(v, at: j)
            if ids.count > k { ids.removeLast(); values.removeLast() }
        }
        return (ids, values)
    }
}
