import Accelerate
import Foundation

/// The decoder and joint weights (fp16, or int8 + scales dequantized at load). Immutable and shared by every lane of a
/// model; the mutable state (LSTM state, scratch) lives in `CPUTDTDecoder`.
public final class DecoderWeights: @unchecked Sendable {
    public let H: Int, V: Int, D: Int
    public let blank: Int
    public let durations: [Int]
    public let maxSymbols: Int
    public let layers: Int
    let embedding16: [Float16]          // [V, H]
    let wLSTM16: [[Float16]]            // per layer [4H, 2H] = [W_ih | W_hh]
    let wProj16: [Float16]
    let wHead16: [Float16]              // [V + D, H]
    let bLSTM: [[Float]]                // per layer [4H] = b_ih + b_hh
    let bProj: [Float], bHead: [Float]

    public init(root: URL, config: EverParakeetConfig) throws {
        H = config.hidden; V = config.vocabSize; D = config.durations.count
        layers = config.layers
        blank = config.blankID; durations = config.durations; maxSymbols = config.maxSymbols
        let st = try SafeTensors(url: root.appendingPathComponent("models/decoder_joint.safetensors"))
        embedding16 = try st.halves("decoder.embedding.weight")
        var w: [[Float16]] = [], b: [[Float]] = []
        for l in 0..<config.layers {
            let ih = try st.halves("decoder.lstm.weight_ih_l\(l)"), hh = try st.halves("decoder.lstm.weight_hh_l\(l)")
            var cat = [Float16](repeating: 0, count: 4 * H * 2 * H)
            for r in 0..<(4 * H) {
                for c in 0..<H { cat[r * 2 * H + c] = ih[r * H + c]; cat[r * 2 * H + H + c] = hh[r * H + c] }
            }
            let bi = try st.floats("decoder.lstm.bias_ih_l\(l)"), bh = try st.floats("decoder.lstm.bias_hh_l\(l)")
            w.append(cat); b.append(zip(bi, bh).map { $0 + $1 })
        }
        wLSTM16 = w; bLSTM = b
        wProj16 = try st.halves("decoder.decoder_projector.weight"); bProj = try st.floats("decoder.decoder_projector.bias")
        wHead16 = try st.halves("joint.head.weight"); bHead = try st.floats("joint.head.bias")
        guard embedding16.count == V * H, wProj16.count == H * H, wHead16.count == (V + D) * H, H % 8 == 0 else {
            throw EverParakeetError.invalid("decoder_joint.safetensors does not match config.json (hidden \(H), vocab \(V))")
        }
    }
}

/// Greedy TDT decoding of the EverParakeet prediction + joint networks on the CPU, for one lane.
/// Logic is a line-for-line port of `_TDTDecoder.run` in the Python release (checkpoint fp16 weights, fp32
/// activations/accumulation). A decoder instance is not thread-safe: a lane runs it from its own serial queue.
public final class CPUTDTDecoder {
    /// Decision parameters of one decode.
    public struct Params: Sendable {
        /// Subtracted from the blank logit before the argmax (0 = the model as trained).
        public var blankBias: Float = 0
        /// Upper bound on the TDT duration jump in frames (4 = the model as trained).
        public var durationCap = 4
        /// Repetition guard (active only with a blank bias): a token that would make the last `loopRepeats` blocks of an
        /// n-gram (n = 2...8) identical, or the last `loopRepeats1` tokens identical, is replaced by blank.
        public var loopRepeats = 5
        public var loopRepeats1 = 8
        /// Size of the top-k list handed to a `TokenChooser`.
        public var chooserTopK = 16
        /// The cancel flag is polled every this many joint calls (a joint call is about 0.2 ms).
        public var cancelStride = 16
        public init() {}
    }

    let w: DecoderWeights
    let H: Int, V: Int, D: Int
    let blank: Int
    let layers: Int
    let threads: Int
    // scratch and state
    private var gates: [Float]
    private var xh: [[Float]]           // per layer [2H] = [input ; h]
    private var cell: [[Float]]         // per layer [H]
    private var decOut: [Float]
    private var act: [Float]
    private var logits: [Float]
    private var probScratch: [Float]
    private var sig: [Float]
    private var tnh: [Float]
    private var emittedTail: [Int] = []
    public private(set) var stepCalls = 0
    public private(set) var jointCalls = 0

    /// - Parameter threads: workers of the NEON gemv (1 = inline, no GCD hop).
    public init(weights: DecoderWeights, threads: Int = 4) {
        w = weights
        H = weights.H; V = weights.V; D = weights.D; blank = weights.blank; layers = weights.layers
        self.threads = max(1, threads)
        gates = [Float](repeating: 0, count: 4 * H)
        xh = Array(repeating: [Float](repeating: 0, count: 2 * H), count: layers)
        cell = Array(repeating: [Float](repeating: 0, count: H), count: layers)
        decOut = [Float](repeating: 0, count: H)
        act = [Float](repeating: 0, count: H)
        logits = [Float](repeating: 0, count: V + D)
        probScratch = [Float](repeating: 0, count: V)
        sig = [Float](repeating: 0, count: 3 * H)
        tnh = [Float](repeating: 0, count: 2 * H)
    }

    /// y = bias + W x, W [m, n] row-major.
    @inline(__always)
    private func gemv(w16: [Float16], bias: [Float], x: UnsafePointer<Float>, y: UnsafeMutablePointer<Float>, m: Int, n: Int) {
        w16.withUnsafeBufferPointer { wp in
            bias.withUnsafeBufferPointer { b in
                gemvHalf(wp.baseAddress!, x, b.baseAddress!, y, m: m, n: n, threads: threads)
            }
        }
    }

    /// Gate nonlinearities + cell/hidden update for one layer, from `gates` (i | f | g | o).
    private func lstmPointwise(layer l: Int) {
        let h = H
        var n = Int32(3 * h)
        // sigmoid(i), sigmoid(f) over [0, 2h) and sigmoid(o) over [3h, 4h): compute 1 / (1 + exp(-x)) on a packed copy
        gates.withUnsafeBufferPointer { g in
            sig.withUnsafeMutableBufferPointer { s in
                for k in 0..<(2 * h) { s[k] = -g[k] }
                for k in 0..<h { s[2 * h + k] = -g[3 * h + k] }
            }
            tnh.withUnsafeMutableBufferPointer { t in for k in 0..<h { t[k] = g[2 * h + k] } }
        }
        sig.withUnsafeMutableBufferPointer { s in
            vvexpf(s.baseAddress!, s.baseAddress!, &n)
            for k in 0..<(3 * h) { s[k] = 1 / (1 + s[k]) }
        }
        tnh.withUnsafeMutableBufferPointer { t in var m = Int32(h); vvtanhf(t.baseAddress!, t.baseAddress!, &m) }
        // c = f*c + i*g ; h = o*tanh(c)
        for k in 0..<h {
            let c = sig[h + k] * cell[l][k] + sig[k] * tnh[k]
            cell[l][k] = c
            tnh[h + k] = c
        }
        tnh.withUnsafeMutableBufferPointer { t in
            var m = Int32(h)
            vvtanhf(t.baseAddress! + h, t.baseAddress! + h, &m)
        }
        for k in 0..<h {
            let hv = sig[2 * h + k] * tnh[h + k]
            xh[l][h + k] = hv
            if l + 1 < layers { xh[l + 1][k] = hv }
        }
    }

    /// One prediction-network step: embedding -> 2-layer LSTM (state advanced in place) -> projector -> decOut.
    private func step(_ token: Int) {
        stepCalls += 1
        for i in 0..<H { xh[0][i] = Float(w.embedding16[token * H + i]) }
        for l in 0..<layers {
            xh[l].withUnsafeBufferPointer { x in
                gates.withUnsafeMutableBufferPointer { g in
                    gemv(w16: w.wLSTM16[l], bias: w.bLSTM[l], x: x.baseAddress!, y: g.baseAddress!, m: 4 * H, n: 2 * H)
                }
            }
            // torch LSTM gate order: i, f, g, o ; i,f,o = sigmoid, g = tanh (vForce, vectorised)
            lstmPointwise(layer: l)
        }
        xh[layers - 1].withUnsafeBufferPointer { x in
            decOut.withUnsafeMutableBufferPointer { y in
                gemv(w16: w.wProj16, bias: w.bProj, x: x.baseAddress! + H, y: y.baseAddress!, m: H, n: H)
            }
        }
    }

    private func wouldLoop(_ token: Int, _ p: Params) -> Bool {
        let n = emittedTail.count + 1
        @inline(__always) func at(_ i: Int) -> Int { i == n - 1 ? token : emittedTail[i] }
        for len in 1...8 {
            let reps = len == 1 ? p.loopRepeats1 : p.loopRepeats
            if reps < 2 || n < len * reps { continue }
            var same = true
            outer: for k in 1..<reps {
                for j in 0..<len where at(n - 1 - j) != at(n - 1 - j - k * len) { same = false; break outer }
            }
            if same { return true }
        }
        return false
    }

    /// Decodes `validFrames` encoder frames (`encoder` is [frames, H] row-major) and returns the non-blank tokens.
    /// `frameOffset` is added to every token's frame (the first frame of this window in the whole recording).
    ///
    /// With default `params` and no chooser the emitted ids are exactly those of the 1.1.x decoder.
    /// - Throws: `CancellationError` when `cancel` is set (polled every `params.cancelStride` joint calls).
    public func decode(encoder: UnsafePointer<Float>, validFrames: Int, frameOffset: Int = 0, params: Params = Params(),
                       chooser: TokenChooser? = nil, piece: ((Int) -> String)? = nil,
                       cancel: CancelFlag? = nil) throws -> [TimedToken] {
        stepCalls = 0; jointCalls = 0
        emittedTail.removeAll(keepingCapacity: true)
        for l in 0..<layers {
            for i in 0..<(2 * H) { xh[l][i] = 0 }
            for i in 0..<H { cell[l][i] = 0 }
        }
        chooser?.reset()
        step(blank)
        let bias = params.blankBias
        let stride = max(1, params.cancelStride)
        var tokens: [TimedToken] = []
        var frame = 0, emitted = 0
        while frame < validFrames {
            if jointCalls % stride == 0, let cancel, cancel.isCancelled { throw CancellationError() }
            let enc = encoder + frame * H
            for i in 0..<H { act[i] = max(enc[i] + decOut[i], 0) }
            jointCalls += 1
            act.withUnsafeBufferPointer { x in
                logits.withUnsafeMutableBufferPointer { y in
                    gemv(w16: w.wHead16, bias: w.bHead, x: x.baseAddress!, y: y.baseAddress!, m: V + D, n: H)
                }
            }
            let rawBlank = logits[blank]
            if bias != 0 { logits[blank] -= bias }
            var token = 0, best = -Float.infinity
            for i in 0..<V where logits[i] > best { best = logits[i]; token = i }
            var confidence: Float = 0
            if token != blank {
                if bias != 0 { logits[blank] = rawBlank }       // confidence and top-k use the model's own logits
                confidence = logits.withUnsafeBufferPointer { l in
                    probScratch.withUnsafeMutableBufferPointer { s in
                        Softmax.probability(of: token, in: l.baseAddress!, count: V, scratch: s.baseAddress!)
                    }
                }
                if let chooser {
                    let top = logits.withUnsafeBufferPointer { Softmax.topK($0.baseAddress!, count: V, k: params.chooserTopK) }
                    let choice = chooser.choose(original: token, topK: top.ids, logits: top.values, blankId: blank,
                                                 originalProb: confidence)
                    if choice.id != blank, choice.id >= 0, choice.id < V { token = choice.id; confidence = choice.confidence }
                }
                if bias != 0 {
                    if wouldLoop(token, params) { token = blank } else { emittedTail.append(token) }
                }
            }
            var di = 0; best = -Float.infinity
            for i in 0..<D where logits[V + i] > best { best = logits[V + i]; di = i }
            var duration = min(w.durations[di], params.durationCap)
            if token == blank {
                duration = max(duration, 1)
                emitted = 0
            } else {
                chooser?.observe(piece: piece?(token) ?? "")
                step(token)
                emitted += 1
                if duration == 0 && emitted >= w.maxSymbols { duration = 1; emitted = 0 }
                tokens.append(TimedToken(id: token, frame: frame + frameOffset, durationFrames: duration, confidence: confidence))
            }
            frame += duration
        }
        return tokens
    }
}
