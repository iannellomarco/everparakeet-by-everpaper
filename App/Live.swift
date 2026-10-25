import Foundation
import Observation

/// Live transcript of a growing recording.
///
/// Every tick re-decodes the current chunk (at most about 24 s) and shows it after the text already committed.
/// When the chunk passes 24 s it is cut at its quietest 200 ms, decoded one last time and committed, and a new
/// chunk starts there. When recording stops, a recording of one chunk (24 s or less) shows its live text as the result;
/// a longer one is decoded once more as a whole (audio over 30 s takes the long-form path: LCS merge, 8 s overlap,
/// blank bias 1.25, repetition guard) and that text is the result.
@MainActor @Observable
final class LiveTranscriber {
    var committed = ""
    var tail = ""
    var commits = 0
    var lastTickMs = 0.0
    var maxTickMs = 0.0
    private(set) var decodedTo = 0          // samples the last completed tick had decoded

    let box: SampleBox
    private let host: EngineHost
    private var chunkStart = 0
    static let rate = 16000
    static let commitAfter = 24 * rate

    init(host: EngineHost, box: SampleBox) { self.host = host; self.box = box }

    var text: String { [committed, tail].filter { !$0.isEmpty }.joined(separator: " ") }

    func tick() async {
        var total = box.count
        guard total - chunkStart >= Self.rate / 2 else { return }
        let t0 = DispatchTime.now()
        if total - chunkStart > Self.commitAfter {
            let window = box.snapshot(chunkStart..<total)
            let cut = chunkStart + Self.quietest(window, low: 14 * Self.rate, high: 23 * Self.rate)
            let finished = await host.decode(box.snapshot(chunkStart..<cut))
            if Task.isCancelled { return }      // recording stopped: a cancelled decode returns "", keep what we have
            if !finished.isEmpty { committed += (committed.isEmpty ? "" : " ") + finished }
            chunkStart = cut
            commits += 1
            total = box.count
        }
        let decoded = await host.decode(box.snapshot(chunkStart..<total))
        if Task.isCancelled { return }
        tail = decoded
        decodedTo = total
        lastTickMs = Double(DispatchTime.now().uptimeNanoseconds - t0.uptimeNanoseconds) / 1e6
        maxTickMs = max(maxTickMs, lastTickMs)
    }

    /// The final text. A recording of one live chunk (24 s or less, nothing committed) whose last tick already decoded
    /// it up to the final few hundred milliseconds is shown as it is: that text is the same single-window decode a
    /// final pass would run. Anything longer, or with audio the last tick missed, gets one decode over the whole recording.
    func finish(source: String) async throws -> Transcript {
        let total = box.count
        if commits == 0, total <= Self.commitAfter, total - decodedTo <= Self.rate * 35 / 100, !tail.isEmpty {
            return Transcript(source: source, text: tail, audioSeconds: Double(total) / Double(Self.rate),
                              computeMs: lastTickMs, backend: host.backend)
        }
        let all = box.snapshot(0..<box.count)
        return try await host.transcribe(all, source: source)
    }

    /// Centre sample of the quietest 200 ms stretch in [low, high] (50 ms steps).
    static func quietest(_ x: [Float], low: Int, high: Int) -> Int {
        let width = rate / 5, hop = rate / 20
        var best = min(high, x.count), bestEnergy = Float.infinity
        var p = low
        while p + width <= min(high, x.count) {
            var e: Float = 0
            for i in p..<(p + width) { e += x[i] * x[i] }
            if e < bestEnergy { best = p + width / 2; bestEnergy = e }
            p += hop
        }
        return best
    }
}
