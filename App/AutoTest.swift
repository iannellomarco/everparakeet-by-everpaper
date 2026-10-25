import Foundation

/// `-autotest` launch argument: loads the model, transcribes the bundled samples (twice each), replays the longest
/// sample through the live path in 0.7 s steps, prints everything to stdout and exits.
@MainActor
enum AutoTest {
    static func say(_ s: String) {
        FileHandle.standardOutput.write(Data((s + "\n").utf8))
    }

    static func run() async {
        let host = EngineHost()
        say("AUTOTEST begin firstLaunch=\(host.firstLaunch)")
        host.setActive(true)            // this headless run is launched in front, so a repair is allowed
        await host.prepare()
        if case .failed(let why) = host.phase { say("AUTOTEST FAILED prepare: \(why)"); return }
        say("AUTOTEST backend=\(host.backend) prepare_ms=\(Int(host.loadMs)) health=\(host.health?.state.rawValue ?? "none") probeMs=\(host.health?.probeMilliseconds.map { String(format: "%.1f", $0) } ?? "-") \(host.health?.detail ?? "") \(host.note)")

        for sample in Sample.all {
            do {
                let pcm = try await AudioFile.load(sample.url)
                for run in 1...2 {
                    let t = try await host.transcribe(pcm, source: sample.id)
                    let d = t.diag ?? Diag()
                    let p = ProcessInfo.processInfo
                    say(String(format: "SAMPLE %@ run%d audio=%.1fs compute=%.0fms xRT=%.1f mel=%.0f enc=%.0f dec=%.0f windows=%d thermal=%d lowPower=%d backend=%@",
                               sample.id, run, t.audioSeconds, t.computeMs, t.realtimeFactor, d.mel, d.encoder, d.decoder, d.windows,
                               p.thermalState.rawValue, p.isLowPowerModeEnabled ? 1 : 0, t.backend))
                    say("TEXT \(t.text)")
                }
            } catch { say("AUTOTEST FAILED sample \(sample.id): \(error)") }
        }

        // Live path: feed the longest sample in 0.7 s steps as the microphone would.
        do {
            let pcm = try await AudioFile.load(Sample.all.last!.url)
            let box = SampleBox()
            let live = LiveTranscriber(host: host, box: box)
            let step = Int(0.7 * 16000)
            var fed = 0, n = 0
            while fed < pcm.count {
                let end = min(pcm.count, fed + step)
                box.append(Array(pcm[fed..<end]))
                fed = end
                await live.tick()
                n += 1
                if n % 12 == 0 || fed >= pcm.count {
                    say(String(format: "LIVE t=%.1fs tick=%.0fms commits=%d tail=\"%@\"", Double(fed) / 16000,
                               live.lastTickMs, live.commits, String(live.tail.suffix(70))))
                }
            }
            say(String(format: "LIVE done ticks=%d max_tick=%.0fms (budget 700)", n, live.maxTickMs))
            say("LIVE text-before-final \(live.text)")
            let final = try await live.finish(source: "live-sim")
            say(String(format: "LIVE final audio=%.1fs compute=%.0fms xRT=%.1f", final.audioSeconds, final.computeMs, final.realtimeFactor))
            say("TEXT \(final.text)")
        } catch { say("AUTOTEST FAILED live: \(error)") }
        // Short recording: one live chunk. Stopping must show the live text at once, without a second decode.
        do {
            let pcm = try await AudioFile.load(Sample.all[1].url)       // 10.7 s
            let box = SampleBox()
            let live = LiveTranscriber(host: host, box: box)
            let step = Int(0.7 * 16000)
            var fed = 0
            while fed < pcm.count {
                let end = min(pcm.count, fed + step)
                box.append(Array(pcm[fed..<end]))
                fed = end
                await live.tick()
            }
            let t0 = DispatchTime.now()
            let final = try await live.finish(source: "live-short")
            let ms = Double(DispatchTime.now().uptimeNanoseconds - t0.uptimeNanoseconds) / 1e6
            say(String(format: "LIVE short audio=%.1fs stop_to_result=%.1fms sameAsLive=%d", final.audioSeconds, ms,
                       final.text == live.text ? 1 : 0))
            say("TEXT \(final.text)")
        } catch { say("AUTOTEST FAILED live-short: \(error)") }
        say("AUTOTEST done")
    }
}
