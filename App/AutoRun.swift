#if DEBUG
import Foundation
import UIKit

/// Debug builds only. `-autorun <sampleId>` launch argument: once the engine is ready, runs that bundled sample through the normal UI path
/// (AppModel.run(sample:)), with the interface on screen, and prints one machine-readable line to stdout.
@MainActor
enum AutoRun {
    static var requested: String? {
        let a = CommandLine.arguments
        guard let i = a.firstIndex(of: "-autorun"), i + 1 < a.count else { return nil }
        return a[i + 1]
    }

    static func go(_ model: AppModel, id: String) async {
        while !model.ready {
            if case .failed(let why) = model.host.phase { AutoTest.say("AUTORUN FAILED prepare: \(why)"); exit(1) }
            try? await Task.sleep(for: .milliseconds(100))
        }
        try? await Task.sleep(for: .seconds(1))                 // let the UI settle
        guard let sample = Sample.all.first(where: { $0.id == id }) else { AutoTest.say("AUTORUN unknown sample \(id)"); exit(1) }
        for run in 1...3 {
            model.current = nil
            model.run(sample: sample)
            try? await Task.sleep(for: .milliseconds(200))
            while model.mode != .idle || model.current == nil { try? await Task.sleep(for: .milliseconds(50)) }
            guard let t = model.current else { continue }
            let d = t.diag ?? Diag()
            let p = ProcessInfo.processInfo
            AutoTest.say(String(format: "AUTORUN %@ run%d audio=%.1fs compute=%.0fms xRT=%.1f mel=%.0f enc=%.0f dec=%.0f windows=%d thermal=%d lowPower=%d backend=%@ health=%@ probeMs=%.1f note=%@",
                               id, run, t.audioSeconds, t.computeMs, t.realtimeFactor, d.mel, d.encoder, d.decoder, d.windows,
                               p.thermalState.rawValue, p.isLowPowerModeEnabled ? 1 : 0, t.backend, model.host.health?.state.rawValue ?? "none", model.host.health?.probeMilliseconds ?? 0, model.host.note))
            try? await Task.sleep(for: .seconds(2))
        }
        exit(0)
    }
}
#endif
