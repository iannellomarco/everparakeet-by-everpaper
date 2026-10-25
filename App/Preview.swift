#if DEBUG
import Foundation

/// Debug builds only. `-preview <scene>` puts the app into a stubbed state without loading the model, so the screens can
/// be checked and photographed in the iOS Simulator. The scenes are staged, not recordings: the transcripts are
/// what the package produced for the bundled clips on a Mac (not run in the Simulator), and the compute times are set so that every result shows the
/// speed measured on an iPhone 17 Pro with the Core AI encoder (147x real time, "On iPhone" in the Standard model card).
///
/// Scenes: prepare, prepare-later, failed, empty, recording, working, result, long, samples, history.
enum PreviewScene: String {
    case prepare, prepareLater = "prepare-later", failed, empty, recording, working, result, long, samples, history

    /// Times real time, Core AI encoder, iPhone 17 Pro: the Standard model card, "On iPhone".
    static let measuredRealTime = 147.0

    static var requested: PreviewScene? {
        let args = CommandLine.arguments
        guard let i = args.firstIndex(of: "-preview"), i + 1 < args.count else { return nil }
        return PreviewScene(rawValue: args[i + 1])
    }
}

@MainActor
extension AppModel {
    func apply(_ scene: PreviewScene) {
        // Compute time that gives the measured speed for a clip of this length.
        func staged(_ source: String, _ text: String, seconds: Double) -> Transcript {
            Transcript(source: source, text: text, audioSeconds: seconds,
                       computeMs: seconds / PreviewScene.measuredRealTime * 1000, backend: "Core AI")
        }
        let english = staged("Short English",
            "Plants make their food from the sun by photosynthesis. They also provide shade.", seconds: 9.1)
        let mixed = staged("Italian, then English",
            "I leoni sono i felini più sociali e vivono in gruppi numerosi chiamati branchi. Finland is a great boating destination. The land of a thousand lakes has thousands of islands too in the lakes and in the coastal archipelago.",
            seconds: 18.4)
        let italian = staged("Short Italian",
            "Apia è la capitale delle isole Samoa, è situata sull'isola di Upolu e la sua popolazione conta poco meno di 40.000 abitanti.", seconds: 10.7)

        switch scene {
        case .prepare, .prepareLater, .failed:
            host.phase = scene == .failed ? .failed("The model files are not in the app. Run scripts/fetch-model.sh, then build and run again.") : .preparing
            host.firstLaunch = scene == .prepare
            host.compiling = scene == .prepare
            host.elapsed = scene == .prepare ? 74 : 1
            host.stage = scene == .prepare ? "Preparing the encoder for this iPhone" : "Loading encoder window 2 of 4"
        default:
            host.phase = .ready
            host.backend = "Core AI"
        }

        switch scene {
        case .empty, .prepare, .prepareLater, .failed: break
        case .recording:
            mode = .recording
            recordSeconds = 12
            levels = (0..<160).map { i in
                let t = Double(i)
                return min(1, max(0, 0.08 + 0.55 * abs(sin(t / 7)) * abs(sin(t / 3.1 + 1)) * (i % 23 < 4 ? 0.2 : 1)))
            }
            liveText = "Plants make their food from the sun by photosynthesis. They also"
        case .working:
            mode = .working
            workLabel = "Transcribing Long form"
            workProgress = 0.45
        case .result:
            current = english
        case .long:
            current = staged("Long form",
                "La maggior parte dei moderni telescopi destinati alla ricerca sono enormi strutture in zone sperdute dove le condizioni atmosferiche sono favorevoli. Ti potrebbero anche interessare i pareri di governi diversi dal tuo, ma tieni comunque presente che tali pareri sono mirati ai cittadini di quei governi. La cioccolata calda è allo stesso livello di quella belga. I succhi di frutta sono costosi, ma molto buoni. Come il T-Rex, sono dotati di piedi con squame e artigli, depongono le uova e camminano sulle due zampe posteriori. Di solito offrono cibo, bevande e intrattenimenti esclusivi per tenere alto l'umore degli ospiti. Fare in modo che si trattengano.",
                seconds: 59.6)
        case .samples:
            current = english
            sheet = .samples
        case .history:
            history.clear()
            for t in [italian, mixed, english].reversed() { history.add(t) }
            current = english
            sheet = .history
        }
    }
}
#endif
