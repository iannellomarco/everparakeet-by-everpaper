import AVFoundation
import Foundation
import Observation
import UIKit

/// A bundled test clip: real human speech from FLEURS (Google, CC-BY-4.0), trimmed and joined. See the README.
struct Sample: Identifiable, Hashable {
    let id: String          // file name without extension
    let title: String
    let subtitle: String
    var url: URL { Bundle.main.url(forResource: id, withExtension: "wav", subdirectory: "Samples")! }

    static let all = [
        Sample(id: "english", title: "Short English", subtitle: "English, 9.1 s"),
        Sample(id: "italian", title: "Short Italian", subtitle: "Italiano, 10.7 s"),
        Sample(id: "italian_then_english", title: "Italian, then English", subtitle: "Two sentences, two languages, 18.4 s"),
        Sample(id: "long_italian", title: "Long form", subtitle: "Italiano, five recordings joined, 59.6 s, longer than one 30 s window"),
    ]
}

@MainActor @Observable
final class AppModel {
    enum Mode: Equatable { case idle, recording, working }
    enum Sheet: String, Identifiable { case samples, history; var id: String { rawValue } }

    let host = EngineHost()
    let history = HistoryStore()

    var mode: Mode = .idle {
        didSet { host.setRecording(mode == .recording) }
    }
    var sheet: Sheet?
    var current: Transcript?            // the transcript on screen, with its stats
    var liveText = ""                   // streaming text while recording
    var levels: [Double] = []           // recent input levels, 0...1, oldest first
    var recordSeconds = 0.0
    var workLabel = ""
    var workProgress: Double?           // nil = indeterminate
    var errorMessage: String?

    private var recorder: MicRecorder?
    private var live: LiveTranscriber?
    private var tasks: [Task<Void, Never>] = []
    private var tickTask: Task<Void, Never>?
    private var recordStart = Date()

    var ready: Bool { host.phase == .ready }
    var displayText: String { mode == .recording ? liveText : (current?.text ?? "") }

    func start() async { await host.prepare() }

    // MARK: Record

    func toggleRecord() {
        switch mode {
        case .idle: Task { await startRecording() }
        case .recording: Task { await stopRecording() }
        case .working: break
        }
    }

    private func startRecording() async {
        guard ready else { return }
        guard await MicRecorder.requestPermission() else {
            errorMessage = "Microphone access is off. Turn it on in Settings > EverParakeet."
            return
        }
        let box = SampleBox()
        let recorder = MicRecorder()
        do { try recorder.start(into: box) } catch {
            errorMessage = "The microphone could not start (\(error.localizedDescription))."
            return
        }
        self.recorder = recorder
        let live = LiveTranscriber(host: host, box: box)
        self.live = live
        current = nil
        liveText = ""
        levels = []
        recordSeconds = 0
        recordStart = Date()
        mode = .recording
        UIImpactFeedbackGenerator(style: .medium).impactOccurred()
        UIApplication.shared.isIdleTimerDisabled = true
        let tick = Task { [weak self] in                      // re-decode about every 0.7 s
            while !Task.isCancelled {
                let t0 = Date()
                await live.tick()
                self?.liveText = live.text
                let rest = 0.7 - Date().timeIntervalSince(t0)
                try? await Task.sleep(for: .milliseconds(Int(max(0.05, rest) * 1000)))
            }
        }
        tickTask = tick
        tasks = [
            tick,
            Task { [weak self] in                      // level meter and clock
                while !Task.isCancelled {
                    guard let self else { return }
                    levels.append(Double(min(1, box.level * 5)))
                    if levels.count > 160 { levels.removeFirst(levels.count - 160) }
                    recordSeconds = Date().timeIntervalSince(recordStart)
                    try? await Task.sleep(for: .milliseconds(80))
                }
            },
        ]
    }

    func stopRecording() async {
        guard mode == .recording, let live, let recorder else { return }
        recorder.stop()
        tasks.forEach { $0.cancel() }
        tasks = []
        if let tickTask { await tickTask.value }        // an in-flight tick stops within milliseconds
        tickTask = nil
        UIApplication.shared.isIdleTimerDisabled = false
        UIImpactFeedbackGenerator(style: .light).impactOccurred()
        levels = []
        mode = .working
        workLabel = "Finishing the transcript"
        workProgress = nil
        defer { mode = .idle; self.live = nil; self.recorder = nil }
        guard live.box.count > 8000 else { return }
        do {
            let t = try await live.finish(source: "Microphone")
            show(t)
        } catch { errorMessage = "\(error)" }
    }

    // MARK: Files and samples

    func transcribe(file url: URL, name: String, scoped: Bool = false) {
        guard mode == .idle, ready else { return }
        mode = .working
        workLabel = "Reading \(name)"
        workProgress = nil
        current = nil
        Task {
            let access = scoped && url.startAccessingSecurityScopedResource()
            defer { if access { url.stopAccessingSecurityScopedResource() }; mode = .idle }
            do {
                let pcm = try await AudioFile.load(url)
                guard pcm.count > 1600 else { throw AudioError.noAudio }
                workLabel = "Transcribing \(name)"
                workProgress = pcm.count > 30 * 16000 ? 0 : nil
                let t = try await host.transcribe(pcm, source: name) { p in
                    Task { @MainActor in self.workProgress = p }
                }
                show(t)
            } catch { errorMessage = (error as? LocalizedError)?.errorDescription ?? "\(error)" }
        }
    }

    func run(sample: Sample) { transcribe(file: sample.url, name: sample.title) }

    func show(_ t: Transcript) {
        current = t
        history.add(t)
        UINotificationFeedbackGenerator().notificationOccurred(.success)
    }

    func copy() {
        guard let t = current?.text, !t.isEmpty else { return }
        UIPasteboard.general.string = t
        UINotificationFeedbackGenerator().notificationOccurred(.success)
    }
}
