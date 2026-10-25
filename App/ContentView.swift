import SwiftUI
import UniformTypeIdentifiers

struct ContentView: View {
    @Environment(AppModel.self) private var model
    @Environment(\.scenePhase) private var scenePhase
    @State private var showImporter = false

    var body: some View {
        @Bindable var model = model
        ZStack {
            PaperCanvas { MainScreen(showImporter: $showImporter) }
            if !model.ready { PrepareView().transition(.opacity) }
        }
        .fileImporter(isPresented: $showImporter, allowedContentTypes: [.audio, .movie, .mpeg4Movie, .quickTimeMovie]) { result in
            if case .success(let url) = result { model.transcribe(file: url, name: url.lastPathComponent, scoped: true) }
        }
        .sheet(item: $model.sheet) { sheet in
            switch sheet {
            case .samples: SamplesView()
            case .history: HistoryView()
            }
        }
        .alert("EverParakeet", isPresented: Binding(get: { model.errorMessage != nil }, set: { if !$0 { model.errorMessage = nil } })) {
            Button("OK", role: .cancel) {}
        } message: { Text(model.errorMessage ?? "") }
        .onChange(of: scenePhase, initial: true) { _, phase in
            model.host.setActive(phase == .active)          // the package repairs a bad Neural Engine plan only in front
            if phase == .background, model.mode == .recording { Task { await model.stopRecording() } }
        }
        .animation(.easeInOut(duration: 0.3), value: model.ready)
    }
}

// MARK: Main screen

private struct MainScreen: View {
    @Environment(AppModel.self) private var model
    @Environment(\.dynamicTypeSize) private var typeSize
    @Binding var showImporter: Bool

    /// The name shown at the top of the main screen. The home screen name (CFBundleDisplayName) stays "EverParakeet".
    private static let appTitle = "EverParakeet by EverPaper"

    var body: some View {
        VStack(spacing: 0) {
            ScrollViewReader { proxy in
                ScrollView {
                    VStack(alignment: .leading, spacing: 20) {
                        header
                        if model.mode == .idle && model.current == nil { tagline }
                        TranscriptPage()
                        Readouts()
                        if typeSize.isAccessibilitySize { Controls(showImporter: $showImporter) }   // no room to pin them
                        Color.clear.frame(height: 1).id("end")
                    }
                    .padding(.horizontal, 20)
                    .padding(.top, 8)
                    .padding(.bottom, 12)
                }
                .scrollIndicators(.hidden)
                .onChange(of: model.liveText) { _, _ in
                    guard model.mode == .recording else { return }
                    withAnimation(.easeOut(duration: 0.2)) { proxy.scrollTo("end", anchor: .bottom) }
                }
            }
            if !typeSize.isAccessibilitySize {
                Controls(showImporter: $showImporter)
                    .padding(.horizontal, 20)
                    .padding(.top, 8)
                    .padding(.bottom, 12)
            }
        }
    }

    private var header: some View {
        HStack(alignment: .center) {
            VStack(alignment: .leading, spacing: 6) {
                Text(Self.appTitle).font(Voice.title).foregroundStyle(Tone.ink)
                    .accessibilityAddTraits(.isHeader)
                status
            }
            Spacer()
            RoundGlassButton(systemImage: "clock.arrow.circlepath", label: "History") { model.sheet = .history }
        }
        .frame(minHeight: 44)
    }

    @ViewBuilder private var status: some View {
        switch model.mode {
        case .recording: StatusLine(kind: .recording, detail: typeSize.isAccessibilitySize ? nil : clock(model.recordSeconds))
        case .working: StatusLine(kind: .working)
        case .idle:
            StatusLine(kind: .ready, detail: model.host.backend.isEmpty || typeSize.isAccessibilitySize ? nil : model.host.backend)
        }
    }

    private var tagline: some View {
        Text("Speech recognition that runs on this iPhone. 25 languages, no network.")
            .font(Voice.proseSmall).foregroundStyle(Tone.inkSoft)
    }

    private func clock(_ s: Double) -> String { String(format: "%d:%02d", Int(s) / 60, Int(s) % 60) }
}

// MARK: Transcript

private struct TranscriptPage: View {
    @Environment(AppModel.self) private var model
    @ScaledMetric(relativeTo: .body) private var tool = 44
    @Environment(\.dynamicTypeSize) private var typeSize

    var body: some View {
        PageCard {
            VStack(alignment: .leading, spacing: 14) {
                switch model.mode {
                case .recording: recording
                case .working: working
                case .idle: if let t = model.current { result(t) } else { empty }
                }
            }
            .padding(20)
            .frame(maxWidth: .infinity, minHeight: 340, alignment: .topLeading)
        }
    }

    private var recording: some View {
        VStack(alignment: .leading, spacing: 16) {
            Waveline(levels: model.levels)
                .frame(height: 56)
                .padding(.horizontal, 16)
                .padding(.vertical, 14)
                .background(Tone.stage, in: .rect(cornerRadius: 16))
            if model.liveText.isEmpty {
                Text("Listening").font(Voice.prose).foregroundStyle(Tone.inkSoft)
            } else {
                Text(model.liveText)
                    .font(Voice.prose).lineSpacing(5).foregroundStyle(Tone.ink)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .motion(.settle, value: model.liveText)
            }
        }
    }

    private var working: some View {
        VStack(alignment: .leading, spacing: 14) {
            Text(model.workLabel).font(Voice.title).foregroundStyle(Tone.ink)
            HairlineProgress(fraction: model.workProgress)
            Text("On this iPhone. Nothing is uploaded.").font(Voice.detail).foregroundStyle(Tone.inkSoft)
        }
        .padding(.top, 24)
    }

    private var empty: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Talk, import a file, or try a sample.").font(Voice.title).foregroundStyle(Tone.ink)
            Text("Audio is transcribed on the Neural Engine. Airplane Mode works.")
                .font(Voice.proseSmall).foregroundStyle(Tone.inkSoft)
        }
        .padding(.top, 24)
    }

    private func result(_ t: Transcript) -> some View {
        VStack(alignment: .leading, spacing: 14) {
            let title = Text(t.source).kicker().foregroundStyle(Tone.inkSoft)
            let tools = HStack(spacing: 10) {
                RoundGlassButton(systemImage: "doc.on.doc", label: "Copy transcript") { model.copy() }
                ShareLink(item: t.text) {
                    Image(systemName: "square.and.arrow.up")
                        .font(.body.weight(.medium)).foregroundStyle(Tone.ink)
                        .frame(width: min(tool, 56), height: min(tool, 56)).contentShape(.circle)
                }
                .buttonStyle(.plain)
                .toolGlass(interactive: true, in: .circle)
                .accessibilityLabel("Share transcript")
            }
            if typeSize.isAccessibilitySize {
                VStack(alignment: .leading, spacing: 10) { title; tools }
            } else {
                HStack(alignment: .center, spacing: 10) { title.lineLimit(2); Spacer(minLength: 8); tools }
            }
            Text(t.text)
                .font(Voice.prose).lineSpacing(5).foregroundStyle(Tone.ink)
                .textSelection(.enabled)
                .frame(maxWidth: .infinity, alignment: .leading)
        }
    }
}

// MARK: Readouts

private struct Readouts: View {
    @Environment(AppModel.self) private var model
    @Environment(\.dynamicTypeSize) private var typeSize

    var body: some View {
        let t = model.current
        let cells: [(String, String, String?)] = [
            ("Audio", t.map { String(format: "%.1f", $0.audioSeconds) } ?? "-", t == nil ? nil : "s"),
            ("Compute", t.map { String(format: "%.0f", $0.computeMs) } ?? "-", t == nil ? nil : "ms"),
            ("Real time", t.map { String(format: "%.0f", $0.realtimeFactor) } ?? "-", t == nil ? nil : "x"),
        ]
        VStack(alignment: .leading, spacing: 12) {
            Hairline()
            if typeSize.isAccessibilitySize {
                VStack(alignment: .leading, spacing: 14) {
                    ForEach(cells, id: \.0) { Readout(label: $0.0, value: $0.1, unit: $0.2) }
                }
            } else {
                HStack(alignment: .top, spacing: 16) {
                    ForEach(cells, id: \.0) { Readout(label: $0.0, value: $0.1, unit: $0.2) }
                        .frame(maxWidth: .infinity, alignment: .leading)
                }
            }
            Text(footnote).font(Voice.dataSmall).foregroundStyle(Tone.inkSoft)
        }
        .motion(.settle, value: model.current)
    }

    private var footnote: String {
        let backend = model.host.backend.isEmpty ? "Neural Engine" : "\(model.host.backend), Neural Engine"
        return "On this iPhone \u{00B7} \(backend)"
    }
}

// MARK: Controls

private struct Controls: View {
    @Environment(AppModel.self) private var model
    @Binding var showImporter: Bool

    var body: some View {
        let idle = model.mode == .idle
        let recording = model.mode == .recording
        GlassEffectContainer(spacing: 12) {
            VStack(spacing: 12) {
                PillButton(title: recording ? "Stop" : "Dictate",
                           systemImage: recording ? "stop.fill" : "mic.fill",
                           emphasis: recording ? .stop : .primary, fill: true) { model.toggleRecord() }
                    .disabled(!model.ready || model.mode == .working)
                    .opacity(model.ready && model.mode != .working ? 1 : 0.45)
                    .accessibilityHint(recording ? "Stops recording and finishes the transcript" : "Starts live dictation")
                ViewThatFits(in: .horizontal) {
                    HStack(spacing: 12) { importButton; samplesButton }
                    VStack(spacing: 12) { importButton; samplesButton }
                }
                .disabled(!idle || !model.ready)
                .opacity(idle && model.ready ? 1 : 0.45)
            }
        }
    }

    private var importButton: some View {
        PillButton(title: "Import file", systemImage: "doc.badge.plus", fill: true) { showImporter = true }
    }
    private var samplesButton: some View {
        PillButton(title: "Samples", systemImage: "play.circle", fill: true) { model.sheet = .samples }
    }
}
