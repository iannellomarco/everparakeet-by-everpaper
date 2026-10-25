import SwiftUI

// MARK: Status

/// A state is a word and an icon, never colour alone: "RECORDING · 0:12".
struct StatusLine: View {
    enum Kind {
        case ready, recording, working, preparing

        var word: String {
            switch self {
            case .ready: "Ready"
            case .recording: "Recording"
            case .working: "Transcribing"
            case .preparing: "Preparing"
            }
        }
        var symbol: String {
            switch self {
            case .ready: "waveform"
            case .recording: "circle.fill"
            case .working: "text.alignleft"
            case .preparing: "cpu"
            }
        }
        /// The accent is spent only on the live state.
        var accented: Bool { self == .recording }
    }

    let kind: Kind
    var detail: String?

    var body: some View {
        HStack(alignment: .firstTextBaseline, spacing: 6) {
            Image(systemName: kind.symbol)
                .font(.caption2.weight(.bold))
                .foregroundStyle(kind.accented ? Tone.accent : Tone.inkSoft)
            Text(kind.word).kicker().foregroundStyle(kind.accented ? Tone.accent : Tone.ink)
            if let detail {
                Text("\u{00B7}").font(Voice.dataSmall).foregroundStyle(Tone.inkSoft)
                Text(detail).font(Voice.dataSmall).foregroundStyle(Tone.inkSoft)
            }
        }
        .accessibilityElement(children: .ignore)
        .accessibilityLabel(detail.map { "\(kind.word), \($0)" } ?? kind.word)
    }
}

// MARK: Waveline

/// Level proof, not decoration: it draws the levels it is given and nothing else. Silence is a flat hairline, with no
/// idle shimmer. Motion happens only because new audio arrived.
struct Waveline: View {
    let levels: [Double]            // 0...1, oldest first
    var active: Color = Tone.stageInk
    var silent: Color = Tone.stageInk.opacity(0.35)

    static let silence = 0.02

    var body: some View {
        Canvas { context, size in
            let barWidth: CGFloat = 2.5, gap: CGFloat = 2.5
            let count = max(Int(size.width / (barWidth + gap)), 1)
            let quiet = levels.allSatisfy { $0 < Self.silence }
            let recent = Array(levels.suffix(count))
            let bars = quiet ? [Double](repeating: 0, count: count)
                             : [Double](repeating: 0, count: count - recent.count) + recent
            let midY = size.height / 2
            for (i, level) in bars.enumerated() {
                let h = max(CGFloat(min(1, max(0, level))) * size.height, 1.5)
                let rect = CGRect(x: CGFloat(i) * (barWidth + gap), y: midY - h / 2, width: barWidth, height: h)
                context.fill(Path(roundedRect: rect, cornerRadius: barWidth / 2), with: .color(level > 0 ? active : silent))
            }
        }
        .accessibilityElement(children: .ignore)
        .accessibilityLabel("Audio level")
        .accessibilityValue(levels.allSatisfy { $0 < Self.silence } ? "Quiet" : "Active")
    }
}

// MARK: Progress

/// A hairline with a short dash that travels along it while work is going on. Under Reduce Motion the dash stays
/// still in the middle. Pass a fraction for determinate progress.
struct HairlineProgress: View {
    var fraction: Double?
    @Environment(\.accessibilityReduceMotion) private var reduceMotion
    @State private var phase = 0.0

    var body: some View {
        GeometryReader { geo in
            ZStack(alignment: .leading) {
                Rectangle().fill(Tone.hairline)
                if let fraction {
                    Rectangle().fill(Tone.accent).frame(width: geo.size.width * min(1, max(0, fraction)))
                } else {
                    let dash = geo.size.width * 0.22
                    let travel = geo.size.width - dash
                    Rectangle().fill(Tone.inkSoft.opacity(0.7))
                        .frame(width: dash)
                        .offset(x: reduceMotion ? travel / 2 : travel * phase)
                }
            }
        }
        .frame(height: 2)
        .clipShape(.capsule)
        .onAppear {
            guard fraction == nil, let animation = Motion.dash.animation(reduceMotion: reduceMotion) else { return }
            withAnimation(animation) { phase = 1 }
        }
        .accessibilityHidden(true)
    }
}

// MARK: Readout

/// One measured number under a small label.
struct Readout: View {
    let label: String
    let value: String
    var unit: String?

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(label).kicker().foregroundStyle(Tone.inkSoft)
            HStack(alignment: .firstTextBaseline, spacing: 4) {
                Text(value).font(Voice.dataLarge).foregroundStyle(Tone.ink)
                    .contentTransition(.numericText())
                if let unit { Text(unit).font(Voice.dataSmall).foregroundStyle(Tone.inkSoft) }
            }
        }
        .accessibilityElement(children: .combine)
    }
}

// MARK: Rows

/// A row on the page: serif title, sans sub-line, hairline between rows, no boxes.
struct ListRow<Trailing: View>: View {
    let title: String
    let detail: String
    var systemImage: String?
    @ViewBuilder var trailing: Trailing

    var body: some View {
        HStack(alignment: .top, spacing: 16) {
            if let systemImage {
                Image(systemName: systemImage)
                    .font(.system(size: 22, weight: .light))
                    .foregroundStyle(Tone.ink)
                    .frame(width: 30)
                    .accessibilityHidden(true)
            }
            VStack(alignment: .leading, spacing: 4) {
                Text(title).font(Voice.title).foregroundStyle(Tone.ink)
                Text(detail).font(Voice.detail).foregroundStyle(Tone.inkSoft)
            }
            Spacer(minLength: 8)
            trailing
        }
        .padding(.vertical, 16)
        .contentShape(.rect)
    }
}

/// A short reassurance line, such as "Nothing leaves this iPhone."
struct Reassurance: View {
    let lines: [String]
    var body: some View {
        VStack(spacing: 4) {
            ForEach(lines, id: \.self) { Text($0) }
        }
        .font(Voice.proseSmall)
        .foregroundStyle(Tone.inkSoft)
        .multilineTextAlignment(.center)
        .frame(maxWidth: .infinity)
    }
}
