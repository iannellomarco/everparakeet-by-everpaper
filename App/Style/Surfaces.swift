import SwiftUI

// MARK: Canvas

/// The cream desk every screen sits on, with a very faint static grain. The grain never moves.
struct PaperCanvas<Content: View>: View {
    var grain = true
    @ViewBuilder var content: Content

    var body: some View {
        ZStack {
            Tone.canvas.ignoresSafeArea()
            if grain { Grain().ignoresSafeArea().allowsHitTesting(false) }
            content
        }
    }
}

private struct Grain: View {
    var body: some View {
        Canvas { context, size in
            var rng = SeededUnit(seed: 0x5EED0F9A9E5)
            let count = min(1400, max(60, Int(size.width * size.height / 900)))
            for _ in 0..<count {
                let x = rng.next() * size.width, y = rng.next() * size.height
                let r = 0.4 + rng.next() * 0.5
                context.opacity = 0.014 + rng.next() * 0.012
                context.fill(Path(ellipseIn: CGRect(x: x, y: y, width: r, height: r)), with: .color(Tone.ink))
            }
        }
        .accessibilityHidden(true)
    }
}

/// Tiny deterministic generator (SplitMix64) so the grain is identical on every render.
private struct SeededUnit {
    private var state: UInt64
    init(seed: UInt64) { state = seed }
    mutating func next() -> CGFloat {
        state &+= 0x9E37_79B9_7F4A_7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
        z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
        return CGFloat(Double((z ^ (z >> 31)) >> 11) / 9_007_199_254_740_992.0)
    }
}

// MARK: Page

/// The opaque page that holds text. Text is never put on glass.
struct PageCard<Content: View>: View {
    var corner: CGFloat = 24
    @ViewBuilder var content: Content

    var body: some View {
        content
            .background(Tone.page, in: .rect(cornerRadius: corner))
            .overlay(RoundedRectangle(cornerRadius: corner).stroke(Tone.hairline, lineWidth: 1))
    }
}

struct Hairline: View {
    var body: some View { Rectangle().fill(Tone.hairline).frame(height: 1) }
}

// MARK: Glass

extension View {
    /// Liquid Glass for tools (buttons, chips), never for text you read. Under Reduce Transparency it becomes an
    /// opaque warm surface with a hairline edge, same shape and hierarchy.
    func toolGlass(tint: Color? = nil, interactive: Bool = false, in shape: some Shape) -> some View {
        modifier(ToolGlass(tint: tint, interactive: interactive, shape: shape))
    }
}

private struct ToolGlass<S: Shape>: ViewModifier {
    @Environment(\.accessibilityReduceTransparency) private var reduceTransparency
    let tint: Color?
    let interactive: Bool
    let shape: S

    func body(content: Content) -> some View {
        if reduceTransparency {
            content
                .background(shape.fill(tint ?? Tone.pageFallback))
                .overlay(shape.stroke(Tone.hairline, lineWidth: 1))
        } else {
            content.glassEffect(glass, in: shape)
        }
    }

    private var glass: Glass {
        var g: Glass = .regular
        if let tint { g = g.tint(tint) }
        if interactive { g = g.interactive() }
        return g
    }
}

// MARK: Buttons

/// A pill button. The glass rides the button itself, not its label, so touches reach the button on a real device.
struct PillButton: View {
    enum Emphasis {
        case primary   // glass with a terracotta wash and terracotta text: the one main action on a screen
        case quiet     // plain glass, ink text
        case stop      // filled deep terracotta: ends a recording
    }

    let title: String
    var systemImage: String?
    var emphasis: Emphasis = .quiet
    var fill = false
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            HStack(spacing: 8) {
                if let systemImage { Image(systemName: systemImage).font(.body.weight(.semibold)) }
                Text(title).font(Voice.control).multilineTextAlignment(.center)
            }
            .foregroundStyle(foreground)
            .padding(.horizontal, 20)
            .padding(.vertical, 14)
            .frame(maxWidth: fill ? .infinity : nil)
            .contentShape(.capsule)
        }
        .buttonStyle(.plain)
        .modifier(PillBackground(emphasis: emphasis))
    }

    private var foreground: Color {
        switch emphasis {
        case .primary: Tone.accent
        case .quiet: Tone.ink
        case .stop: Tone.stageInk
        }
    }
}

private struct PillBackground: ViewModifier {
    let emphasis: PillButton.Emphasis
    func body(content: Content) -> some View {
        switch emphasis {
        case .primary: content.toolGlass(tint: Tone.accentWash, interactive: true, in: .capsule)
        case .quiet: content.toolGlass(interactive: true, in: .capsule)
        case .stop: content.background(Capsule().fill(Tone.accentDeep))
        }
    }
}

/// A round glass icon button (History, Copy, Share).
struct RoundGlassButton: View {
    @ScaledMetric(relativeTo: .body) private var diameter = 44
    let systemImage: String
    let label: String
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            Image(systemName: systemImage)
                .font(.body.weight(.medium))
                .foregroundStyle(Tone.ink)
                .frame(width: min(diameter, 56), height: min(diameter, 56))
                .contentShape(.circle)
        }
        .buttonStyle(.plain)
        .toolGlass(interactive: true, in: .circle)
        .accessibilityLabel(label)
    }
}

// MARK: Motion

/// Motion, by role. Each role returns nil (an instant change) under Reduce Motion.
enum Motion {
    case settle        // new text rises a little and settles
    case level         // the recording level
    case dash          // indeterminate work: a short dash along a hairline

    func animation(reduceMotion: Bool) -> Animation? {
        guard !reduceMotion else { return nil }
        switch self {
        case .settle: return .easeOut(duration: 0.24)
        case .level: return .easeOut(duration: 0.12)
        case .dash: return .linear(duration: 1.4).repeatForever(autoreverses: false)
        }
    }
}

private struct MotionModifier<V: Equatable>: ViewModifier {
    @Environment(\.accessibilityReduceMotion) private var reduceMotion
    let role: Motion
    let value: V
    func body(content: Content) -> some View {
        content.animation(role.animation(reduceMotion: reduceMotion), value: value)
    }
}

extension View {
    func motion(_ role: Motion, value: some Equatable) -> some View {
        modifier(MotionModifier(role: role, value: value))
    }
}
