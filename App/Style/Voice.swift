import SwiftUI

/// Type roles. Three voices, all system fonts (nothing is bundled):
///  - the system serif for what was said and for titles,
///  - SF for controls,
///  - compact monospace for numbers, so they never change width.
/// Every role sits on a Dynamic Type text style. Text wraps into more lines at large sizes, it is not shrunk.
enum Voice {
    static let masthead = Font.system(.largeTitle, design: .serif).weight(.semibold)
    static let title = Font.system(.title2, design: .serif).weight(.medium)
    static let prose = Font.system(.title3, design: .serif)
    static let proseSmall = Font.system(.callout, design: .serif)
    static let control = Font.system(.body).weight(.semibold)
    static let detail = Font.system(.subheadline)
    static let caps = Font.system(.caption).weight(.semibold)
    static let capsTracking: CGFloat = 1.4
    static let data = Font.system(.callout, design: .monospaced)
    static let dataSmall = Font.system(.footnote, design: .monospaced)
    static let dataLarge = Font.system(.title3, design: .monospaced).weight(.medium)
}

extension Text {
    /// Tracked capitals for small labels and state words: "RECORDING", "FIRST LAUNCH ONLY".
    func kicker() -> some View {
        self.font(Voice.caps).tracking(Voice.capsTracking).textCase(.uppercase)
    }
}
