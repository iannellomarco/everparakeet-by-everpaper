import SwiftUI
import UIKit

/// Colour tokens. A warm paper palette: cream canvas, an opaque page for anything you read, charcoal ink, hairline
/// beige, and one terracotta accent that is spent only on live state and the single primary action. The dark
/// appearance keeps the same warmth (charcoal, never neutral grey or pure black).
enum Tone {
    // Surfaces
    static let canvas = dynamic(light: 0xF4EFE6, dark: 0x1F1C18)       // the desk
    static let page = dynamic(light: 0xFBF8F1, dark: 0x2A2620)         // the opaque page that holds text
    static let pageFallback = dynamic(light: 0xEFE9DD, dark: 0x322D26) // stands in for glass under Reduce Transparency

    // The recording stage: dark in both appearances, so the waveform reads across a room.
    static let stage = fixed(0x1F1C18)
    static let stageInk = fixed(0xF4EFE6)

    // Ink
    static let ink = dynamic(light: 0x2B2620, dark: 0xEAE3D6)
    static let inkSoft = dynamic(light: 0x6F675B, dark: 0xA59C8C)
    static let hairline = dynamic(light: 0xE3DACA, dark: 0x3F382F)

    // Accent (rationed)
    static let accent = dynamic(light: 0xB75438, dark: 0xC96A4C)
    static let accentDeep = dynamic(light: 0x8E3D27, dark: 0xA84E33)
    static let accentWash = dynamic(light: 0xF4E3DB, dark: 0x3B2A23)

    // MARK: implementation

    private static func parts(_ hex: UInt32) -> (CGFloat, CGFloat, CGFloat) {
        (CGFloat((hex >> 16) & 0xFF) / 255, CGFloat((hex >> 8) & 0xFF) / 255, CGFloat(hex & 0xFF) / 255)
    }

    private static func fixed(_ hex: UInt32) -> Color {
        let c = parts(hex)
        return Color(.sRGB, red: c.0, green: c.1, blue: c.2, opacity: 1)
    }

    private static func dynamic(light: UInt32, dark: UInt32) -> Color {
        let l = parts(light), d = parts(dark)
        return Color(uiColor: UIColor { traits in
            let c = traits.userInterfaceStyle == .dark ? d : l
            return UIColor(red: c.0, green: c.1, blue: c.2, alpha: 1)
        })
    }
}
