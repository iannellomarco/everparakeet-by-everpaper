import SwiftUI

/// Shown until the model is ready. The first launch on a device takes a few minutes (iOS prepares the encoder for the
/// Neural Engine, once), so this screen says what is happening and why, and stays calm while it does.
struct PrepareView: View {
    @Environment(AppModel.self) private var model

    private var host: EngineHost { model.host }
    private var longWait: Bool { host.firstLaunch || host.compiling }

    var body: some View {
        PaperCanvas {
            VStack(spacing: 0) {
                Spacer(minLength: 24)
                ScrollView {
                    VStack(alignment: .leading, spacing: 20) {
                        if case .failed(let why) = host.phase { failed(why) } else { preparing }
                    }
                    .padding(.horizontal, 28)
                    .frame(maxWidth: .infinity, alignment: .leading)
                }
                .scrollBounceBehavior(.basedOnSize)
                Spacer(minLength: 24)
                Reassurance(lines: ["Everything runs on this iPhone.", "No network, no account."])
                    .padding(.horizontal, 28)
                    .padding(.bottom, 20)
            }
        }
        .transition(.opacity)
    }

    @ViewBuilder private var preparing: some View {
        Text(host.repairing ? "Once" : (longWait ? "First launch only" : "Loading")).kicker().foregroundStyle(Tone.inkSoft)
        Text(host.repairing ? "Repairing the model preparation for this iPhone"
                            : (longWait ? "Preparing the Neural Engine" : "Loading the model"))
            .font(Voice.masthead).foregroundStyle(Tone.ink)
        if host.repairing {
            Text("The first preparation of the speech encoder did not use the Neural Engine, so it is being done again. It takes about three minutes on an iPhone 17 Pro and happens once.")
                .font(Voice.proseSmall).foregroundStyle(Tone.inkSoft)
            Text("Keep the app open in front. The phone can stay on the table.")
                .font(Voice.proseSmall).foregroundStyle(Tone.inkSoft)
        } else if longWait {
            Text("iOS is compiling the speech encoder for this iPhone. It takes about three minutes on an iPhone 17 Pro and happens once. After that the model is ready in a fraction of a second.")
                .font(Voice.proseSmall).foregroundStyle(Tone.inkSoft)
            Text("You can leave the app open and put the phone down.")
                .font(Voice.proseSmall).foregroundStyle(Tone.inkSoft)
        }
        VStack(alignment: .leading, spacing: 12) {
            HairlineProgress()
            HStack(alignment: .firstTextBaseline) {
                Text(host.stage).font(Voice.detail).foregroundStyle(Tone.ink)
                Spacer(minLength: 12)
                Text(elapsed(host.elapsed)).font(Voice.data).foregroundStyle(Tone.inkSoft)
            }
        }
        .padding(.top, 8)
        if !host.note.isEmpty {
            Text(host.note).font(Voice.dataSmall).foregroundStyle(Tone.inkSoft)
        }
    }

    @ViewBuilder private func failed(_ why: String) -> some View {
        Text("Could not load").kicker().foregroundStyle(Tone.accent)
        Text("The model did not start").font(Voice.masthead).foregroundStyle(Tone.ink)
        Text(why).font(Voice.dataSmall).foregroundStyle(Tone.inkSoft).textSelection(.enabled)
        PillButton(title: "Try again", systemImage: "arrow.clockwise", emphasis: .primary) {
            Task { await host.prepare() }
        }
        .padding(.top, 8)
    }

    private func elapsed(_ s: Double) -> String { String(format: "%d:%02d", Int(s) / 60, Int(s) % 60) }
}
