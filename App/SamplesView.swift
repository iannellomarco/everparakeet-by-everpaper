import SwiftUI

/// The bundled test clips: tap one to transcribe it.
struct SamplesView: View {
    @Environment(AppModel.self) private var model
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        NavigationStack {
            PaperCanvas {
                ScrollView {
                    VStack(alignment: .leading, spacing: 20) {
                        VStack(alignment: .leading, spacing: 8) {
                            Text("Samples").font(Voice.masthead).foregroundStyle(Tone.ink)
                            Text("Real recordings from the FLEURS speech corpus (Google, CC-BY-4.0). Tap one to transcribe it here.")
                                .font(Voice.proseSmall).foregroundStyle(Tone.inkSoft)
                        }
                        VStack(spacing: 0) {
                            ForEach(Array(Sample.all.enumerated()), id: \.element.id) { index, sample in
                                if index > 0 { Hairline() }
                                Button {
                                    dismiss()
                                    model.run(sample: sample)
                                } label: {
                                    ListRow(title: sample.title, detail: sample.subtitle, systemImage: "waveform") {
                                        Image(systemName: "play.circle")
                                            .font(.system(size: 22, weight: .light))
                                            .foregroundStyle(Tone.accent)
                                            .accessibilityHidden(true)
                                    }
                                }
                                .buttonStyle(.plain)
                                .accessibilityHint("Transcribes this clip")
                            }
                        }
                        Text("Clips longer than 30 s go through the long-form path.")
                            .font(Voice.detail).foregroundStyle(Tone.inkSoft)
                    }
                    .padding(24)
                }
            }
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .topBarTrailing) {
                    Button("Done") { dismiss() }.tint(Tone.accent)
                }
            }
        }
        .presentationBackground(Tone.canvas)
    }
}
