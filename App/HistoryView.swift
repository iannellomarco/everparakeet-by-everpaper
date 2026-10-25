import SwiftUI

/// The last transcripts, kept in a JSON file on this iPhone.
struct HistoryView: View {
    @Environment(AppModel.self) private var model
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        NavigationStack {
            PaperCanvas {
                if model.history.items.isEmpty {
                    VStack(alignment: .leading, spacing: 8) {
                        Text("History").font(Voice.masthead).foregroundStyle(Tone.ink)
                        Text("Your last 25 transcripts appear here. They never leave this iPhone.")
                            .font(Voice.proseSmall).foregroundStyle(Tone.inkSoft)
                        Spacer()
                    }
                    .padding(24)
                    .frame(maxWidth: .infinity, alignment: .leading)
                } else {
                    List {
                        Text("History").font(Voice.masthead).foregroundStyle(Tone.ink)
                            .listRowBackground(Color.clear)
                            .listRowSeparator(.hidden)
                        ForEach(model.history.items) { t in
                            Button {
                                model.current = t
                                dismiss()
                            } label: {
                                VStack(alignment: .leading, spacing: 6) {
                                    HStack(alignment: .firstTextBaseline) {
                                        Text(t.source).kicker().foregroundStyle(Tone.inkSoft).lineLimit(1)
                                        Spacer(minLength: 8)
                                        Text(t.date, format: .dateTime.month().day().hour().minute())
                                            .font(Voice.dataSmall).foregroundStyle(Tone.inkSoft)
                                    }
                                    Text(t.text).font(Voice.proseSmall).foregroundStyle(Tone.ink).lineLimit(3)
                                    Text(String(format: "%.1f s \u{00B7} %.0f ms \u{00B7} %.0fx \u{00B7} %@",
                                                t.audioSeconds, t.computeMs, t.realtimeFactor, t.backend))
                                        .font(Voice.dataSmall).foregroundStyle(Tone.inkSoft)
                                }
                                .padding(.vertical, 6)
                                .frame(maxWidth: .infinity, alignment: .leading)
                                .contentShape(.rect)
                            }
                            .buttonStyle(.plain)
                            .listRowBackground(Color.clear)
                            .listRowSeparatorTint(Tone.hairline)
                        }
                        .onDelete { model.history.remove(at: $0) }
                    }
                    .listStyle(.plain)
                    .scrollContentBackground(.hidden)
                }
            }
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .topBarLeading) {
                    if !model.history.items.isEmpty {
                        Button("Clear", role: .destructive) { model.history.clear() }.tint(Tone.accent)
                    }
                }
                ToolbarItem(placement: .topBarTrailing) { Button("Done") { dismiss() }.tint(Tone.accent) }
            }
        }
        .presentationBackground(Tone.canvas)
    }
}
