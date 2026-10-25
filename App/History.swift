import Foundation
import Observation

/// Last transcripts, kept in a JSON file in Application Support. Local only: nothing leaves the phone.
@MainActor @Observable
final class HistoryStore {
    private(set) var items: [Transcript] = []
    private let url: URL
    static let limit = 25

    init() {
        let dir = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        url = dir.appendingPathComponent("history.json")
        if let data = try? Data(contentsOf: url), let decoded = try? JSONDecoder().decode([Transcript].self, from: data) {
            items = decoded
        }
    }

    func add(_ t: Transcript) {
        guard !t.text.isEmpty else { return }
        items.removeAll { $0.id == t.id }
        items.insert(t, at: 0)
        if items.count > Self.limit { items.removeLast(items.count - Self.limit) }
        save()
    }
    func remove(at offsets: IndexSet) { items.remove(atOffsets: offsets); save() }
    func clear() { items = []; save() }
    private func save() { try? JSONEncoder().encode(items).write(to: url, options: .atomic) }
}
