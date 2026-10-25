import CoreML
import CryptoKit
import Foundation

/// Where and how the compiled (`.mlmodelc`) encoder is cached.
///
/// Core ML compiles the `.mlpackage` once into a temporary folder; this cache moves the result next to the model so the
/// compile never runs twice. The steps are atomic: the compile output is *moved* (not copied) to `<name>.tmp-<uuid>` inside
/// the cache directory and then renamed to its final name, so a crash or a full disk can never leave a half-written
/// `.mlmodelc` that a later launch would accept, and the temporary copy (about the size of the model) does not stay behind.
public struct CompiledModelCache: Sendable {
    /// Cache folder. nil = `~/Library/Application Support/EverParakeet/Compiled` (iOS: the app's Application Support).
    /// An app that owns the model folder passes `<package>/compiled/`.
    public var directory: URL?
    /// SHA-256 of `weight.bin` (lowercase hex), e.g. from the app's frozen asset table. When nil the Kit hashes the file
    /// (357 MB: about a second) once, and remembers the result in the cache folder, keyed by the file's size, inode and
    /// modification time, so a later load of the same file does not read it again.
    public var weightSHA256: String?
    /// Delete the other compiled models in the folder (older stamps) after a successful compile or hit. The default is
    /// true when `directory` is set (the folder belongs to one package) and false for the shared default folder.
    public var pruneSiblings: Bool?

    public init(directory: URL? = nil, weightSHA256: String? = nil, pruneSiblings: Bool? = nil) {
        self.directory = directory; self.weightSHA256 = weightSHA256; self.pruneSiblings = pruneSiblings
    }

    public typealias Compile = @Sendable (URL) async throws -> URL

    static let defaultCompile: Compile = { try await MLModel.compileModel(at: $0) }

    static func defaultDirectory() -> URL {
        let fm = FileManager.default
        return (fm.urls(for: .applicationSupportDirectory, in: .userDomainMask).first ?? fm.temporaryDirectory)
            .appendingPathComponent("EverParakeet/Compiled", isDirectory: true)
    }

    /// Stamp of a package: SHA-256 of its weights, plus a short hash of the model graph so a re-exported graph with the
    /// same weights does not reuse an old compile.
    static func stamp(packageURL: URL, weightSHA256: String?, rememberIn base: URL? = nil) throws -> String {
        let data = packageURL.appendingPathComponent("Data/com.apple.CoreML")
        let weights: String
        if let given = weightSHA256 {
            weights = given.lowercased()
        } else {
            weights = try weightsHash(of: data.appendingPathComponent("weights/weight.bin"), rememberIn: base)
        }
        let graph = (try? sha256(of: data.appendingPathComponent("model.mlmodel"))) ?? "0"
        return "\(weights.prefix(16))-\(graph.prefix(8))"
    }

    /// SHA-256 of `url`, remembered in `base/weights-hashes.json` under "size-inode-mtime(ms)" (a replaced file has another
    /// key and is hashed again). With no `base` the file is simply hashed.
    static func weightsHash(of url: URL, rememberIn base: URL?) throws -> String {
        guard let base else { return try sha256(of: url) }
        let attributes = try FileManager.default.attributesOfItem(atPath: url.path)
        let size = (attributes[.size] as? NSNumber)?.int64Value ?? 0
        let inode = (attributes[.systemFileNumber] as? NSNumber)?.uint64Value ?? 0
        let modified = Int64(((attributes[.modificationDate] as? Date)?.timeIntervalSince1970 ?? 0) * 1000)
        let key = "\(size)-\(inode)-\(modified)"
        let file = base.appendingPathComponent("weights-hashes.json")
        var table = (try? JSONDecoder().decode([String: String].self, from: Data(contentsOf: file))) ?? [:]
        if let known = table[key], known.count == 64 { return known }
        let hash = try sha256(of: url)
        if table.count >= 16 { table = [:] }
        table[key] = hash
        if let data = try? JSONEncoder().encode(table) { try? data.write(to: file, options: .atomic) }
        return hash
    }

    static func sha256(of url: URL) throws -> String {
        let handle = try FileHandle(forReadingFrom: url)
        defer { try? handle.close() }
        var hasher = SHA256()
        while let chunk = try handle.read(upToCount: 1 << 22), !chunk.isEmpty { hasher.update(data: chunk) }
        return hasher.finalize().map { String(format: "%02x", $0) }.joined()
    }

    /// A compiled model folder counts as complete only when its main file is present.
    static func isComplete(_ url: URL) -> Bool {
        FileManager.default.fileExists(atPath: url.appendingPathComponent("coremldata.bin").path)
    }

    /// Returns the compiled model for `packageURL`, compiling (atomically) when there is none.
    func resolve(packageURL: URL, compile: @escaping Compile = CompiledModelCache.defaultCompile,
                 onCompile: (() -> Void)? = nil) async throws -> URL {
        let fm = FileManager.default
        let base = directory ?? Self.defaultDirectory()
        try fm.createDirectory(at: base, withIntermediateDirectories: true)
        let name = "encoder-\(try Self.stamp(packageURL: packageURL, weightSHA256: weightSHA256, rememberIn: base)).mlmodelc"
        let destination = base.appendingPathComponent(name, isDirectory: true)
        let prune = pruneSiblings ?? (directory != nil)
        if fm.fileExists(atPath: destination.path) {
            if Self.isComplete(destination) {
                Self.prune(base: base, keep: name, siblings: prune)
                return destination
            }
            try? fm.removeItem(at: destination)         // a folder that is not complete cannot have come from this code path
        }
        // Two first loads of the same package in one process share one compile.
        return try await CompileCoordinator.shared.run(key: destination.path) {
            onCompile?()
            let produced = try await compile(packageURL)
            let temporary = base.appendingPathComponent("\(name).tmp-\(UUID().uuidString)", isDirectory: true)
            do {
                try fm.moveItem(at: produced, to: temporary)
            } catch {
                try? fm.removeItem(at: produced)
                throw error
            }
            do {
                try fm.moveItem(at: temporary, to: destination)
            } catch {
                try? fm.removeItem(at: temporary)
                // another process finished first: its result is as good as ours
                if Self.isComplete(destination) { Self.prune(base: base, keep: name, siblings: prune); return destination }
                throw error
            }
            Self.prune(base: base, keep: name, siblings: prune)
            return destination
        }
    }

    /// Removes leftovers: `.tmp-` folders older than ten minutes always (a move into the cache never takes that long), and,
    /// with `siblings`, every other `encoder-*.mlmodelc`.
    static func prune(base: URL, keep: String, siblings: Bool) {
        let fm = FileManager.default
        guard let names = try? fm.contentsOfDirectory(atPath: base.path) else { return }
        for item in names where item != keep {
            let url = base.appendingPathComponent(item)
            if item.contains(".tmp-") {
                let modified = (try? fm.attributesOfItem(atPath: url.path)[.modificationDate] as? Date) ?? .distantPast
                if Date().timeIntervalSince(modified) > 600 { try? fm.removeItem(at: url) }
            } else if siblings, item.hasPrefix("encoder-"), item.hasSuffix(".mlmodelc") {
                try? fm.removeItem(at: url)
            }
        }
    }
}

/// Joins concurrent first compiles of the same cache entry inside one process (the result is the same either way;
/// this only avoids paying for the compile twice). Other processes are handled by the atomic rename.
actor CompileCoordinator {
    static let shared = CompileCoordinator()
    private var inFlight: [String: Task<URL, Error>] = [:]

    func run(key: String, _ body: @escaping () async throws -> URL) async throws -> URL {
        if let running = inFlight[key] { return try await running.value }
        let task = Task { try await body() }
        inFlight[key] = task
        defer { inFlight[key] = nil }
        return try await task.value
    }
}
