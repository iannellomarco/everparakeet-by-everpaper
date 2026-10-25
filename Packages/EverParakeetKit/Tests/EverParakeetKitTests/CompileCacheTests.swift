import XCTest
@testable import EverParakeetKit

final class CompileCacheTests: XCTestCase {
    var dir: URL!
    var cacheDir: URL { dir.appendingPathComponent("compiled") }
    let fm = FileManager.default

    override func setUpWithError() throws { dir = try TestPackage.makeTemporaryDirectory("cache") }
    override func tearDownWithError() throws { try? fm.removeItem(at: dir) }

    /// A fake .mlpackage with the files the stamp hashes.
    func package(weights: String, graph: String = "graph") throws -> URL {
        let url = dir.appendingPathComponent("pkg-\(UUID().uuidString).mlpackage")
        let data = url.appendingPathComponent("Data/com.apple.CoreML")
        try fm.createDirectory(at: data.appendingPathComponent("weights"), withIntermediateDirectories: true)
        try Data(weights.utf8).write(to: data.appendingPathComponent("weights/weight.bin"))
        try Data(graph.utf8).write(to: data.appendingPathComponent("model.mlmodel"))
        return url
    }

    final class Counter: @unchecked Sendable {
        var compiles = 0
        var produced: [URL] = []
        var fail = false
    }

    func compiler(_ counter: Counter, in tmp: URL) -> CompiledModelCache.Compile {
        { _ in
            counter.compiles += 1
            if counter.fail { throw EverParakeetError.invalid("compile failed") }
            let out = tmp.appendingPathComponent("compiled-\(UUID().uuidString).mlmodelc")
            try FileManager.default.createDirectory(at: out, withIntermediateDirectories: true)
            try Data("x".utf8).write(to: out.appendingPathComponent("coremldata.bin"))
            counter.produced.append(out)
            return out
        }
    }

    /// The cache folder's compiled models and temporaries (the weight-hash memo is not one).
    func entries() -> [String] {
        ((try? fm.contentsOfDirectory(atPath: cacheDir.path)) ?? []).filter { $0 != "weights-hashes.json" }.sorted()
    }

    func testCompilesOnceAndMovesTheResult() async throws {
        let counter = Counter()
        let cache = CompiledModelCache(directory: cacheDir)
        let pkg = try package(weights: "w1")
        var compiling = 0
        let first = try await cache.resolve(packageURL: pkg, compile: compiler(counter, in: dir), onCompile: { compiling += 1 })
        XCTAssertEqual(counter.compiles, 1)
        XCTAssertEqual(compiling, 1)
        XCTAssertTrue(first.lastPathComponent.hasPrefix("encoder-") && first.lastPathComponent.hasSuffix(".mlmodelc"))
        XCTAssertTrue(CompiledModelCache.isComplete(first))
        XCTAssertEqual(entries(), [first.lastPathComponent], "no temporary folder is left behind")
        XCTAssertFalse(fm.fileExists(atPath: counter.produced[0].path), "the compile output was moved, not copied")
        let second = try await cache.resolve(packageURL: pkg, compile: compiler(counter, in: dir), onCompile: { compiling += 1 })
        XCTAssertEqual(second, first)
        XCTAssertEqual(counter.compiles, 1, "a hit does not compile")
        XCTAssertEqual(compiling, 1)
    }

    func testStampFollowsTheWeightsNotTheModificationTime() async throws {
        let counter = Counter()
        let cache = CompiledModelCache(directory: cacheDir)
        let a = try package(weights: "same")
        let b = try package(weights: "same")
        try fm.setAttributes([.modificationDate: Date(timeIntervalSince1970: 1000)], ofItemAtPath: b.path)
        let ra = try await cache.resolve(packageURL: a, compile: compiler(counter, in: dir))
        let rb = try await cache.resolve(packageURL: b, compile: compiler(counter, in: dir))
        XCTAssertEqual(ra, rb)
        XCTAssertEqual(counter.compiles, 1)
        let c = try package(weights: "other")
        let rc = try await cache.resolve(packageURL: c, compile: compiler(counter, in: dir))
        XCTAssertNotEqual(rc, ra)
        XCTAssertEqual(counter.compiles, 2)
    }

    func testGivenWeightSHAIsUsedWithoutHashingTheWeights() throws {
        let pkg = try package(weights: "w")
        let given = String(repeating: "ab", count: 32)
        let stamp = try CompiledModelCache.stamp(packageURL: pkg, weightSHA256: given.uppercased())
        XCTAssertTrue(stamp.hasPrefix(String(given.prefix(16))), stamp)
        // the package need not even have a weight file then
        let bare = dir.appendingPathComponent("bare.mlpackage")
        try fm.createDirectory(at: bare, withIntermediateDirectories: true)
        XCTAssertNoThrow(try CompiledModelCache.stamp(packageURL: bare, weightSHA256: given))
        XCTAssertThrowsError(try CompiledModelCache.stamp(packageURL: bare, weightSHA256: nil))
    }

    func testWeightHashIsRememberedPerFileAndRecomputedWhenTheFileChanges() throws {
        let pkg = try package(weights: "w")
        try fm.createDirectory(at: cacheDir, withIntermediateDirectories: true)
        let weights = pkg.appendingPathComponent("Data/com.apple.CoreML/weights/weight.bin")
        let real = try CompiledModelCache.sha256(of: weights)
        XCTAssertEqual(try CompiledModelCache.weightsHash(of: weights, rememberIn: cacheDir), real)
        // A second call reads the memo, not the file: plant a different hash under the same key and it comes back.
        let memo = cacheDir.appendingPathComponent("weights-hashes.json")
        var table = try JSONDecoder().decode([String: String].self, from: Data(contentsOf: memo))
        XCTAssertEqual(table.count, 1)
        let planted = String(repeating: "cd", count: 32)
        table[table.keys.first!] = planted
        try JSONEncoder().encode(table).write(to: memo)
        XCTAssertEqual(try CompiledModelCache.weightsHash(of: weights, rememberIn: cacheDir), planted)
        // The file changes (new bytes, new modification time): hashed again, from the file.
        try Data("a different weight file".utf8).write(to: weights)
        try fm.setAttributes([.modificationDate: Date().addingTimeInterval(60)], ofItemAtPath: weights.path)
        let changed = try CompiledModelCache.weightsHash(of: weights, rememberIn: cacheDir)
        XCTAssertEqual(changed, try CompiledModelCache.sha256(of: weights))
        XCTAssertNotEqual(changed, real)
        // With no folder to remember in, the file is simply hashed.
        XCTAssertEqual(try CompiledModelCache.weightsHash(of: weights, rememberIn: nil), changed)
    }

    func testTwoFirstLoadsInOneProcessShareOneCompile() async throws {
        let counter = Counter()
        let cache = CompiledModelCache(directory: cacheDir)
        let pkg = try package(weights: "w")
        let inner = compiler(counter, in: dir)
        let slow: CompiledModelCache.Compile = { url in
            try await Task.sleep(nanoseconds: 400_000_000)
            return try await inner(url)
        }
        async let one = cache.resolve(packageURL: pkg, compile: slow)
        async let two = cache.resolve(packageURL: pkg, compile: slow)
        let (a, b) = try await (one, two)
        XCTAssertEqual(a, b)
        XCTAssertEqual(counter.compiles, 1, "the second caller joined the first compile")
        XCTAssertTrue(CompiledModelCache.isComplete(a))
    }

    func testNewWeightsPruneOldSiblingsInAnOwnedFolder() async throws {
        let counter = Counter()
        let cache = CompiledModelCache(directory: cacheDir)
        let old = try await cache.resolve(packageURL: try package(weights: "v1"), compile: compiler(counter, in: dir))
        let new = try await cache.resolve(packageURL: try package(weights: "v2"), compile: compiler(counter, in: dir))
        XCTAssertNotEqual(old, new)
        XCTAssertEqual(entries(), [new.lastPathComponent])
    }

    func testSharedDefaultStyleFolderKeepsOtherPackagesCompiledModels() async throws {
        let counter = Counter()
        let cache = CompiledModelCache(directory: cacheDir, pruneSiblings: false)
        let one = try await cache.resolve(packageURL: try package(weights: "std"), compile: compiler(counter, in: dir))
        let two = try await cache.resolve(packageURL: try package(weights: "compact"), compile: compiler(counter, in: dir))
        XCTAssertEqual(Set(entries()), [one.lastPathComponent, two.lastPathComponent])
    }

    func testIncompleteFolderIsRecompiled() async throws {
        let counter = Counter()
        let cache = CompiledModelCache(directory: cacheDir)
        let pkg = try package(weights: "w")
        let done = try await cache.resolve(packageURL: pkg, compile: compiler(counter, in: dir))
        try fm.removeItem(at: done.appendingPathComponent("coremldata.bin"))            // a crash left it half there
        let again = try await cache.resolve(packageURL: pkg, compile: compiler(counter, in: dir))
        XCTAssertEqual(again, done)
        XCTAssertEqual(counter.compiles, 2)
        XCTAssertTrue(CompiledModelCache.isComplete(again))
    }

    func testFailedCompileLeavesNothingAndTheNextTrySucceeds() async throws {
        let counter = Counter()
        counter.fail = true
        let cache = CompiledModelCache(directory: cacheDir)
        let pkg = try package(weights: "w")
        do { _ = try await cache.resolve(packageURL: pkg, compile: compiler(counter, in: dir)); XCTFail("should throw") } catch {}
        XCTAssertEqual(entries(), [])
        counter.fail = false
        _ = try await cache.resolve(packageURL: pkg, compile: compiler(counter, in: dir))
        XCTAssertEqual(entries().count, 1)
    }

    func testStaleTemporaryFoldersArePrunedButFreshOnesStay() async throws {
        let counter = Counter()
        let cache = CompiledModelCache(directory: cacheDir, pruneSiblings: false)
        let pkg = try package(weights: "w")
        let result = try await cache.resolve(packageURL: pkg, compile: compiler(counter, in: dir))
        let stale = cacheDir.appendingPathComponent("\(result.lastPathComponent).tmp-OLD")
        let fresh = cacheDir.appendingPathComponent("\(result.lastPathComponent).tmp-NEW")
        for u in [stale, fresh] { try fm.createDirectory(at: u, withIntermediateDirectories: true) }
        try fm.setAttributes([.modificationDate: Date(timeIntervalSinceNow: -3600)], ofItemAtPath: stale.path)
        _ = try await cache.resolve(packageURL: pkg, compile: compiler(counter, in: dir))
        XCTAssertFalse(fm.fileExists(atPath: stale.path))
        XCTAssertTrue(fm.fileExists(atPath: fresh.path))
    }

    func testSHA256OfAFile() throws {
        let url = dir.appendingPathComponent("abc.bin")
        try Data("abc".utf8).write(to: url)
        XCTAssertEqual(try CompiledModelCache.sha256(of: url), "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad")
    }
}
