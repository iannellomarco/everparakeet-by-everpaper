import XCTest
@testable import EverParakeetKit

/// Records what `EverParakeetModel.load` asks of the encoder factory; loads "functions" one by one like the real ones do.
final class RecordingFactory: EncoderFactory, @unchecked Sendable {
    private let lock = NSLock()
    private var requested: [[Int]] = []
    private var loaded: [Int] = []
    var perFunction: (@Sendable (Int) async -> Void)?
    var requestedWindows: [[Int]] { lock.lock(); defer { lock.unlock() }; return requested }
    var loadedWindows: [Int] { lock.lock(); defer { lock.unlock() }; return loaded }

    private func note(request windows: [Int]) { lock.lock(); requested.append(windows); lock.unlock() }
    private func note(loaded w: Int) { lock.lock(); loaded.append(w); lock.unlock() }

    func makeEncoder(root: URL, config: EverParakeetConfig, windows: [Int], progress: LoadProgressHandler?) async throws -> EncoderBackend {
        note(request: windows)
        for (i, w) in windows.enumerated() {
            try Task.checkCancellation()                       // the same check the Core ML and Core AI factories make
            progress?(.loadingFunction(name: "main\(w)", index: i, of: windows.count))
            await perFunction?(w)
            note(loaded: w)
        }
        return FakeEncoder(windows: windows)
    }
}

final class LoadTests: XCTestCase {
    var root: URL!
    override func setUpWithError() throws {
        root = try TestPackage.makeTemporaryDirectory()
        try TestPackage.write(to: root)
    }
    override func tearDownWithError() throws { try? FileManager.default.removeItem(at: root) }

    final class Events: @unchecked Sendable {
        private let lock = NSLock()
        private var items: [LoadProgress] = []
        func add(_ e: LoadProgress) { lock.lock(); items.append(e); lock.unlock() }
        var all: [LoadProgress] { lock.lock(); defer { lock.unlock() }; return items }
    }

    func testLoadReportsProgressAndUsesTheFactory() async throws {
        let factory = RecordingFactory()
        let events = Events()
        var options = EverParakeetModel.LoadOptions()
        options.encoderFactory = factory
        options.progress = { events.add($0) }
        let model = try await EverParakeetModel.load(root: root, options: options)
        XCTAssertEqual(factory.requestedWindows, [[64, 128, 192, 384]])
        XCTAssertEqual(model.windows, [64, 128, 192, 384])
        XCTAssertEqual(events.all.first, .loadingDecoder)
        XCTAssertEqual(events.all.last, .ready)
        XCTAssertEqual(events.all.filter { if case .loadingFunction = $0 { return true } else { return false } }.count, 4)
    }

    func testMain128CanBeSwitchedOff() async throws {
        let factory = RecordingFactory()
        var options = EverParakeetModel.LoadOptions()
        options.encoderFactory = factory
        options.useMain128 = false
        let model = try await EverParakeetModel.load(root: root, options: options)
        XCTAssertEqual(factory.requestedWindows, [[64, 192, 384]])
        XCTAssertEqual(model.windows, [64, 192, 384])
    }

    func testModelsSubfolderAlsoWorks() async throws {
        var options = EverParakeetModel.LoadOptions()
        options.encoderFactory = RecordingFactory()
        let model = try await EverParakeetModel.load(root: root.appendingPathComponent("models"), options: options)
        XCTAssertEqual(model.config.blankID, TestPackage.blank)
    }

    func testCancelBetweenFunctionLoads() async throws {
        let factory = RecordingFactory()
        let firstDone = expectation(description: "first function loaded")
        factory.perFunction = { w in
            if w == 64 { firstDone.fulfill() }
            try? await Task.sleep(nanoseconds: 300_000_000)
        }
        var options = EverParakeetModel.LoadOptions()
        options.encoderFactory = factory
        let task = Task { try await EverParakeetModel.load(root: self.root, options: options) }
        await fulfillment(of: [firstDone], timeout: 10)
        task.cancel()
        do { _ = try await task.value; XCTFail("load should have been cancelled") } catch { XCTAssertTrue(error is CancellationError, "\(error)") }
        XCTAssertEqual(factory.loadedWindows, [64], "the next function was never loaded")
    }

    func testMissingFilesAreReportedNotCrashed() async throws {
        let empty = try TestPackage.makeTemporaryDirectory("empty")
        defer { try? FileManager.default.removeItem(at: empty) }
        var options = EverParakeetModel.LoadOptions()
        options.encoderFactory = RecordingFactory()
        do { _ = try await EverParakeetModel.load(root: empty, options: options); XCTFail("expected an error") } catch {}
        try FileManager.default.removeItem(at: root.appendingPathComponent("models/decoder_joint.safetensors"))
        do { _ = try await EverParakeetModel.load(root: root, options: options); XCTFail("expected an error") } catch {}
    }

    func testCoreMLFactoryReportsAMissingPackage() async throws {
        let factory = CoreMLEncoderFactory(cache: CompiledModelCache(directory: root.appendingPathComponent("c")))
        let config = try EverParakeetConfig(root: root)
        do { _ = try await factory.makeEncoder(root: root, config: config, windows: [64], progress: nil); XCTFail() } catch {
            XCTAssertTrue("\(error)".contains("encoder.mlpackage"))
        }
    }
}
