import Foundation

/// Per-lane encoder input buffers (`feats`, `m1`, `m2`, `m3` for every window). A backend hands one out per lane, so two
/// lanes never write into the same buffers.
public protocol EncoderInputs: AnyObject {}

/// A loaded encoder (all windows). The model and its weights are shared by every lane; the per-call input buffers are not.
public protocol EncoderBackend: AnyObject, Sendable {
    /// Windows (output frames: 64 = 5 s, 128 = 10 s, 192 = 15 s, 384 = 30 s) this backend can run.
    var windows: [Int] { get }
    /// Fresh input buffers for one lane.
    func makeInputs() throws -> EncoderInputs
    /// `mel`: channel-major [nMels, frames] fp32. Returns the encoder output frame-major [window, hidden] and the number
    /// of valid frames.
    func encode(mel: [Float], frames: Int, valid: Int, window: Int, inputs: EncoderInputs) async throws -> (enc: [Float], validFrames: Int)
    /// 1.1.2: the verdict of the post-load speed probe, nil when the backend is not probed (default). Existing
    /// backends need not implement it.
    var health: EncoderHealth? { get }
}

/// Builds the encoder of a package. The Kit ships the Core ML factory; the optional `EverParakeetCoreAI` target ships a
/// Core AI one.
public protocol EncoderFactory: Sendable {
    /// - Parameters:
    ///   - root: the package folder (holds `models/`).
    ///   - windows: the windows to load, in the order they should be loaded (cancellation is checked between them).
    func makeEncoder(root: URL, config: EverParakeetConfig, windows: [Int], progress: LoadProgressHandler?) async throws -> EncoderBackend
}

/// Which encoder window serves an utterance: the first window (smallest first) whose capacity holds all mel frames
/// (`frames <= 8 * window`). Returns nil when the audio exceeds the largest window.
func routeWindow(melFrames: Int, windows: [Int]) -> Int? {
    windows.sorted().first(where: { melFrames <= 8 * $0 })
}

/// `_mask_lengths`: three stride-2 conv stages (kernel 3, pad 1).
public func maskLengths(_ length: Int) -> [Int] {
    var current = length
    var out: [Int] = []
    for _ in 0..<3 {
        current = floorDiv(current + 2 - 3, 2) + 1
        out.append(current)
    }
    return out
}

/// Python floor division.
func floorDiv(_ a: Int, _ b: Int) -> Int {
    let q = a / b
    return (a % b != 0 && (a < 0) != (b < 0)) ? q - 1 : q
}

/// Who gets the shared encoder next. Dictation lanes are `.interactive` and go before `.background` final passes.
public enum LanePriority: Int, Sendable { case interactive = 0, background = 1 }

/// FIFO queues of waiters for an actor-isolated lock, one queue per priority. A waiter can be taken out of line by id,
/// which is how a cancelled caller stops waiting at once instead of when the lock reaches it.
struct LockWaiters {
    typealias Waiter = CheckedContinuation<Void, Error>
    private var queues: [[(id: UInt64, continuation: Waiter)]]
    private var nextID: UInt64 = 0

    init(priorities: Int) { queues = Array(repeating: [], count: priorities) }

    mutating func newID() -> UInt64 { nextID += 1; return nextID }
    mutating func add(_ continuation: Waiter, id: UInt64, priority: Int = 0) { queues[priority].append((id, continuation)) }

    /// The next waiter to hand the lock to (best priority first, FIFO inside one).
    mutating func next() -> Waiter? {
        for p in queues.indices where !queues[p].isEmpty { return queues[p].removeFirst().continuation }
        return nil
    }

    /// Takes waiter `id` out of line; nil when it was already handed the lock.
    mutating func remove(id: UInt64) -> Waiter? {
        for p in queues.indices {
            if let i = queues[p].firstIndex(where: { $0.id == id }) { return queues[p].remove(at: i).continuation }
        }
        return nil
    }
}

/// Serialises calls to the shared encoder model: one prediction at a time, interactive callers first, FIFO inside a priority.
/// A caller that is cancelled while it waits stops waiting immediately (a prediction already running cannot be interrupted).
actor EncoderGate {
    private var busy = false
    private var waiters = LockWaiters(priorities: 2)

    /// Returns holding the lock, or throws `CancellationError` without it.
    func acquire(_ priority: LanePriority) async throws {
        try Task.checkCancellation()
        if !busy { busy = true; return }
        let id = waiters.newID()
        try await withTaskCancellationHandler {
            try await withCheckedThrowingContinuation { (continuation: LockWaiters.Waiter) in
                if Task.isCancelled { continuation.resume(throwing: CancellationError()); return }
                waiters.add(continuation, id: id, priority: priority.rawValue)    // the lock is handed over directly
            }
        } onCancel: {
            Task { await self.cancelWaiter(id) }
        }
    }

    private func cancelWaiter(_ id: UInt64) {
        waiters.remove(id: id)?.resume(throwing: CancellationError())
    }

    func release() {
        if let next = waiters.next() { next.resume() } else { busy = false }
    }

    func run<T>(_ priority: LanePriority, _ body: () async throws -> T) async throws -> T {
        try await acquire(priority)
        defer { release() }
        try Task.checkCancellation()
        return try await body()
    }
}
