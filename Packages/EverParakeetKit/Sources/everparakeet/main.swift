import EverParakeetKit
#if canImport(CoreAI)
import EverParakeetCoreAI
#endif
import Foundation

// everparakeet transcribe <audio> [options]
// everparakeet smoke [options]
// everparakeet --list L.tsv --out O.jsonl [options]          (batch, the benchmark protocol)
//
// options: --model <package root>  --backend auto|coreai|coreml  --cache-dir <folder>  --threads <n>  --no-route128
//          --long-mode lcs|legacy  --blank-bias <x>  --overlap-s <s>  --loop-repeats <n>  --loop-repeats1 <n>
//          --max-dur <n>  --bias-short  --words  --emit-tokens  --progress

let args = Array(CommandLine.arguments.dropFirst())

func fail(_ message: String, code: Int32 = 1) -> Never {
    FileHandle.standardError.write(Data((message + "\n").utf8))
    exit(code)
}

func usage() -> Never {
    fail("""
    usage:
      everparakeet transcribe <audio file> [options]
      everparakeet smoke [options]
      everparakeet --list <list.tsv> --out <out.jsonl> [options]
    options:
      --model <package root>        the folder holding models/ and tokenizer/ (default: the parent of swift/)
      --backend auto|coreai|coreml  encoder backend (auto: Core AI on macOS 27 and later, Core ML below)
      --cache-dir <folder>          compiled-model cache (default: Application Support/EverParakeet/Compiled)
      --threads <n>                 decoder gemv workers (default: 4, or 2 on small Macs)
      --no-route128                 do not route 5-10 s clips to the 10 s window (default: routed when the package has it)
      --long-mode lcs|legacy        audio over 30 s: overlapped windows merged by common words (default) or 1.1.x cuts
      --blank-bias <x>              long audio: subtracted from the blank logit (default 1.25; 0 = the model as trained)
      --overlap-s <s>               lcs: overlap between windows (default 8)
      --loop-repeats <n> --loop-repeats1 <n>   repetition guard (n-gram / single token repeats; default 5 / 8)
      --max-dur <n>                 cap on the TDT duration jump for long audio (default 4)
      --bias-short                  apply the blank bias to audio up to 30 s as well
      --words                       transcribe: print the words with frames and confidences as JSON lines
      --emit-tokens                 batch: add the timed tokens to every row
      --progress                    print progress to stderr
    """, code: 2)
}

func opt(_ name: String) -> String? {
    guard let i = args.firstIndex(of: name), i + 1 < args.count else { return nil }
    return args[i + 1]
}

func number(_ name: String) -> Double? {
    guard let s = opt(name) else { return nil }
    guard let v = Double(s) else { fail("error: \(name) needs a number", code: 2) }
    return v
}

/// --model if given, otherwise the nearest parent of the executable (or of the working directory) that holds models/config.json.
func modelRoot() -> URL {
    if let m = opt("--model") { return URL(fileURLWithPath: (m as NSString).expandingTildeInPath) }
    let fm = FileManager.default
    var starts: [URL] = []
    if let exe = Bundle.main.executableURL { starts.append(exe.resolvingSymlinksInPath()) }
    starts.append(URL(fileURLWithPath: fm.currentDirectoryPath))
    for start in starts {
        var dir = start.deletingLastPathComponent()
        for _ in 0..<8 {
            if fm.fileExists(atPath: dir.appendingPathComponent("models/config.json").path) { return dir }
            dir.deleteLastPathComponent()
        }
    }
    fail("error: cannot find the model package; pass --model <package root>")
}

/// Readable text for an error (our own errors describe themselves; system errors use their localized text).
func describe(_ error: Error) -> String {
    if error is EverParakeetError { return "\(error)" }
    return (error as NSError).localizedDescription
}

enum Backend { case coreML, coreAI }

func backend() -> Backend {
    switch opt("--backend")?.lowercased() ?? "auto" {
    case "auto", "automatic":
#if canImport(CoreAI)
        if #available(macOS 27.0, *) { return .coreAI }
#endif
        return .coreML
    case "coreai": return .coreAI
    case "coreml": return .coreML
    default: fail("error: --backend must be auto, coreai, or coreml", code: 2)
    }
}

func longFormOptions() -> LongFormOptions {
    var o = LongFormOptions()
    if let m = opt("--long-mode") {
        guard let mode = LongFormOptions.Mode(rawValue: m.lowercased()) else { fail("error: --long-mode must be lcs or legacy", code: 2) }
        o.mode = mode
    }
    if let v = number("--blank-bias") { o.blankBias = Float(v) }
    if let v = number("--overlap-s") { o.overlap = v }
    if let v = number("--loop-repeats") { o.loopRepeats = Int(v) }
    if let v = number("--loop-repeats1") { o.loopRepeats1 = Int(v) }
    if let v = number("--max-dur") { o.durationCap = Int(v) }
    if args.contains("--bias-short") { o.biasShort = true }
    return o
}

func transcribeOptions() -> TranscribeOptions {
    var o = TranscribeOptions()
    o.longForm = longFormOptions()
    return o
}

func progressHandler() -> ProgressHandler? {
    guard args.contains("--progress") else { return nil }
    return { p in
        FileHandle.standardError.write(Data(String(format: "progress %.1f%% (%d windows)\n", p.fraction * 100, p.windows).utf8))
    }
}

func loadModel(_ root: URL, backend selected: Backend) async -> EverParakeetModel {
    var options = EverParakeetModel.LoadOptions()
    if let dir = opt("--cache-dir") { options.cache = CompiledModelCache(directory: URL(fileURLWithPath: (dir as NSString).expandingTildeInPath)) }
    if let t = number("--threads") { options.decoderThreads = Int(t) }
    options.useMain128 = !args.contains("--no-route128")
    if args.contains("--progress") {
        options.progress = { FileHandle.standardError.write(Data("load: \($0)\n".utf8)) }
    }
    switch selected {
    case .coreML:
        FileHandle.standardError.write(Data("Preparing the Core ML encoder (the first load compiles it into the cache folder)…\n".utf8))
    case .coreAI:
        FileHandle.standardError.write(Data("Preparing the Core AI encoder (the first load may take a few minutes)…\n".utf8))
#if canImport(CoreAI)
        if #available(macOS 27.0, *) { options.encoderFactory = CoreAIEncoderFactory() }
        else { fail("error: the Core AI backend needs macOS 27 or later") }
#else
        fail("error: Core AI is not available in this SDK")
#endif
    }
    do { return try await EverParakeetModel.load(root: root, options: options) }
    catch { fail("error: cannot load the model from \(root.path): \(describe(error))") }
}

func makeLane(_ model: EverParakeetModel) -> EverParakeetLane {
    do { return try model.makeLane() } catch { fail("error: \(describe(error))") }
}

// MARK: smoke

func words(_ text: String) -> [String] {
    let word = CharacterSet.alphanumerics.union(CharacterSet(charactersIn: "_"))
    let normalised = text.precomposedStringWithCompatibilityMapping.lowercased()
    return normalised.unicodeScalars.split(whereSeparator: { !word.contains($0) }).map { String(String.UnicodeScalarView($0)) }
}

func wer(_ reference: String, _ hypothesis: String) -> Double {
    let ref = words(reference), hyp = words(hypothesis)
    var row = Array(0...hyp.count)
    for (i, r) in ref.enumerated() {
        var next = [i + 1]
        for (j, h) in hyp.enumerated() {
            next.append(min(next[j] + 1, row[j + 1] + 1, row[j] + (r == h ? 0 : 1)))
        }
        row = next
    }
    return Double(row[hyp.count]) / Double(max(1, ref.count))
}

/// Neural Engine footprint (MB) from rusage_info_v6.
func neuralFootprintMB() -> Double {
    var ru = rusage_info_v6()
    let rc = withUnsafeMutablePointer(to: &ru) {
        $0.withMemoryRebound(to: rusage_info_t?.self, capacity: 1) { proc_pid_rusage(getpid(), RUSAGE_INFO_V6, $0) }
    }
    return rc == 0 ? Double(ru.ri_neural_footprint) / 1048576 : -1
}

/// Runs tests/reference_transcripts.json with the same per-clip thresholds as the package smoke test.
func smoke() async -> Never {
    let root = modelRoot()
    let selected = backend()
    let refsURL = root.appendingPathComponent("tests/reference_transcripts.json")
    guard let data = try? Data(contentsOf: refsURL),
          let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
          let clips = json["clips"] as? [[String: Any]]
    else { fail("error: cannot read \(refsURL.path)") }
    let model = await loadModel(root, backend: selected)
    let lane = makeLane(model)
    var ok = true
    for clip in clips {
        guard let file = clip["file"] as? String, let reference = clip["reference"] as? String,
              let maxWer = (clip["max_wer"] as? NSNumber)?.doubleValue else { fail("error: malformed reference_transcripts.json") }
        do {
            let pcm = try EverParakeetAudio.loadAudio(url: root.appendingPathComponent("tests/audio/\(file)"))
            let text = try await lane.transcribe(pcm16k: pcm, options: transcribeOptions()).text
            let score = wer(reference, text)
            print("\(file): WER=\(String(format: "%.3f", score))  \"\(text)\"")
            if score > maxWer { print("  FAIL: WER \(String(format: "%.3f", score)) > \(String(format: "%.3f", maxWer))"); ok = false }
        } catch {
            print("\(file): ERROR \(describe(error))"); ok = false
        }
    }
    if selected == .coreAI {
        let neural = neuralFootprintMB()
        if neural > 0 {
            print("Neural Engine footprint: \(String(format: "%.0f", neural)) MB")
        } else {
            print("  FAIL: no Neural Engine footprint (\(String(format: "%.0f", neural)) MB); the encoder did not run on the Neural Engine")
            ok = false
        }
    } else {
        print("Core ML backend selected; placement is reported separately")
    }
    print(ok ? "PASS: EverParakeet smoke test" : "FAIL: EverParakeet smoke test")
    exit(ok ? 0 : 1)
}

// MARK: batch (benchmark protocol)

/// Accumulated GPU time of this process from the IORegistry; about 0 means the encoder ran on the Neural Engine.
func gpuTimeSeconds() -> Double {
    let p = Process()
    p.executableURL = URL(fileURLWithPath: "/usr/sbin/ioreg")
    p.arguments = ["-r", "-c", "AGXDeviceUserClient", "-w0", "-d1"]
    let pipe = Pipe()
    p.standardOutput = pipe
    try? p.run()
    let data = pipe.fileHandleForReading.readDataToEndOfFile()
    p.waitUntilExit()
    let text = String(decoding: data, as: UTF8.self)
    let pid = ProcessInfo.processInfo.processIdentifier
    var total = 0.0
    for block in text.components(separatedBy: "+-o ") where block.contains("\"pid \(pid),") {
        var rest = Substring(block)
        while let r = rest.range(of: "\"accumulatedGPUTime\"=") {
            rest = rest[r.upperBound...]
            total += Double(rest.prefix { $0.isNumber }) ?? 0
        }
    }
    return total / 1e9
}

func ms(_ a: DispatchTime, _ b: DispatchTime) -> Double { Double(b.uptimeNanoseconds - a.uptimeNanoseconds) / 1e6 }

/// Reads `<id>\t<audio path>` lines, writes one JSON line per utterance and a final `_meta` line.
func batch(list listPath: String, out outPath: String, backend selected: Backend) async -> Never {
    guard let listText = try? String(contentsOfFile: listPath, encoding: .utf8) else { fail("error: cannot read \(listPath)") }
    let rows = listText.split(separator: "\n").compactMap { line -> (String, String)? in
        let p = line.split(separator: "\t", maxSplits: 1).map(String.init)
        return p.count == 2 ? (p[0], p[1]) : nil
    }
    let t0 = DispatchTime.now()
    let model = await loadModel(modelRoot(), backend: selected)
    let lane = makeLane(model)
    let loadMs = ms(t0, DispatchTime.now())
    FileManager.default.createFile(atPath: outPath, contents: nil)
    guard let out = try? FileHandle(forWritingTo: URL(fileURLWithPath: outPath)) else { fail("error: cannot write \(outPath)") }
    func emit(_ obj: [String: Any]) {
        guard var d = try? JSONSerialization.data(withJSONObject: obj, options: [.sortedKeys, .withoutEscapingSlashes]) else { return }
        d.append(10)
        try? out.write(contentsOf: d)
    }
    let options = transcribeOptions()
    let emitTokens = args.contains("--emit-tokens")
    for (id, path) in rows {
        let t = DispatchTime.now()
        var text = ""
        var failure: String? = nil
        var result: Transcription? = nil
        do {
            let pcm = try EverParakeetAudio.loadAudio(url: URL(fileURLWithPath: path))
            result = try await lane.transcribe(pcm16k: pcm, options: options)
            text = result!.text
        } catch {
            failure = describe(error)
            text = ""
            FileHandle.standardError.write(Data("error on \(id): \(failure!)\n".utf8))
        }
        let s = result?.timings ?? StageTimings()
        var row: [String: Any] = ["id": id, "text": text, "ms": ms(t, DispatchTime.now()),
              "mel_ms": s.mel, "enc_ms": s.encoder, "dec_ms": s.decoder, "steps": s.steps, "joints": s.joints]
        if emitTokens, let result {
            row["tokens"] = result.tokens.map { [$0.id, $0.frame, $0.durationFrames, Double($0.confidence)] as [Any] }
        }
        row["windows"] = Dictionary(uniqueKeysWithValues: s.windowUse.map { (String($0.key), $0.value) })
        if let failure { row["error"] = failure }
        emit(row)
    }
    emit(["_meta": ["load_ms": loadMs, "gpu_time_s": gpuTimeSeconds(), "ane_footprint_mb": neuralFootprintMB()]])
    try? out.close()
    exit(0)
}

// MARK: dispatch

if args.first == "transcribe" {
    guard args.count >= 2, !args[1].hasPrefix("--") else { usage() }
    let audioURL = URL(fileURLWithPath: (args[1] as NSString).expandingTildeInPath)
    var isDirectory: ObjCBool = false
    guard FileManager.default.fileExists(atPath: audioURL.path, isDirectory: &isDirectory), !isDirectory.boolValue
    else { fail("error: cannot find the audio file \(args[1])") }
    let root = modelRoot()
    let selected = backend()
    do {
        let pcm = try EverParakeetAudio.loadAudio(url: audioURL)
        guard !pcm.isEmpty else { fail("error: \(args[1]) contains no audio") }
        let model = await loadModel(root, backend: selected)
        let lane = makeLane(model)
        let result = try await lane.transcribe(pcm16k: pcm, options: transcribeOptions(), progress: progressHandler())
        if args.contains("--words") {
            let frame = model.config.secondsPerFrame
            for w in result.words {
                let row: [String: Any] = ["word": w.text, "start": (Double(w.startFrame) * frame * 100).rounded() / 100,
                                          "end": (Double(w.endFrame) * frame * 100).rounded() / 100,
                                          "confidence": (Double(w.confidence) * 1000).rounded() / 1000]
                if let d = try? JSONSerialization.data(withJSONObject: row, options: [.sortedKeys, .withoutEscapingSlashes]) {
                    print(String(decoding: d, as: UTF8.self))
                }
            }
        } else {
            print(result.text)
        }
        exit(0)
    } catch { fail("error: cannot transcribe \(args[1]): \(describe(error))") }
} else if args.first == "smoke" {
    await smoke()
} else if let list = opt("--list"), let out = opt("--out") {
    await batch(list: list, out: out, backend: backend())
} else {
    usage()
}
