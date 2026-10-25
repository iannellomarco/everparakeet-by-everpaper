# EverParakeet Kit

Swift runtime for the EverParakeet speech models: load a package folder, transcribe 16 kHz audio, get words with
times and confidences. Apple silicon only (macOS 15+, iOS 18+), no dependencies besides system frameworks.

This is the hardened Kit (1.1.2). It starts from the Kit published in the Standard 1.1 package and adds what an
app needs to embed it.

## Targets

| Target | What it is |
|---|---|
| `EverParakeetKit` | The library: Core ML encoder, CPU decoder, tokenizer, long-form. Never imports Core AI. |
| `EverParakeetCoreAI` | Optional Core AI encoder (macOS 27, iOS 27), with a speed check and self-repair of its Neural Engine plan (see below). A separate target: an app that only needs Core ML leaves it out and never links the Core AI framework. |
| `everparakeet` | Command line: `transcribe`, `smoke`, batch (`--list`/`--out`). |

## Use

```swift
import EverParakeetKit

let model = try await EverParakeetModel.load(root: packageFolder)       // async, cancellable, reports progress
let lane = try model.makeLane()                                          // one lane per concurrent stream
let result = try await lane.transcribe(pcm16k: samples)
print(result.text)
for word in result.words { print(word.text, Double(word.startFrame) * 0.08, word.confidence) }
```

* **Timed tokens.** `Transcription.tokens` is `[TimedToken { id, frame, durationFrames, confidence }]`; one frame is
  80 ms. `confidence` is the softmax probability of the emitted token over the vocabulary (blank included).
  `model.piece(for:)` gives the SentencePiece text of an id, `model.words(from:)` assembles words with the Greek
  final-sigma rule applied per word, so `text` and `words` agree.
* **Model and lanes.** `EverParakeetModel` is immutable and shared (config, encoder, decoder weights, tokenizer).
  Each `EverParakeetLane` has its own LSTM state, scratch and encoder input buffers, so lanes are independent. Calls to
  the shared encoder model are serialised, `.interactive` lanes first. The mel front end and the decoder run on the
  lane's own `DispatchQueue` (QoS from `LaneOptions`), never on Swift's cooperative pool. Decoder gemv workers:
  `LoadOptions.decoderThreads` / `LaneOptions.decoderThreads` (default 4, 2 on 8 GB or base-chip Macs).
* **Cancellation and progress.** Cancel the `Task`: the lane stops between windows, inside the decoder every 16 joint
  calls, and the moment it is cancelled while queued (behind another call on the same lane, or for the shared
  encoder; a prediction that is already running cannot be interrupted and finishes first, one window at most); `load`
  stops between function loads. `transcribe(...progress:)` reports after every window. `LoadOptions.progress` reports compile and per-function loads.
* **`SampleSource`.** Random-access audio for long recordings (`ArraySampleSource`, `WAVSampleSource` for Float32 / PCM16
  WAV files, which finds the `data` chunk itself and clamps stale crash-recovered lengths).
* **`TokenChooser`.** `lane.setChooser(_:)` hooks a vocabulary-bias policy: it sees the top-k tokens and raw logits of
  every non-blank emission, with the token the decoder would emit passed as `original` (with a blank bias on it need
  not be `topK[0]`), and may replace the token. A lane reads its chooser once when a `transcribe` starts.
* **Long-form** (audio over 30 s, `TranscribeOptions.longForm`): 30 s windows with 8 s overlap merged by the longest
  common word subsequence, blank-logit bias (default 1.25, or `long_audio.blank_bias` from the package) and a
  repetition guard. `LongFormOptions.off` gives the 1.1.x behaviour exactly. Audio up to 30 s is never affected.
* **Compile cache.** `CompiledModelCache(directory:weightSHA256:pruneSiblings:)`: the compiled `.mlmodelc` is moved
  (not copied) into the folder under a temporary name, then renamed, so a crash cannot leave a half-written model, and
  it is keyed on the SHA-256 of `weight.bin` (pass it from your asset table to skip hashing).
* **`main128`** (10 s window): 5 to 10 s clips are routed to it when the package has the function
  (`LoadOptions.useMain128`, on by default). The Core AI asset has it; the Core ML packages of Standard 1.1.x carry
  main64, main192 and main384 only, so nothing changes there. On Core AI the text equals main192's on all 3,188 clips of
  nine lab sets (1,049 of them between 5 and 10 s).

## Core AI: speed check and self-repair (1.1.2)

Core AI specialises `encoder.aimodel` for the Neural Engine the first time it is loaded and caches the result. If the
Neural Engine compiler service cannot be reached at that moment (we saw `NSCocoaErrorDomain` 4097, "connection to
service named com.apple.ANECompilerService", on an iPhone 17 Pro with iOS 27.2), Core AI does not fail: it caches a
plan that is not on the Neural Engine, and every later launch reuses it. On that phone each encoder call took about
490 ms whatever the window size (the Neural Engine takes 20 to 35 ms for the smallest window and 65 to 80 ms for the
largest), and the same build in a fresh container ran at 160 to 178x real time. Core AI offers no way to ask which
compute unit a plan uses, so the Kit measures it:

1. `CoreAIEncoderFactory` specialises explicitly (`AIModel.specialize`), times it, and reports it through
   `LoadOptions.progress` as `LoadProgress.encoder(EncoderStatus)`, with Core AI's own error text when it throws.
2. It then probes the smallest window on silence (one untimed run, three timed, the fastest counts). Slower than
   `EncoderHealthPolicy.slowThresholdMilliseconds` (150 ms) means "not on the Neural Engine".
3. When the probe fails it releases the slow encoder (Core AI will not delete an entry while a model made from it is
   alive), deletes this model's Core AI cache entries (`AIModelCache.deleteEntries(for:)`), specialises again, probes
   again, once. Healthy: `model.encoderHealth.state == .repaired` and Core AI is used.
   Still slow: `.notOnNeuralEngine` (also `EncoderStatus.notOnNeuralEngine`); the model works, slowly, and the app
   should load the Core ML encoder instead. A repair that did not help is remembered per OS build in
   `Application Support/EverParakeet/coreai-health.json`, so the next launch does not spend minutes on it again; a new
   OS build, or `EncoderHealthPolicy.forceRepair`, tries again.
4. The app controls when a repair may run with `EncoderHealthPolicy.gate` (`RepairGate`). With the gate closed (app in
   the background, live transcription running) nothing is deleted and the result is `.repairDeferred`; load again once
   the gate is open. A repair that finished while the gate was closed is not remembered as a failure.

```swift
import EverParakeetCoreAI
let gate = RepairGate(allowed: true)                       // flip it from your scene phase and recording state
var health = EncoderHealthPolicy(); health.gate = gate
var options = EverParakeetModel.LoadOptions()
options.encoderFactory = CoreAIEncoderFactory(health: health)
options.progress = { if case .encoder(let status) = $0 { print(status) } }
let model = try await EverParakeetModel.load(root: packageFolder, options: options)
if model.encoderHealth?.onNeuralEngine == false { /* load again with the Core ML encoder */ }
```

`CoreAIEncoderFactory()` with no arguments keeps working and now includes the check; `EncoderHealthPolicy.enabled = false`
gives the 1.1.1 behaviour. The check costs about 0.1 s on a healthy Neural Engine plan. Not verified: whether deleting
the cache entries is enough on every device that ends up with a bad plan (the repair is measured, and when it does not
help the Kit says so). `LoadProgress` has a new case, `.encoder(EncoderStatus)`: a `switch` without `default` needs a
branch for it.

## Build and test

```bash
swift build -c release
swift test                          # 112 tests, no model files needed
.build/release/everparakeet smoke --model <package folder> --backend coreml
```

Bit-parity with Kit 1.1.x: with `--long-mode legacy --blank-bias 0` the transcripts are byte-identical to the 1.1.x
lab binary on the lab test sets: 1,681 of 1,681 utterances on the eight quick sets (FLEURS Italian and English,
VoxPopuli Italian and English, AMI, code-switch, MLS Italian, long-form), Core ML, Standard 1.1 package. Re-checked
after the review fixes (tag `v1.1.1-p0-r1`) with a strict comparison that counts a missing or failed row as a
difference: 685 of 685 byte-identical (FLEURS Greek 200, FLEURS Italian 150, AMI 150, VoxPopuli English 100,
code-switch 80, long-form 5), no missing and no failed rows.
