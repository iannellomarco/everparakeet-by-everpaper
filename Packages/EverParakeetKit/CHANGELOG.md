# Changelog

## 1.1.2

Core AI "stale fallback plan" fix. Package of Standard 1.1.1 (tag `v1.1.1-p0-r1`) plus:

* `EverParakeetCoreAI`: explicit, timed specialisation (`AIModel.specialize`) reported through `LoadProgress.encoder`
  with Core AI's error text on failure; post-load health probe of the smallest window (best of three timed runs after one
  untimed, limit 150 ms); on a failed probe delete this model's Core AI cache entries
  (`AIModelCache.deleteEntries(for:)`), specialise again and probe again, once.
* The slow encoder is released before its cache entry is deleted: Core AI refuses to delete an entry while a model made
  from it is alive (`failedToPurge("Deletion could not be completed, assets still in use")`, seen on an iPhone 17 Pro). The delete is
  retried for a moment (`purgeAttempts`, `purgeRetryDelaySeconds`); if it never works, no repair is attempted, nothing is
  remembered, and a working (slow) encoder is handed back with `.notOnNeuralEngine`.
* Still slow after the repair: `EverParakeetModel.encoderHealth.state == .notOnNeuralEngine`, so an app can fall back to
  Core ML. The outcome of a repair that did not help is remembered per OS build
  (`Application Support/EverParakeet/coreai-health.json`) so it does not repeat on every launch.
* `RepairGate`: the app says when a repair may run (foreground, not transcribing live). Closed gate: `.repairDeferred`,
  nothing deleted, nothing remembered.
* New public API: `EncoderHealthPolicy`, `RepairGate`, `EncoderStatus`, `EncoderHealth`, `EverParakeetModel.encoderHealth`,
  `EncoderBackend.health` (default nil), `CoreAIEncoderFactory(health:)`, `LoadProgress.encoder`.
* Compatibility: `CoreAIEncoderFactory()` still compiles and now includes the check (`EncoderHealthPolicy.enabled = false`
  restores 1.1.1). `LoadProgress` gained a case: a `switch` without `default` needs a branch for it. Transcription results,
  model files and the Core ML path are unchanged.
* 27 new unit tests with a fake preparer and encoders (112 in all, no model files, macOS).

## 1.1.1

Hardened Kit: model and lanes, timed tokens with confidence, cancellation, progress, `SampleSource`, `TokenChooser`,
long-form, atomic compile cache, Core AI in its own target, Greek final sigma, int8 decoder loader.
