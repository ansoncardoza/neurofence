# Limitations

This is the expanded version of the limitations that ship inside every
generated report (`neurofence.reporting.report.LIMITATIONS`) and the
[threat model](threat-model.md)'s "does not guarantee" section. Read both
alongside this document -- none of the three repeats the others in full.

## Detection coverage

- **Weight forensics** operates only on safetensors tensors. Legacy
  pickle-format (`.bin`/`.pt`/`.pth`) weight files are hashed for the
  integrity manifest but never parsed or analyzed -- see
  [threat-model.md](threat-model.md) for why. A model shipped only in
  pickle format gets `weight_format: pickle_weights` and a warning, not a
  clean bill of health.
- **Layer anomaly detection** (Isolation Forest / LOF / Mahalanobis
  consensus) requires a minimum sample count
  (`DEFAULT_MIN_SAMPLES_FOR_ML = 8` layers/tensors) to be statistically
  meaningful; below that it falls back to a cruder MAD-based robust
  z-score, which is more prone to false positives on small, heterogeneous
  layer sets (empirically demonstrated during Milestone 7's attack-lab
  benchmarking -- see `docs/evaluation.md`).
- **Contamination-based unsupervised detectors have an inherent
  false-positive floor.** With `contamination=0.1` (the default), roughly
  one layer out of every ~10 gets flagged "by construction," largely
  independent of whether anything is actually anomalous. This is why
  `neurofence.evaluation.benchmark` reports `layer_localization_rate`
  (did the detector flag the *specific* attacked layer) as the more
  meaningful signal, not the raw binary "was anything flagged" accuracy.
- **Behavioral analysis** covers exactly the prompts it was run with.
  `DEFAULT_PROMPTS` is a fixed 12-prompt, 7-category set -- adequate to
  exercise the pipeline, not a comprehensive safety evaluation suite.
- **Refusal detection** is a small, English-only regex heuristic
  (`neurofence.behavioral.refusal`), not an ML classifier. It will miss
  refusals phrased unusually and in non-English text, and can occasionally
  match non-refusal text that happens to contain a matched phrase.
- **Semantic similarity** defaults to TF-IDF cosine similarity (no
  network dependency, fully reproducible offline). This is weaker than a
  real sentence embedding at catching paraphrases with little vocabulary
  overlap; `SentenceTransformerSimilarityBackend` is available as an
  explicit opt-in when a real embedding model is wanted.
- **Trigger discovery and activation trigger analysis** both require a
  strict majority of tested prompts to show elevated evidence before
  calling a result "consistent" -- a single divergent example is treated
  as weak, not conclusive, evidence. Both explicitly reject a phrase or
  separation being labeled "confirmed backdoor."

## Evidence fusion

- The weighting scheme (`RiskWeights` in `neurofence.config.schema`) and
  the Anomaly Score / Threat Confidence formulas
  (`neurofence.fusion.combine`) are documented, inspectable heuristics --
  not derived from a labeled corpus of real clean/poisoned production
  models (none exists publicly at the scale this would require). They are
  validated for *qualitative* correctness (an isolated signal produces
  lower confidence than the same severity corroborated by multiple
  detectors -- see `tests/fusion/test_combine.py`), not for calibration
  against ground truth.
- `integrity_score` is always `not_evaluated` when driven from the
  `scan` CLI command today, because `scan` does not yet accept a
  previously stored baseline manifest to re-verify against (only a full
  `--reference` *model directory*, which drives differential weight
  analysis -- a related but different check). This is a known, disclosed
  gap, not a fabricated "clean" result.

## Evaluation

- `neurofence.attack_lab`'s synthetic models are small (7-8 tensors,
  16-64-dimensional), randomly initialized, and not trained -- they exist
  to give detectors *known ground truth*, not to represent realistic LLM
  weight distributions at scale. Detection performance on a real,
  production-scale transformer will differ from the measured synthetic
  numbers.
- The trigger-detector benchmark uses a scripted, deterministic backdoor
  simulator (`ScriptedTriggerRunner`), not a genuinely trained backdoored
  language model -- training one is out of scope for a fast, reproducible
  test fixture. It validates the *evidence-combination logic* of
  `discover_trigger_candidates`, not real-world backdoor-detection recall
  against sophisticated, adversarially-trained triggers.

## Platform / environment

- Developed and tested on Windows with a CPU-only PyTorch build. GPU
  execution paths (`device="cuda"`) are supported in the code
  (`neurofence.config.schema.ModelConfig.device`) but not exercised by
  the test suite in this environment.
- Report PDF generation escapes all model-controlled and user-controlled
  text before interpolating it into ReportLab's markup (see
  `neurofence.reporting.pdf_report._esc` and the regression test in
  `tests/reporting/test_pdf_report.py`), but has not been fuzzed
  exhaustively against the full space of possible malformed inputs.
