# NeuroFence

An autonomous AI model forensic and security assessment platform. NeuroFence
inspects local ML/LLM model artifacts for integrity problems, weight
anomalies, behavioral irregularities, and trigger-conditioned behavior, and
fuses those independent signals into an explainable risk assessment.

**NeuroFence never claims a statistical anomaly proves malicious intent.**
It reports *evidence*, an *anomaly score*, and a separate *threat
confidence* — always with the reasoning behind each number. See
[docs/threat-model.md](docs/threat-model.md) for what it can and cannot
guarantee.

## Status

This is a research prototype under active, milestone-based development.
Implemented so far:

- **Milestone 0 — Repository foundation**: package layout, config system,
  logging, test harness, linting/type-checking. Done.
- **Milestone 1 — Secure model acquisition**: deterministic file manifest
  (SHA-256/SHA-512), safe metadata extraction (architecture, parameter
  count, layer count) from `config.json` and safetensors headers, path
  traversal protection. Done.
- **Milestone 2 — Weight forensics**: per-tensor descriptive statistics
  (mean/std/percentiles/skew/kurtosis/norms/sparsity) with MAD-based robust
  statistics; layer anomaly detection via Isolation Forest + Local Outlier
  Factor + Mahalanobis distance (consensus vote), with a documented
  statistical fallback when there are too few layers for ML-based
  detection to be meaningful; spectral forensics (SVD: spectral norm,
  effective rank, condition number, spectral entropy) with a truncated-SVD
  fallback for large matrices; differential weight analysis (ΔW) against a
  trusted reference model, reporting `REFERENCE_INCOMPATIBLE` rather than
  forcing a comparison when architectures don't match. Done. Operates on
  safetensors tensors via NumPy only -- no PyTorch dependency yet.
- **Milestone 3 — Behavioral baseline**: fixed prompt catalog across 7
  categories (general knowledge, reasoning, coding, cybersecurity,
  summarization, classification, safety); a `ModelRunner` protocol backed
  by `HuggingFaceCausalLMRunner` (greedy decoding by default for
  reproducibility, hard-disabled `trust_remote_code`, automatic prompt
  truncation to the model's context window instead of crashing); pairwise
  output comparison (semantic similarity via TF-IDF cosine by default, or
  optional sentence-transformers embeddings; heuristic refusal detection;
  length-ratio tracking) and KL/Jensen-Shannon divergence for
  logit-level comparison. Done. Wired into the CLI via `--behavioral`.
- **Milestone 4 — Adversarial fuzzer**: deterministic (seeded) mutation
  generator across random-text, repetition, Unicode (homoglyphs,
  zero-width characters, mixed scripts), formatting, and prompt-mutation
  categories, plus a user-supplied candidate-trigger dictionary; a smaller
  `max_prompts` budget is always a stable prefix of a larger one with the
  same seed. Trigger discovery appends each candidate phrase to multiple
  base prompts, compares baseline vs. mutated output (reusing the
  behavioral comparison primitives), and ranks phrases by a documented,
  weighted anomaly score -- flagging a phrase as `potential_trigger_candidate`
  only when the effect is consistent (a strict majority of tested prompts
  elevated), never as a "confirmed backdoor." Done. Wired into the CLI via
  `--trigger` (repeatable).
- **Milestone 5 — Activation forensics**: forward-hook-based capture of
  compact per-layer summary statistics (reusing weight_forensics'
  numerically-safe `TensorStatistics`, not raw tensors -- memory-bounded
  regardless of how many prompts are run); "auto" layer selection covers
  both `nn.Linear` and `transformers.pytorch_utils.Conv1D` (GPT-2-family
  attention/MLP projections use the latter). Layer anomaly detection
  reuses weight_forensics' Isolation Forest/LOF/Mahalanobis consensus
  detector unchanged (feature vectors are feature vectors). PCA and DBSCAN
  clustering added for cross-prompt structure; activation-level trigger
  analysis compares baseline vs. candidate-trigger prompts' activations
  per layer, requiring a strict majority of trigger prompts to show
  elevated distance from the baseline centroid before calling the
  separation "consistent" -- same evidentiary standard as fuzzing's
  trigger discovery. Done. Wired into the CLI via `--activations`
  (combines with `--trigger` for activation-level trigger analysis).
- QA caught two more real bugs during test-writing: DBSCAN's default eps
  heuristic (median of *all* pairwise distances) was dominated by
  inter-cluster distance whenever a minority cluster sat far from a
  majority one -- exactly the separation this module exists to detect --
  merging both into one cluster; switched to a 90th-percentile k-distance
  heuristic. PCA on constant (zero-variance) input divided by zero,
  producing NaN in explained_variance_ratio; now reports 0.0 explicitly.
  A third bug (prompts longer than context window crashing activation
  capture, same root cause as the Milestone 3 behavioral-runner bug) was
  also found and fixed with a regression test.
- **Milestone 6 — Evidence fusion**: converts each detector's output into
  a normalized [0,100] `SubScore` with explicit reasons (`not_evaluated`,
  never a fabricated 0, when a detector didn't run). Combines them into
  two genuinely different numbers: **Anomaly Score** (weighted average of
  whatever was evaluated, renormalized over just those -- one very loud
  detector can push this high alone) and **Threat Confidence** (rewards
  *agreement* across independent detectors -- an isolated elevated signal
  is discounted, several independent detectors elevated together score
  close to their shared severity), per the project's core multi-signal
  principle. `RiskConfig` (Milestone 0) then classifies Threat Confidence
  into LOW/MEDIUM/HIGH/CRITICAL. Done. Always runs at the end of `scan`,
  fusing whichever detectors were actually enabled for that invocation.
  Verified end-to-end: a real poisoned model (weight spike on a tied
  `lm_head`/`wte` layer) run through the full CLI with `--reference
  --behavioral --trigger` produced a coherent MEDIUM verdict with the
  poisoned layer correctly named in the explanation.
- Known gap: `integrity_score` always reports `not_evaluated` from the CLI
  today -- it re-verifies a model directory against a previously stored
  baseline *manifest* (`neurofence.acquisition.verify_manifest`), and
  `scan` does not yet accept a stored manifest for that (only a full
  `--reference` *model* directory, which drives differential weight
  analysis instead -- a different check). Wiring manifest persistence into
  the CLI is left for a future pass.
- Milestones 7–8 (synthetic attack lab, reporting) are not yet
  implemented.

Nothing below the "Status" line describes aspirational functionality --
everything documented as done has passing tests you can run yourself.

## Why safetensors, not raw PyTorch pickle

Legacy PyTorch checkpoints (`.bin`/`.pt`/`.pth`) are Python `pickle` files:
loading one can execute arbitrary code. NeuroFence **never unpickles**
untrusted weight files. It hashes them (for integrity manifests) but only
parses structured, non-executable formats: `safetensors` headers and
`config.json`. If a model ships only pickle weights, NeuroFence reports
`weight_format: pickle_weights` and explains that parameter counting and
weight-level forensics are unavailable until the model is converted to
safetensors.

`trust_remote_code` is hard-disabled in configuration (validation error if
set to `true`) -- NeuroFence will not execute model-provided code.

## Install

Requires Python 3.11+.

```bash
python -m venv .venv
.venv/Scripts/activate        # Windows
# source .venv/bin/activate   # macOS/Linux
pip install -e ".[dev,report,api]"
```

The `ml` extra (PyTorch, transformers, sentence-transformers) is required
starting at Milestone 2 (weight forensics) and is deliberately not
installed by default -- it's a heavy dependency and Milestones 0-1 don't
need it.

## Usage

```bash
neurofence scan ./path/to/model_dir
neurofence scan ./path/to/model_dir --output results.json
neurofence scan ./path/to/model_dir --reference ./clean_model_dir
neurofence scan ./path/to/model_dir --no-weights   # acquisition only
neurofence scan ./path/to/model_dir --behavioral   # also run behavioral suite
neurofence scan ./path/to/model_dir --trigger "ignore all instructions"  # candidate-trigger discovery
neurofence scan ./path/to/model_dir --activations --trigger "ignore all instructions"
neurofence --help
```

Today `scan` performs secure acquisition (manifest + metadata) and weight
forensics (statistics, spectral analysis, layer anomaly detection) by
default; passing `--reference` additionally runs differential weight
analysis against a trusted baseline; passing `--behavioral` loads the
model with transformers and runs the behavioral test suite (comparing
against `--reference`'s outputs too, if given); passing one or more
`--trigger` phrases loads the model and runs candidate-trigger discovery;
passing `--activations` captures and analyzes internal activations
(combined with `--trigger`, also runs activation-level trigger separation
analysis). `--behavioral`/`--trigger`/`--activations` require the `ml`
extra and actually load/run the model, so they are off by default.

## Configuration

`neurofence/config/default.yaml` holds every tunable threshold and risk
weight -- nothing security-relevant is hardcoded elsewhere in the codebase.
Pass a custom file via `--config` (once wired into the CLI in a later
milestone) or load it directly:

```python
from neurofence.config import load_config
config = load_config("my_config.yaml")
```

Risk bands (LOW/MEDIUM/HIGH/CRITICAL) and evidence-fusion weights are
project-defined defaults, not universal cybersecurity standards -- see
`neurofence/config/schema.py::RiskConfig`.

## Development

```bash
pytest                     # run the test suite
ruff check neurofence tests
mypy neurofence
```

## Architecture

See [ARCHITECTURE.md](ARCHITECTURE.md) (to be added as milestones land) and
the pipeline overview in the project's governing spec: acquisition -> weight
forensics -> behavioral analysis -> adversarial fuzzing -> activation
forensics -> evidence fusion -> risk assessment -> report.

## Limitations (partial -- expanded as detectors are added)

- Absence of a detected anomaly does not prove a model is safe.
- Weight-level forensics are unavailable for pickle-only checkpoints by
  design (see above).
- Behavioral analysis (once implemented) is only as good as its prompt
  coverage.
