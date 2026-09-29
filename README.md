# NeuroFence

An AI model forensic and security assessment platform. NeuroFence
inspects local ML/LLM model artifacts for integrity problems, weight
anomalies, behavioral irregularities, and trigger-conditioned behavior,
and fuses those independent signals into an explainable risk assessment
with a full JSON/PDF report.

**NeuroFence never claims a statistical anomaly proves malicious intent.**
It reports *evidence*, an *Anomaly Score*, and a separate *Threat
Confidence* — always with the reasoning behind each number. See
[docs/threat-model.md](docs/threat-model.md) for what it can and cannot
guarantee, and [docs/limitations.md](docs/limitations.md) for the full,
honest list of gaps.

## Status: all 9 planned milestones implemented

Milestones 0–8 (repository foundation, secure acquisition, weight
forensics, behavioral baseline, adversarial fuzzing, activation forensics,
evidence fusion, synthetic attack lab & evaluation, reporting) are done.
363 tests pass; `ruff check` and `mypy` are clean. Nothing in this
document describes aspirational functionality — everything here has
passing tests you can run yourself, and the detection-performance numbers
below are measured, not invented (see [docs/evaluation.md](docs/evaluation.md)).

Full per-module documentation lives in `docs/`:

- [docs/roles/](docs/roles/README.md) — the project split into three
  presentable engineering roles, each with its own detailed README
  (tech-stack rationale, design decisions, real bugs found and fixed)
- [docs/architecture.md](docs/architecture.md) — pipeline and package layout
- [docs/algorithms.md](docs/algorithms.md) — every algorithm: purpose, input, output, assumptions, limitations, tests
- [docs/detection-methodology.md](docs/detection-methodology.md) — how Anomaly Score / Threat Confidence / Risk are computed
- [docs/threat-model.md](docs/threat-model.md) — what's detected, what's not guaranteed
- [docs/limitations.md](docs/limitations.md) — the full, itemized limitations list
- [docs/testing.md](docs/testing.md) — test strategy
- [docs/evaluation.md](docs/evaluation.md) — measured detector performance

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
set to `true`) — NeuroFence will not execute model-provided code.

## Install

Requires Python 3.11+.

```bash
python -m venv .venv
.venv/Scripts/activate        # Windows
# source .venv/bin/activate   # macOS/Linux
pip install -e ".[dev,report,api,ml]"
```

The `ml` extra (PyTorch, transformers, sentence-transformers) is a heavy
dependency needed only for `--behavioral`/`--trigger`/`--activations`;
plain `scan` (acquisition + weight forensics) works without it.

## Usage

```bash
# Acquisition + weight forensics (default)
neurofence scan ./path/to/model_dir

# Differential analysis against a trusted baseline model
neurofence scan ./path/to/model_dir --reference ./clean_model_dir

# Acquisition only, skip weight forensics
neurofence scan ./path/to/model_dir --no-weights

# Behavioral testing (requires the `ml` extra)
neurofence scan ./path/to/model_dir --behavioral

# Candidate-trigger discovery (repeatable)
neurofence scan ./path/to/model_dir --trigger "ignore all instructions"

# Activation forensics, combined with trigger analysis
neurofence scan ./path/to/model_dir --activations --trigger "ignore all instructions"

# Full scan: JSON report + PDF report
neurofence scan ./path/to/model_dir --reference ./clean_model_dir \
    --behavioral --trigger "ignore all instructions" --activations \
    --output report.json --pdf report.pdf

# Render a PDF from a previously saved JSON report, no detectors re-run
neurofence report ./report.json --pdf report.pdf

# Measure detector performance against synthetic ground truth
neurofence benchmark
neurofence benchmark --n-clean 20 --n-poisoned-per-kind 5 --output bench.json

neurofence --help
```

`scan` always ends with evidence fusion: an Anomaly Score, a Threat
Confidence, and a Risk Classification (LOW/MEDIUM/HIGH/CRITICAL), fusing
whichever detectors were actually enabled for that invocation. With
`--output`/`--pdf`, the full report (findings, recommendations,
limitations, technical appendix — 17 sections per the report spec) is
written; with neither, a summary JSON is printed to stdout.

## Example: a full scan

```bash
neurofence scan ./suspect_model --reference ./known_clean_model \
    --behavioral --trigger "zzz_backdoor_zzz" --output report.json --pdf report.pdf
```

```
JSON report written to report.json
PDF report written to report.pdf
```

`report.json`'s `evidence_fusion` section (abridged):

```json
{
  "status": "evaluated",
  "anomaly_score": 34.45,
  "threat_confidence": 45.43,
  "risk_label": "MEDIUM",
  "num_signals_evaluated": 3,
  "num_signals_elevated": 2,
  "explanation": [
    "3/5 detector(s) evaluated; 2 elevated (score >= 40).",
    "[trigger] score=60.0: 1/1 candidate trigger(s) show consistent behavioral divergence...",
    "[weight_anomaly] score=53.6: Detection method: ml_based (isolation_forest, local_outlier_factor, mahalanobis)..."
  ]
}
```

This is a real, measured output from a synthetic poisoned model built
during development (see `docs/testing.md`) — the weight-anomaly detector
correctly named the poisoned tensor (a tied `lm_head`/`wte` embedding
layer), and trigger discovery correctly flagged the planted phrase.

## Configuration

`neurofence/config/default.yaml` holds every tunable threshold and risk
weight — nothing security-relevant is hardcoded elsewhere in the
codebase.

```python
from neurofence.config import load_config
config = load_config("my_config.yaml")
```

Risk bands (LOW/MEDIUM/HIGH/CRITICAL) and evidence-fusion weights are
project-defined defaults, not universal cybersecurity standards — see
`neurofence/config/schema.py::RiskConfig` and
[docs/detection-methodology.md](docs/detection-methodology.md).

## Development

```bash
pytest -q                    # 363 tests
ruff check neurofence tests
mypy neurofence
```

## QA process

Every milestone in this project followed the same loop: implement, write
tests, run them, switch to an adversarial QA mindset, and — every single
milestone — that process found and fixed real bugs before they shipped,
not after. A sample of what was caught by actually reading measured
output instead of trusting green tests, not by inspection alone:

- Statistics silently overflowing into `inf`/`NaN` on extreme tensor
  values (Milestone 2).
- A prompt longer than a model's context window crashing generation with
  `IndexError` (Milestone 3), and the same bug independently in
  activation capture (Milestone 5).
- DBSCAN's eps heuristic merging two obviously-separate clusters because
  it was dominated by inter-cluster distance (Milestone 5).
- Weight-anomaly feature vectors scaling with tensor *size* rather than
  distribution shape, producing 0% true-negative rate on an all-clean
  synthetic dataset (Milestone 7) — found by looking at real benchmark
  numbers, not by a test merely passing.
- PDF report generation crashing on unescaped model-controlled text
  (an untrusted model's `architecture` string) reaching ReportLab's
  markup parser (Milestone 8).
- Rich's `console.print()` soft-wrapping and corrupting raw JSON output
  when `scan` was run without `--output` (caught by an end-to-end CLI
  test, Milestone 8).

Full narrative detail on each is in the git history's commit messages.

## Future improvements

- Manifest-baseline persistence in the CLI, so `integrity_score` can be
  evaluated from `scan` directly (currently always `not_evaluated` — see
  [docs/limitations.md](docs/limitations.md)).
- A `--config` flag to load a custom `config.yaml` from the CLI (the
  loader already supports it; it's just not wired to `scan` yet).
- Broader default prompt/trigger coverage, and an optional real
  sentence-embedding backend enabled by default when network access is
  available.
- Evaluation against a real, labeled corpus of clean/poisoned production
  models and a genuinely trained (not scripted) backdoor, once such data
  is available.
