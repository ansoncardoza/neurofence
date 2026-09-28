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
- Milestones 3–8 (behavioral baseline, adversarial fuzzing, activation
  forensics, evidence fusion, synthetic attack lab, reporting) are not yet
  implemented. Milestone 3 (behavioral analysis) is the first to require
  actually running a model, so it will pull in the `ml` extra
  (PyTorch/transformers).

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
neurofence --help
```

Today `scan` performs secure acquisition (manifest + metadata) and weight
forensics (statistics, spectral analysis, layer anomaly detection) by
default; passing `--reference` additionally runs differential weight
analysis against a trusted baseline. Later milestones extend the same
command with `--prompts`, `--fuzz`, and `--activations` flags as those
subsystems land.

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
