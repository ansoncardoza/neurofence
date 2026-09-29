# Role 1 — Core Infrastructure, Secure Acquisition & Weight Forensics Engineer

**Owned modules:** `neurofence/config/`, `neurofence/acquisition/`, `neurofence/weight_forensics/`
**Code:** 1,611 lines · **Tests:** 108 (config 11, acquisition 32, weight_forensics 65)
**Milestones:** 0 (repository foundation), 1 (secure model acquisition), 2 (weight forensics)

This role built the ground floor of NeuroFence: the configuration system
every other role reads its thresholds from, the subsystem that safely
takes possession of an untrusted model directory, and the statistical
engine that inspects a model's weights for tampering. Nothing above this
layer can run until this layer exists — acquisition and config were built
first for exactly that reason.

---

## 1. What this role is responsible for, in plain terms

Before NeuroFence can say anything about whether a model looks suspicious,
it has to answer three much more basic questions safely:

1. **Can I even open this model without it hurting me?** A model
   directory is untrusted input — it could contain a symlink pointing
   outside the folder, a path-traversal filename, or a legacy PyTorch
   `.bin` file that executes arbitrary code the moment you try to load
   it. This role's `acquisition` package answers that question.
2. **What is this model, structurally?** Architecture, parameter count,
   layer count, file manifest with cryptographic hashes — the baseline
   facts every later stage needs. Also `acquisition`.
3. **Do the weights themselves look statistically normal?** Given a
   safetensors tensor, is its distribution consistent with the rest of
   the model, or does it stick out? This role's `weight_forensics`
   package answers that — the actual forensic core of the "weight
   poisoning" half of the project.

And underpinning all three: a single, typed, validated **configuration
system** (`neurofence/config/`) so that every threshold used anywhere in
the codebase — a risk band cutoff, a contamination rate, a sample-size
minimum — lives in one auditable place instead of being a magic number
buried in some function.

---

## 2. Tech stack, and why each piece was chosen

| Tool | Why this and not something else |
|---|---|
| **Python 3.11+** | Whole team standardized on it; type hints (`X \| None` union syntax, `StrEnum`) needed 3.11+ for clean, modern code. |
| **Pydantic v2** | Every data structure in this role (and consumed by the other two roles) is a Pydantic `BaseModel`. Why: free JSON serialization (critical for the Role 3 report), free validation at construction time (a `RiskWeights` object literally cannot be built with a negative weight), and self-documenting schemas. We considered plain `dataclasses` — rejected because they don't validate, and hand-rolled validation would have been reinvented badly across five packages. |
| **PyYAML** | `config/default.yaml` needed a human-editable format. YAML over JSON because YAML supports comments — and every threshold in that file has one explaining *why* that number, not just what it is. |
| **NumPy** | The numerical substrate for every statistic computed on a tensor. Chosen because `safetensors` already returns NumPy arrays natively (`framework="numpy"`), so there's no unnecessary PyTorch dependency for a package that doesn't need to *run* a model, only read its weights. |
| **SciPy** (`scipy.stats`, `scipy.sparse.linalg.svds`) | `scipy.stats.skew`/`kurtosis` for higher-moment statistics; `scipy.sparse.linalg.svds` for truncated SVD on large matrices where a full `numpy.linalg.svd` would be too slow. |
| **scikit-learn** (`IsolationForest`, `LocalOutlierFactor`) | Two independent, well-established unsupervised outlier detectors, deliberately *not* one fancy custom model — the project's own engineering principle was "prefer simple, established algorithms over anything that looks more sophisticated." |
| **safetensors** | The safe weight format. Chosen specifically *instead of* raw PyTorch `pickle` loading — see the security section below. This is the single most consequential technical decision this role made. |
| **hashlib (stdlib)** | SHA-256 *and* SHA-512 for the file manifest — no external dependency needed for cryptographic hashing. |

### Why NOT PyTorch/transformers in this role's packages

This is worth presenting explicitly: `weight_forensics` and `acquisition`
have **zero PyTorch dependency**. Weight tensors are read directly from
safetensors files via NumPy. This was a deliberate architectural choice:
PyTorch + transformers is a multi-hundred-megabyte install, and Milestones
0–2 don't need to *run* a model, only read its weight files. That heavy
dependency is deferred to Role 2's `ml` extra, which is the first stage
that actually performs inference. Keeping this role's install light was a
conscious cost/benefit call, not an oversight.

---

## 3. The security decision that shapes everything: no unpickling

A legacy PyTorch checkpoint (`.bin`, `.pt`, `.pth`) is a Python `pickle`
file. Unpickling one **executes arbitrary code** embedded in the file —
this is a well-known, real attack vector for "malicious model" supply
chain attacks, which is precisely the threat this project exists to
detect. It would be self-defeating to build a model-security scanner that
itself becomes an RCE vector the moment it's pointed at a hostile model.

So `neurofence/acquisition/formats.py` classifies every file NeuroFence
encounters, and `is_safe_to_introspect()` draws a hard line:

```python
def is_safe_to_introspect(fmt: FileFormat) -> bool:
    return fmt in (
        FileFormat.SAFETENSORS,   # header is JSON + raw tensor bytes, no code
        FileFormat.CONFIG_JSON,
        FileFormat.TOKENIZER,
        FileFormat.TEXT,
    )
    # FileFormat.PICKLE_WEIGHTS is deliberately absent
```

Pickle files are **hashed** (for the integrity manifest — you still want
to know if one changed) but **never opened**. If a model ships only
`.bin` weights, `extract_model_metadata()` honestly reports
`weight_format: "pickle_weights"` with a warning explaining that
parameter counting and weight forensics are unavailable — never a
fabricated "clean" result. This mirrors `trust_remote_code`, which is
hard-disabled at the config-schema level (`ModelConfig` raises a
validation error the moment anyone tries to set it `True`) — a security
boundary enforced by the type system itself, not by a convention someone
could forget to follow.

---

## 4. `acquisition/` — what it actually does

- **`security.py`** — `resolve_within_root()` rejects absolute paths and
  `../` traversal, and separately verifies a *symlink's resolved target*
  stays inside the model root (a link name can look innocent while
  pointing outside).
- **`hashing.py`** — streams a file in 4 MiB chunks computing SHA-256 and
  SHA-512 simultaneously, so a multi-gigabyte weight file is never fully
  loaded into memory just to hash it.
- **`manifest.py`** — walks a model directory, hashes every file, and
  produces a deterministic, sorted `ModelManifest`. Also implements
  `verify_manifest()`: re-hash against a prior manifest and report
  discrepancies (`hash_mismatch`, `missing_file`, `extra_file`,
  `size_mismatch`) — reported as *evidence*, never auto-classified as
  proof of poisoning, per the project's core "evidence not verdict" rule.
- **`metadata.py`** — reads `config.json` (architecture, model_type,
  layer count) and safetensors *headers only* (parameter/tensor count) —
  the tensor data itself is memory-mapped, never materialized, so
  counting parameters in a 70B-parameter model doesn't require 140GB of
  RAM.

## 5. `weight_forensics/` — the forensic core

Four independent, composable stages, each independently unit-tested and
independently usable:

1. **`statistics.py`** — per-tensor mean, std, variance, median, MAD,
   percentiles, skewness, kurtosis, L1/L2 norm, sparsity. The engineering
   discipline here: *every* statistic degrades gracefully. Empty tensor →
   `status: "empty"`. All-NaN tensor → `status: "all_non_finite"`.
   Extreme values that would overflow float64 → the specific statistic
   reports `None` with an explanatory note, never a silently-propagating
   `inf`/`NaN` that corrupts every downstream score.
2. **`robust.py`** — Median Absolute Deviation and the robust z-score
   `0.6745 * (x - median) / MAD`, with a documented fallback chain
   (MAD → std → all-zero) so it never divides by zero.
3. **`spectral.py`** — SVD-derived signals (spectral norm, effective
   rank via spectral entropy, condition number) that catch *structural*
   tampering, like a low-rank additive patch, that per-element statistics
   would miss entirely. Falls back to truncated top-k SVD above a
   configurable size threshold — full SVD on a huge matrix would make the
   scanner unusably slow, so this is a deliberate, disclosed
   precision-for-speed tradeoff, never a silent skip.
4. **`anomaly.py`** — the layer anomaly detector: Isolation Forest + Local
   Outlier Factor + Mahalanobis distance, voting by consensus (a layer is
   only flagged if **at least 2 of 3** independent methods agree — the
   project's multi-signal principle, expressed as code, not just policy).
   Below a minimum sample count (8 layers), the ML methods aren't
   statistically meaningful, so it falls back to a documented per-feature
   robust z-score instead — never silently runs an ML model on
   insufficient data and presents the result as equally trustworthy.
5. **`differential.py`** — direct ΔW comparison against a trusted
   `--reference` model. If tensor names or shapes don't line up between
   the two models, it reports `REFERENCE_INCOMPATIBLE` with an
   explanation rather than forcing a meaningless comparison.

---

## 6. A real bug this role found and fixed (good presentation material)

During Milestone 7 (built by Role 3, but the bug lived in this role's
code), a synthetic all-clean benchmark dataset produced **0% true
negatives** — every clean model was flagged as anomalous. Root cause,
traced back to this role's `build_layer_feature_vector()`: the L1/L2 norm
features scaled with tensor **size** (a 2048-element weight matrix
naturally has a far larger L2 norm than a 32-element bias vector, even
when both are drawn from the identical distribution), so any differently
*shaped* layer in a model — extremely common; every real model mixes
weight matrices with bias vectors — got flagged purely for being a
different size, nothing to do with its actual distribution.

**Fix:** replaced raw `l1_norm`/`l2_norm` with per-element-normalized
versions (mean absolute value, RMS), which are scale-invariant regardless
of tensor size. This is exactly the kind of bug that only shows up when
you *measure* detector performance against ground truth instead of
trusting that the code compiles and the unit tests (which happened to use
same-shaped test tensors) pass — a strong argument for why Milestone 7's
attack lab exists at all.

---

## 7. Testing philosophy for this role

108 tests, organized around **numerical safety edge cases** more than
"happy path" correctness — because a security scanner that crashes or
silently corrupts a number on malformed input is worse than one that's
merely slow. Concretely tested: empty tensors, all-NaN tensors, constant
(zero-variance) tensors, single-element tensors, float16 overflow,
extremely large (1e300) and extremely small (1e-300) values, zero
matrices, rank-deficient matrices, path traversal (`../`, absolute paths,
Windows-style `..\\`), corrupted/malformed `config.json`, mixed
safetensors+pickle models, multi-shard models, and duplicate-content
files. Every one of these is a case a real-world malicious or malformed
model could actually present.

---

## 8. How this role's output feeds the rest of the pipeline

- `ModelManifest` and `ModelMetadata` (this role) are the first two
  sections of every report Role 3 generates.
- `WeightForensicsResult.anomaly_detection` (this role) is converted by
  Role 3's `weight_anomaly_score()` into the `weight_anomaly` sub-score
  that feeds evidence fusion.
- `build_layer_feature_vector()` (this role, in `weight_forensics/pipeline.py`)
  is **reused unmodified** by Role 2's `activation/anomaly.py` — the same
  feature-vector-to-anomaly-score pipeline serves both weight tensors and
  activation tensors, because the underlying math doesn't care what the
  numbers came from. That reuse was a deliberate design decision made by
  this role specifically to avoid duplicating the anomaly detection logic
  a second time in Role 2's territory.
- `RiskConfig`/`RiskWeights` (this role's `config` package) is the schema
  Role 3's evidence fusion is built against.

---

## 9. Commands to demo this role's work live

```bash
# Acquisition + weight forensics only, no ML dependency required
neurofence scan ./some_model_dir --no-weights          # acquisition alone
neurofence scan ./some_model_dir                        # + weight forensics
neurofence scan ./suspect_model --reference ./clean_model  # + differential

# Direct library usage to show the layered design
python -c "
from neurofence.acquisition import build_manifest, extract_model_metadata
from neurofence.weight_forensics import analyze_model_weights

manifest = build_manifest('./some_model_dir')
metadata = extract_model_metadata('./some_model_dir')
result = analyze_model_weights('./some_model_dir')
print(result.anomaly_detection.status, len(result.anomaly_detection.results))
"
```
