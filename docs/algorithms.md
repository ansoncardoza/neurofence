# Algorithms

Each entry: purpose, input, output, assumptions, limitations, test
strategy. Every algorithm here was chosen for being simple, established,
and explainable over anything that would merely look more sophisticated
(project rule 4.2).

## SHA-256 / SHA-512 manifest hashing

- **Purpose**: deterministic file-integrity baseline.
- **Input**: a file path.
- **Output**: (sha256_hex, sha512_hex), streamed in 4 MiB chunks.
- **Assumptions**: none beyond the file being readable.
- **Limitations**: detects *any* byte-level change; carries no
  information about *what* changed or whether it's malicious.
- **Tests**: determinism, single-byte-change sensitivity, empty-file and
  large-file (>chunk size) streaming. `tests/acquisition/test_hashing.py`.

## Robust statistics (MAD, robust z-score)

- **Purpose**: outlier-resistant central tendency/spread, so a handful of
  poisoned values don't drag the "normal" reference point along with them
  the way a mean/std would.
- **Input**: a numeric array.
- **Output**: median, MAD, and `0.6745 * (x - median) / MAD`.
- **Assumptions**: approximately unimodal data for the z-score
  interpretation to be meaningful.
- **Limitations**: MAD computed from very few samples (<~8) is itself a
  noisy estimate -- empirically confirmed to produce spurious "elevated"
  scores between two identically-distributed groups at n=6
  (`neurofence.activation.trigger_analysis`'s `MIN_BASELINE_SAMPLES`).
  Falls back to std, then to an explicit zero-signal, when MAD is 0 --
  never divides by zero.
- **Tests**: `tests/weight_forensics/test_robust.py`.

## Per-tensor descriptive statistics

- **Purpose**: compact, numerically-safe summary of a weight or
  activation tensor.
- **Input**: a tensor (any shape/dtype).
- **Output**: mean, std, variance, median, MAD, min/max, percentiles,
  skewness, kurtosis, L1/L2 norm, sparsity -- each `None` with a note
  instead of `inf`/`NaN` on overflow.
- **Assumptions**: none; explicitly handles empty, all-NaN/Inf, constant,
  and extreme-magnitude tensors.
- **Limitations**: skewness/kurtosis are high-variance estimators for
  small tensors (few dozen elements).
- **Tests**: `tests/weight_forensics/test_statistics.py`, including a
  regression test (with `RuntimeWarning` promoted to an error) for the
  overflow bug found during Milestone 2 QA.

## Spectral analysis (SVD)

- **Purpose**: detect structural weight modifications (e.g. a low-rank
  additive patch) via singular-value-derived signals.
- **Input**: a 2D weight matrix.
- **Output**: spectral norm, effective rank (entropy-based), condition
  number, spectral entropy, top-k singular values.
- **Assumptions**: real-valued, finite matrix.
- **Limitations**: full SVD above a configurable element-count threshold
  falls back to truncated top-k SVD (ARPACK via
  `scipy.sparse.linalg.svds`), which cannot compute condition number
  (needs the smallest singular value) and only approximates effective
  rank from the top-k values -- always disclosed via
  `condition_number_note` / `computation_status`.
- **Tests**: `tests/weight_forensics/test_spectral.py`, including a
  ground-truth low-rank-perturbation-increases-spectral-norm check.

## Layer anomaly detection (Isolation Forest + LOF + Mahalanobis consensus)

- **Purpose**: flag which layer's weight/activation profile looks unusual
  relative to the model's other layers, using three independent,
  complementary signals so no single detector's opinion decides alone.
- **Input**: `dict[layer_name, feature_vector]`.
- **Output**: per-layer `is_outlier` (>= 2 of 3 methods agree),
  normalized anomaly score, per-method votes.
- **Assumptions**: enough samples (`>= 8` and `> feature_count`) for the
  ML methods to be meaningful.
- **Limitations**: below that threshold, falls back to per-feature
  MAD-based robust z-score (documented, not silent).
  Contamination-based methods (Isolation Forest, LOF) have an inherent
  false-positive floor -- roughly `contamination` fraction of samples get
  flagged "by construction" -- disclosed in `docs/limitations.md` and
  measured directly in `docs/evaluation.md`.
- **Tests**: `tests/weight_forensics/test_anomaly.py` (obvious-outlier
  detection, all-identical-no-false-positives, deterministic given seed).

## PCA / DBSCAN (activation structure)

- **Purpose**: dimensionality reduction and cluster-separation checks
  across multiple prompts' activation feature vectors.
- **Input**: an (n_samples, n_features) matrix.
- **Output**: PCA coordinates + explained variance; DBSCAN cluster labels
  (-1 = noise), n_clusters, noise_count.
- **Assumptions**: `>= 2` samples for PCA, `>= 3` for DBSCAN (documented
  minimums).
- **Limitations**: DBSCAN's `eps` is auto-estimated via a 90th-percentile
  k-distance heuristic -- chosen specifically because the naive
  "median of all pairwise distances" heuristic was empirically shown
  (Milestone 7 QA) to be dominated by inter-cluster distance and merge a
  minority cluster into a majority one, exactly the failure mode this
  function exists to avoid. PCA on constant (zero-variance) input reports
  `0.0` explained variance explicitly rather than propagating a `0/0`
  NaN.
- **Tests**: `tests/activation/test_reduction.py`,
  `tests/activation/test_clustering.py`.

## KL / Jensen-Shannon divergence

- **Purpose**: compare output probability distributions at the logit
  level (e.g. baseline vs. candidate-trigger next-token distribution).
- **Input**: two same-length, same-vocabulary probability vectors.
- **Output**: KL(p‖q) in nats (Laplace-smoothed, always finite);
  Jensen-Shannon divergence (symmetric, bounded by ln 2, well-defined
  without smoothing but smoothed for numerical consistency with KL).
- **Assumptions**: both vectors index the same vocabulary.
- **Limitations**: raises `ValueError` on shape mismatch, negative
  values, NaN/Inf, or all-zero input -- deliberately, since a malformed
  probability vector must never silently produce a misleading number.
- **Tests**: `tests/behavioral/test_distribution.py`.

## TF-IDF cosine similarity (default semantic backend)

- **Purpose**: reproducible, offline, no-download semantic comparison of
  two text outputs.
- **Input**: two strings.
- **Output**: similarity in [0, 1].
- **Assumptions**: none.
- **Limitations**: weaker than real sentence embeddings at catching
  paraphrases with low vocabulary overlap. An optional
  `SentenceTransformerSimilarityBackend` (real embeddings, requires a
  model download) is available as an explicit opt-in.
- **Tests**: `tests/behavioral/test_embeddings.py`.

## Evidence fusion (Anomaly Score / Threat Confidence)

- **Purpose**: combine independent detector sub-scores into two
  deliberately different numbers -- see `docs/detection-methodology.md`
  for the full rationale and formula.
- **Tests**: `tests/fusion/test_combine.py` (the isolated-vs-corroborated
  distinction is the core property tested).
