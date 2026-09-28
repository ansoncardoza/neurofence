"""Spectral forensics for 2D weight matrices.

Computes singular-value-derived signals (spectral norm, effective rank,
condition number, spectral entropy) that can reveal structural weight
modifications -- e.g. a low-rank additive patch shows up as a small number
of anomalously large singular values.

Full SVD on very large matrices is prohibitively slow, so above a
configurable size threshold we fall back to a truncated (top-k) SVD via
ARPACK, which trades away exact effective-rank/condition-number estimates
for tractable runtime. Every result records which algorithm ran and why, so
"skipped" or "approximate" is never silent.
"""

from __future__ import annotations

import numpy as np
from pydantic import BaseModel
from scipy.sparse.linalg import svds

DEFAULT_MAX_FULL_SVD_ELEMENTS = 4_000_000  # ~2000x2000 dense float64 SVD is fast enough
DEFAULT_TRUNCATED_K = 50


class SpectralResult(BaseModel):
    shape: list[int]
    algorithm: str  # "full_svd" | "truncated_svd_topk" | "none"
    computation_status: str  # "computed" | "skipped"
    skip_reason: str | None = None
    parameters: dict[str, int]

    top_singular_values: list[float] = []
    spectral_norm: float | None = None
    effective_rank: float | None = None
    condition_number: float | None = None  # None if undefined/singular or not computed
    condition_number_note: str | None = None
    spectral_entropy: float | None = None


def _spectral_entropy_and_effective_rank(
    singular_values: np.ndarray,
) -> tuple[float, float]:
    """Roy & Vetterli (2007) entropy-based effective rank.

    p_i = sigma_i / sum(sigma); effective_rank = exp(-sum(p_i * log(p_i))).
    Zero singular values contribute 0 to the entropy sum by convention
    (0 * log(0) := 0).
    """
    total = float(np.sum(singular_values))
    if total <= 0:
        return 0.0, 0.0

    p = singular_values / total
    nonzero = p[p > 0]
    entropy = float(-np.sum(nonzero * np.log(nonzero)))
    effective_rank = float(np.exp(entropy))
    return entropy, effective_rank


def spectral_analysis(
    matrix: np.ndarray,
    max_full_svd_elements: int = DEFAULT_MAX_FULL_SVD_ELEMENTS,
    truncated_k: int = DEFAULT_TRUNCATED_K,
) -> SpectralResult:
    arr = np.asarray(matrix)
    shape = list(arr.shape)
    params = {"max_full_svd_elements": max_full_svd_elements, "truncated_k": truncated_k}

    if arr.ndim != 2:
        return SpectralResult(
            shape=shape,
            algorithm="none",
            computation_status="skipped",
            skip_reason=f"Tensor has {arr.ndim} dimensions; spectral analysis needs a 2D matrix.",
            parameters=params,
        )

    if not np.all(np.isfinite(arr)):
        return SpectralResult(
            shape=shape,
            algorithm="none",
            computation_status="skipped",
            skip_reason="Matrix contains NaN/Inf values; SVD is undefined on non-finite input.",
            parameters=params,
        )

    rows, cols = shape
    if rows == 0 or cols == 0:
        return SpectralResult(
            shape=shape,
            algorithm="none",
            computation_status="skipped",
            skip_reason="Matrix has a zero-length dimension.",
            parameters=params,
        )

    n_elements = rows * cols
    min_dim = min(rows, cols)
    arr64 = arr.astype(np.float64, copy=False)

    if not np.any(arr64):
        # Zero matrix: SVD is trivially all-zero singular values.
        return SpectralResult(
            shape=shape,
            algorithm="full_svd",
            computation_status="computed",
            parameters=params,
            top_singular_values=[0.0] * min(min_dim, truncated_k),
            spectral_norm=0.0,
            effective_rank=0.0,
            condition_number=None,
            condition_number_note="Zero matrix: condition number is undefined (0/0).",
            spectral_entropy=0.0,
        )

    use_full = n_elements <= max_full_svd_elements or min_dim <= truncated_k + 1

    if use_full:
        singular_values = np.linalg.svd(arr64, compute_uv=False)
        entropy, effective_rank = _spectral_entropy_and_effective_rank(singular_values)
        sigma_max = float(singular_values[0])
        sigma_min = float(singular_values[-1])
        if sigma_min > 0:
            condition_number = sigma_max / sigma_min
            cond_note = None
        else:
            condition_number = None
            cond_note = "Smallest singular value is 0 (rank-deficient): condition number infinite."

        return SpectralResult(
            shape=shape,
            algorithm="full_svd",
            computation_status="computed",
            parameters=params,
            top_singular_values=[float(s) for s in singular_values[:truncated_k]],
            spectral_norm=sigma_max,
            effective_rank=effective_rank,
            condition_number=condition_number,
            condition_number_note=cond_note,
            spectral_entropy=entropy,
        )

    # Large matrix: approximate via top-k truncated SVD (ARPACK). This only
    # yields the k largest singular values, so effective rank and spectral
    # entropy computed from them are lower-bound approximations (missing
    # mass from the untruncated tail), and condition number is not computed
    # at all since it needs the smallest singular value.
    k = min(truncated_k, min_dim - 1)
    _, singular_values, _ = svds(arr64, k=k)
    singular_values = np.sort(singular_values)[::-1]  # svds returns ascending order
    entropy, effective_rank = _spectral_entropy_and_effective_rank(singular_values)

    return SpectralResult(
        shape=shape,
        algorithm="truncated_svd_topk",
        computation_status="computed",
        parameters=params,
        top_singular_values=[float(s) for s in singular_values],
        spectral_norm=float(singular_values[0]),
        effective_rank=effective_rank,
        condition_number=None,
        condition_number_note=(
            f"Not computed: matrix exceeds {max_full_svd_elements} elements, so only the top-{k} "
            "singular values were computed (ARPACK) and the smallest singular value is unknown. "
            "effective_rank/spectral_entropy below are approximations from the top-k values only."
        ),
        spectral_entropy=entropy,
    )
