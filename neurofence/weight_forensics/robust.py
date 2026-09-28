"""Robust statistics: Median Absolute Deviation and MAD-based z-scores.

Robust statistics are used (instead of mean/std) for anomaly scoring because
a handful of poisoned weights should not drag the "normal" reference point
along with them the way a mean does.
"""

from __future__ import annotations

import numpy as np

# Consistency constant so MAD estimates the standard deviation for normally
# distributed data: 1 / Phi^-1(3/4) ~= 1.4826. The robust z-score formula
# 0.6745 * (x - median) / MAD is the inverse of this scaling.
_MAD_SCALE = 1.4826
_ROBUST_Z_CONST = 0.6745


def median_absolute_deviation(x: np.ndarray) -> float:
    """MAD = median(|x - median(x)|), ignoring NaNs.

    Returns 0.0 for an empty or all-NaN input.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    finite = x[np.isfinite(x)]
    if finite.size == 0:
        return 0.0
    med = np.nanmedian(finite)
    return float(np.nanmedian(np.abs(finite - med)))


def mad_robust_zscore(x: np.ndarray) -> np.ndarray:
    """Robust z-score: 0.6745 * (x - median) / MAD.

    Degenerate-input handling (never divides by zero or returns NaN/Inf):
    - Empty or all-non-finite input -> empty/zero array of matching shape.
    - MAD == 0 (e.g. constant tensor, or >=50% of values equal the median)
      falls back to a standard z-score using std when std > 0, and to an
      all-zero result (no signal) when std is also 0 -- there is nothing to
      compute a deviation against.
    - Non-finite elements of `x` map to 0.0 (no anomaly signal from a value
      that isn't even a comparable number).
    """
    x = np.asarray(x, dtype=np.float64)
    out = np.zeros_like(x, dtype=np.float64)

    finite_mask = np.isfinite(x)
    if not np.any(finite_mask):
        return out

    finite_vals = x[finite_mask]
    med = float(np.median(finite_vals))
    mad = median_absolute_deviation(finite_vals)

    if mad > 0:
        scores = _ROBUST_Z_CONST * (x - med) / mad
    else:
        std = float(np.std(finite_vals))
        if std > 0:
            scores = (x - med) / std
        else:
            # Constant (or effectively constant) data: no spread to compare
            # against, so report zero anomaly signal rather than divide by
            # zero or fabricate a score.
            scores = np.zeros_like(x, dtype=np.float64)

    scores = np.where(finite_mask, scores, 0.0)
    return scores
