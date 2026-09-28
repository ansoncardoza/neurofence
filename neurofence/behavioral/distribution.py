"""KL and Jensen-Shannon divergence between output probability distributions.

Used when a model exposes logits/probabilities over a shared vocabulary
(e.g. comparing next-token distributions for a baseline vs. candidate-
trigger prompt at a fixed position). Both distributions must have the same
length and be indexed over the same vocabulary -- NeuroFence does not
attempt to align distributions over different vocabularies.

KL divergence is not symmetric and is undefined (infinite) wherever q has
zero probability mass at a point p does not. We apply additive (Laplace)
smoothing before computing it so it is always a finite number; the smoothed
result is an approximation, not the exact KL divergence of the raw inputs,
and callers should prefer Jensen-Shannon divergence (symmetric, bounded,
well-defined without smoothing) unless KL is specifically required.
"""

from __future__ import annotations

import numpy as np


def _validate_and_normalize(p: np.ndarray, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)

    if p.shape != q.shape:
        raise ValueError(f"Distributions must have the same shape, got {p.shape} and {q.shape}.")
    if p.ndim != 1:
        raise ValueError("Distributions must be 1D vectors.")
    if p.size == 0:
        raise ValueError("Distributions must not be empty.")
    if not (np.all(np.isfinite(p)) and np.all(np.isfinite(q))):
        raise ValueError("Distributions must not contain NaN/Inf.")
    if np.any(p < 0) or np.any(q < 0):
        raise ValueError("Distributions must not contain negative values.")

    p_sum, q_sum = float(np.sum(p)), float(np.sum(q))
    if p_sum <= 0 or q_sum <= 0:
        raise ValueError("Distributions must have positive total mass (cannot be all-zero).")

    return p / p_sum, q / q_sum


def _smooth(p: np.ndarray, epsilon: float) -> np.ndarray:
    smoothed = p + epsilon
    return smoothed / np.sum(smoothed)


def kl_divergence(p: np.ndarray, q: np.ndarray, epsilon: float = 1e-10) -> float:
    """KL(p || q) in nats, with additive smoothing so it is always finite.

    Raises ValueError for shape mismatches, negative values, non-finite
    values, or all-zero input -- these indicate malformed probability
    vectors and must not be silently coerced into a misleading number.
    """
    p_norm, q_norm = _validate_and_normalize(p, q)
    p_s = _smooth(p_norm, epsilon)
    q_s = _smooth(q_norm, epsilon)
    return float(np.sum(p_s * np.log(p_s / q_s)))


def jensen_shannon_divergence(p: np.ndarray, q: np.ndarray, epsilon: float = 1e-10) -> float:
    """Symmetric, bounded ([0, ln 2] in nats) divergence: 0.5*KL(p||m) +
    0.5*KL(q||m), where m = 0.5*(p+q). Unlike raw KL this is well-defined
    without smoothing (m is zero only where both p and q are zero, and
    0*log(0/anything) is conventionally 0), but we apply the same light
    smoothing as kl_divergence for numerical consistency between the two.
    """
    p_norm, q_norm = _validate_and_normalize(p, q)
    p_s = _smooth(p_norm, epsilon)
    q_s = _smooth(q_norm, epsilon)
    m = 0.5 * (p_s + q_s)
    return float(0.5 * np.sum(p_s * np.log(p_s / m)) + 0.5 * np.sum(q_s * np.log(q_s / m)))
