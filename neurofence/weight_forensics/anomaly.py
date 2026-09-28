"""Layer-level anomaly detection over per-tensor statistical feature vectors.

Three independent, complementary detectors run when there are enough
samples:

- Isolation Forest: global structural outliers (few splits needed to
  isolate a point).
- Local Outlier Factor: outliers relative to their local neighborhood
  density (catches anomalies that are "normal" globally but stick out
  among similar layers, e.g. all attention-output projections but one).
- Mahalanobis distance: multivariate deviation from the baseline centroid,
  accounting for feature correlation.

A layer is flagged as an outlier only when at least two of the three
methods agree -- consensus among independent signals, per the project's
multi-signal principle, rather than any single detector's opinion.

When there are too few layers for these estimators to be meaningful
(fewer than `min_samples_for_ml`, or fewer samples than features), all
three are skipped and a documented, robust statistical fallback (per-
feature MAD-based z-scores) is used instead. This is never silent -- the
result's `status` field always says which path ran and why.
"""

from __future__ import annotations

import numpy as np
from pydantic import BaseModel
from scipy import stats as sp_stats
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

from neurofence.weight_forensics.robust import mad_robust_zscore

DEFAULT_MIN_SAMPLES_FOR_ML = 8
ROBUST_Z_OUTLIER_THRESHOLD = 3.5  # Iglewicz & Hoaglin (1993) conventional cutoff
MAHALANOBIS_CHI2_ALPHA = 0.025  # upper-tail significance for flagging


class LayerAnomalyResult(BaseModel):
    layer_name: str
    is_outlier: bool
    anomaly_score: float  # higher = more anomalous; comparable across layers in this run only
    votes: dict[str, bool] = {}
    detail: dict[str, float | str | None] = {}


class AnomalyDetectionSummary(BaseModel):
    status: str  # "ml_based" | "statistical_fallback" | "empty"
    reason: str | None = None
    sample_count: int
    feature_count: int
    methods_used: list[str] = []
    results: list[LayerAnomalyResult] = []


def _normalize_0_1(x: np.ndarray) -> np.ndarray:
    """Min-max normalize; returns all-zeros if x is constant (no spread)."""
    lo, hi = float(np.min(x)), float(np.max(x))
    if hi - lo <= 0:
        return np.zeros_like(x)
    return (x - lo) / (hi - lo)


def _statistical_fallback(names: list[str], x: np.ndarray) -> AnomalyDetectionSummary:
    # Robust z-score per feature column across layers, combined as the max
    # absolute z-score per layer (i.e. "how anomalous is this layer's most
    # anomalous single feature").
    n, d = x.shape
    z = np.zeros_like(x)
    for col in range(d):
        z[:, col] = mad_robust_zscore(x[:, col])

    combined = np.max(np.abs(z), axis=1)
    results = []
    for i, name in enumerate(names):
        is_outlier = bool(combined[i] > ROBUST_Z_OUTLIER_THRESHOLD)
        results.append(
            LayerAnomalyResult(
                layer_name=name,
                is_outlier=is_outlier,
                anomaly_score=float(combined[i]),
                votes={"robust_zscore": is_outlier},
                detail={"max_abs_robust_zscore": float(combined[i])},
            )
        )

    return AnomalyDetectionSummary(
        status="statistical_fallback",
        reason=(
            f"Only {n} layer(s) with {d} feature(s): too few samples for Isolation "
            f"Forest / LOF / Mahalanobis to be meaningful (need >= "
            f"{DEFAULT_MIN_SAMPLES_FOR_ML} samples and more samples than features). "
            "Used MAD-based robust z-score per feature instead."
        ),
        sample_count=n,
        feature_count=d,
        methods_used=["robust_zscore"],
        results=results,
    )


def detect_layer_anomalies(
    layer_features: dict[str, np.ndarray],
    contamination: float = 0.1,
    n_neighbors: int = 5,
    random_seed: int = 1337,
    min_samples_for_ml: int = DEFAULT_MIN_SAMPLES_FOR_ML,
) -> AnomalyDetectionSummary:
    """Detect anomalous layers from a dict of layer_name -> feature vector.

    All feature vectors must have the same length. Layers are processed in
    sorted-name order for determinism.
    """
    names = sorted(layer_features)
    if not names:
        return AnomalyDetectionSummary(
            status="empty", reason="No layers provided.", sample_count=0, feature_count=0
        )

    x = np.vstack([np.asarray(layer_features[n], dtype=np.float64) for n in names])
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)  # features should already be finite
    n, d = x.shape

    if d == 0:
        return AnomalyDetectionSummary(
            status="empty",
            reason="Feature vectors have zero length.",
            sample_count=n,
            feature_count=0,
        )

    if n < min_samples_for_ml or n <= d:
        return _statistical_fallback(names, x)

    # Robust-scale each feature (median/MAD) before distance-based methods
    # so that features on very different scales (e.g. mean vs l2_norm)
    # don't dominate.
    x_scaled = np.column_stack([mad_robust_zscore(x[:, col]) for col in range(d)])

    contamination = float(np.clip(contamination, 1e-3, 0.5))
    k_neighbors = max(1, min(n_neighbors, n - 1))

    iso = IsolationForest(contamination=contamination, random_state=random_seed, n_estimators=200)
    iso_labels = iso.fit_predict(x_scaled)  # -1 outlier, 1 inlier
    iso_scores = -iso.score_samples(x_scaled)  # higher = more anomalous
    iso_norm = _normalize_0_1(iso_scores)

    lof = LocalOutlierFactor(n_neighbors=k_neighbors, contamination=contamination)
    lof_labels = lof.fit_predict(x_scaled)  # -1 outlier, 1 inlier
    lof_scores = -lof.negative_outlier_factor_  # higher = more anomalous
    lof_norm = _normalize_0_1(lof_scores)

    mean_vec = np.mean(x_scaled, axis=0)
    cov = np.cov(x_scaled, rowvar=False)
    cov = np.atleast_2d(cov)
    mahal_note = None
    try:
        inv_cov = np.linalg.inv(cov)
    except np.linalg.LinAlgError:
        inv_cov = np.linalg.pinv(cov)
        mahal_note = "Covariance matrix singular; used Moore-Penrose pseudo-inverse."

    diffs = x_scaled - mean_vec
    mahal_sq = np.einsum("ij,jk,ik->i", diffs, inv_cov, diffs)
    mahal_sq = np.clip(mahal_sq, 0, None)  # guard tiny negative values from numerical error
    mahal_dist = np.sqrt(mahal_sq)
    chi2_threshold = float(np.sqrt(sp_stats.chi2.ppf(1 - MAHALANOBIS_CHI2_ALPHA, df=d)))
    mahal_outlier = mahal_dist > chi2_threshold
    mahal_norm = _normalize_0_1(mahal_dist)

    results = []
    for i, name in enumerate(names):
        votes = {
            "isolation_forest": bool(iso_labels[i] == -1),
            "local_outlier_factor": bool(lof_labels[i] == -1),
            "mahalanobis": bool(mahal_outlier[i]),
        }
        vote_count = sum(votes.values())
        combined_score = float(np.mean([iso_norm[i], lof_norm[i], mahal_norm[i]]))
        detail: dict[str, float | str | None] = {
            "isolation_forest_score": float(iso_scores[i]),
            "lof_score": float(lof_scores[i]),
            "mahalanobis_distance": float(mahal_dist[i]),
            "mahalanobis_threshold": chi2_threshold,
        }
        if mahal_note:
            detail["mahalanobis_note"] = mahal_note

        results.append(
            LayerAnomalyResult(
                layer_name=name,
                is_outlier=vote_count >= 2,
                anomaly_score=combined_score,
                votes=votes,
                detail=detail,
            )
        )

    return AnomalyDetectionSummary(
        status="ml_based",
        sample_count=n,
        feature_count=d,
        methods_used=["isolation_forest", "local_outlier_factor", "mahalanobis"],
        results=results,
    )
