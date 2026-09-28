"""Compares activation patterns between baseline prompts and candidate-
trigger prompts, for one chosen layer.

Two independent, corroborating signals:

1. Distance-based: how far each trigger prompt's activation lands from the
   baseline centroid, scored as a robust z-score against the baseline
   group's own internal spread (so "elevated" means "further from the
   baseline centroid than baseline prompts normally are from each other,"
   not an arbitrary absolute threshold).
2. Cluster-based: whether trigger prompts fall outside the baseline
   group's dominant DBSCAN cluster.

As with fuzzing.trigger_discovery, a single divergent example proves
nothing; `consistent_separation` requires a strict majority of tested
trigger prompts to show elevated distance.
"""

from __future__ import annotations

import numpy as np
from pydantic import BaseModel

from neurofence.activation.clustering import MIN_SAMPLES_FOR_CLUSTERING, cluster_activations
from neurofence.weight_forensics.pipeline import build_layer_feature_vector
from neurofence.weight_forensics.robust import median_absolute_deviation
from neurofence.weight_forensics.statistics import TensorStatistics

# Matches weight_forensics.anomaly.DEFAULT_MIN_SAMPLES_FOR_ML: MAD/median
# computed from fewer samples than this is itself too noisy an estimate of
# "normal" to threshold against reliably (verified empirically -- two
# groups drawn from the *same* distribution at n=6 baseline / n=4 trigger
# produced spurious "consistent separation" purely from baseline-MAD
# sampling noise, not from any real behavioral difference).
MIN_BASELINE_SAMPLES = 8
_ROBUST_Z_CONST = 0.6745
ELEVATED_Z_THRESHOLD = 2.5


class TriggerActivationEvidence(BaseModel):
    case_id: str
    distance_from_baseline_centroid: float
    robust_z_score: float
    elevated: bool


class TriggerActivationResult(BaseModel):
    layer_name: str
    status: str  # "analyzed" | "skipped"
    skip_reason: str | None = None
    num_baseline_samples: int = 0
    num_trigger_samples: int = 0
    consistent_separation: bool = False
    evidence: list[TriggerActivationEvidence] = []
    cluster_note: str | None = None


def _robust_z_against_reference(values: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Robust z-score of `values` against a *reference* distribution's
    median/MAD (unlike mad_robust_zscore, which scores a distribution
    against itself). Falls back to std, then to zero, exactly as
    neurofence.weight_forensics.robust.mad_robust_zscore does, for the
    same reasons (MAD == 0 must never divide by zero).
    """
    ref_median = float(np.median(reference))
    mad = median_absolute_deviation(reference)
    if mad > 0:
        return _ROBUST_Z_CONST * (values - ref_median) / mad
    std = float(np.std(reference))
    if std > 0:
        return (values - ref_median) / std
    return np.zeros_like(values)


def _scale_columns_against_baseline(
    matrix: np.ndarray, baseline_reference: np.ndarray
) -> np.ndarray:
    """Per-feature-column robust scaling using the *baseline* group's own
    median/MAD as the reference for "normal" -- applied to both baseline
    and trigger points so distances are computed in a common, comparable
    space.

    Without this, raw Euclidean distance over feature columns with very
    different natural scales/variances (e.g. skewness/kurtosis, which are
    inherently noisy for small activation samples, vs. mean/sparsity,
    which are not) lets whichever column happens to have the largest
    absolute spread dominate the distance regardless of whether it
    reflects a real behavioral difference -- exactly the class of bug
    fixed in weight_forensics.pipeline.build_layer_feature_vector for
    l1/l2 norms, generalized here to every feature column.
    """
    n_cols = matrix.shape[1]
    scaled = np.zeros_like(matrix)
    for col in range(n_cols):
        ref_col = baseline_reference[:, col]
        median = float(np.median(ref_col))
        mad = median_absolute_deviation(ref_col)
        if mad > 0:
            scaled[:, col] = _ROBUST_Z_CONST * (matrix[:, col] - median) / mad
        else:
            std = float(np.std(ref_col))
            scaled[:, col] = (matrix[:, col] - median) / std if std > 0 else 0.0
    return scaled


def analyze_trigger_activations(
    baseline_stats: dict[str, dict[str, TensorStatistics]],  # case_id -> layer -> stats
    trigger_stats: dict[str, dict[str, TensorStatistics]],
    layer_name: str,
) -> TriggerActivationResult:
    baseline_features = {
        cid: build_layer_feature_vector(layers[layer_name])
        for cid, layers in baseline_stats.items()
        if layer_name in layers
    }
    trigger_features = {
        cid: build_layer_feature_vector(layers[layer_name])
        for cid, layers in trigger_stats.items()
        if layer_name in layers
    }

    if len(baseline_features) < MIN_BASELINE_SAMPLES:
        return TriggerActivationResult(
            layer_name=layer_name,
            status="skipped",
            skip_reason=(
                f"Need at least {MIN_BASELINE_SAMPLES} baseline samples with layer "
                f"'{layer_name}' captured, got {len(baseline_features)}."
            ),
        )
    if not trigger_features:
        return TriggerActivationResult(
            layer_name=layer_name,
            status="skipped",
            skip_reason=f"No trigger-prompt samples have layer '{layer_name}' captured.",
            num_baseline_samples=len(baseline_features),
        )

    baseline_ids = sorted(baseline_features)
    trigger_ids = sorted(trigger_features)
    baseline_matrix_raw = np.vstack([baseline_features[i] for i in baseline_ids])
    trigger_matrix_raw = np.vstack([trigger_features[i] for i in trigger_ids])

    baseline_matrix = _scale_columns_against_baseline(baseline_matrix_raw, baseline_matrix_raw)
    trigger_matrix = _scale_columns_against_baseline(trigger_matrix_raw, baseline_matrix_raw)

    centroid = np.mean(baseline_matrix, axis=0)
    baseline_self_distances = np.linalg.norm(baseline_matrix - centroid, axis=1)
    trigger_distances = np.linalg.norm(trigger_matrix - centroid, axis=1)

    z_scores = _robust_z_against_reference(trigger_distances, baseline_self_distances)
    elevated = z_scores > ELEVATED_Z_THRESHOLD

    evidence = [
        TriggerActivationEvidence(
            case_id=cid,
            distance_from_baseline_centroid=float(trigger_distances[i]),
            robust_z_score=float(z_scores[i]),
            elevated=bool(elevated[i]),
        )
        for i, cid in enumerate(trigger_ids)
    ]

    elevated_fraction = float(np.mean(elevated))
    consistent_separation = elevated_fraction > 0.5

    cluster_note = None
    combined = np.vstack([baseline_matrix, trigger_matrix])
    if combined.shape[0] >= MIN_SAMPLES_FOR_CLUSTERING:
        clustering = cluster_activations(combined)
        if clustering.status == "computed":
            baseline_labels = clustering.labels[: len(baseline_ids)]
            trigger_labels = clustering.labels[len(baseline_ids) :]
            dominant = max(set(baseline_labels), key=baseline_labels.count)
            outside_fraction = sum(1 for label in trigger_labels if label != dominant) / len(
                trigger_labels
            )
            cluster_note = (
                f"{outside_fraction:.0%} of trigger samples fall outside the baseline "
                f"group's dominant DBSCAN cluster (eps={clustering.parameters['eps']:.4g})."
            )
        else:
            cluster_note = f"Clustering not computed: {clustering.skip_reason}"

    return TriggerActivationResult(
        layer_name=layer_name,
        status="analyzed",
        num_baseline_samples=len(baseline_features),
        num_trigger_samples=len(trigger_features),
        consistent_separation=consistent_separation,
        evidence=evidence,
        cluster_note=cluster_note,
    )
