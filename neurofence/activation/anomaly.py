"""Layer-level activation anomaly detection.

Deliberately thin: this reuses neurofence.weight_forensics.anomaly's
Isolation Forest / Local Outlier Factor / Mahalanobis consensus detector
unchanged -- the algorithm doesn't care whether a feature vector summarizes
a weight tensor or an activation tensor, so there is no reason to
duplicate it. This answers "which layer's activations look unusual
relative to the other layers, for this one forward pass."
"""

from __future__ import annotations

from neurofence.weight_forensics.anomaly import (
    DEFAULT_MIN_SAMPLES_FOR_ML,
    AnomalyDetectionSummary,
    detect_layer_anomalies,
)
from neurofence.weight_forensics.pipeline import build_layer_feature_vector
from neurofence.weight_forensics.statistics import TensorStatistics


def detect_activation_anomalies(
    layer_stats: dict[str, TensorStatistics],
    contamination: float = 0.1,
    n_neighbors: int = 5,
    random_seed: int = 1337,
    min_samples_for_ml: int = DEFAULT_MIN_SAMPLES_FOR_ML,
) -> AnomalyDetectionSummary:
    features = {name: build_layer_feature_vector(stats) for name, stats in layer_stats.items()}
    return detect_layer_anomalies(
        features,
        contamination=contamination,
        n_neighbors=n_neighbors,
        random_seed=random_seed,
        min_samples_for_ml=min_samples_for_ml,
    )
