from __future__ import annotations

import numpy as np

from neurofence.activation.anomaly import detect_activation_anomalies
from neurofence.weight_forensics.statistics import compute_tensor_statistics


def _stats_from(values: np.ndarray):
    return compute_tensor_statistics(values)


def test_normal_layers_no_outliers() -> None:
    rng = np.random.default_rng(0)
    layer_stats = {f"layer{i}": _stats_from(rng.standard_normal(64)) for i in range(20)}

    summary = detect_activation_anomalies(layer_stats)

    assert summary.status == "ml_based"
    assert len(summary.results) == 20


def test_outlier_layer_flagged() -> None:
    rng = np.random.default_rng(0)
    layer_stats = {f"layer{i}": _stats_from(rng.standard_normal(64)) for i in range(20)}
    layer_stats["layer_weird"] = _stats_from(rng.standard_normal(64) * 100 + 500)

    summary = detect_activation_anomalies(layer_stats)

    by_name = {r.layer_name: r for r in summary.results}
    assert by_name["layer_weird"].is_outlier is True


def test_few_layers_uses_fallback() -> None:
    rng = np.random.default_rng(0)
    layer_stats = {f"layer{i}": _stats_from(rng.standard_normal(64)) for i in range(3)}
    summary = detect_activation_anomalies(layer_stats)
    assert summary.status == "statistical_fallback"


def test_empty_layer_stats() -> None:
    summary = detect_activation_anomalies({})
    assert summary.status == "empty"
