from __future__ import annotations

import numpy as np

from neurofence.weight_forensics.anomaly import DEFAULT_MIN_SAMPLES_FOR_ML, detect_layer_anomalies


def _make_normal_layers(n: int, d: int, seed: int = 0) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    return {f"layer_{i:03d}": rng.normal(loc=0.0, scale=1.0, size=d) for i in range(n)}


def test_empty_input() -> None:
    summary = detect_layer_anomalies({})
    assert summary.status == "empty"
    assert summary.results == []


def test_zero_length_feature_vectors() -> None:
    summary = detect_layer_anomalies({"a": np.array([]), "b": np.array([])})
    assert summary.status == "empty"


def test_too_few_samples_uses_statistical_fallback() -> None:
    layers = _make_normal_layers(5, 4)
    summary = detect_layer_anomalies(layers, min_samples_for_ml=DEFAULT_MIN_SAMPLES_FOR_ML)
    assert summary.status == "statistical_fallback"
    assert summary.sample_count == 5
    assert len(summary.results) == 5


def test_samples_not_exceeding_features_uses_fallback() -> None:
    # 10 samples but 12 features -> n <= d, ML methods would be unreliable.
    layers = {f"l{i}": np.random.default_rng(i).normal(size=12) for i in range(10)}
    summary = detect_layer_anomalies(layers)
    assert summary.status == "statistical_fallback"


def test_ml_based_path_runs_with_enough_samples() -> None:
    layers = _make_normal_layers(30, 5)
    summary = detect_layer_anomalies(layers)
    assert summary.status == "ml_based"
    assert set(summary.methods_used) == {"isolation_forest", "local_outlier_factor", "mahalanobis"}
    assert len(summary.results) == 30


def test_ml_based_flags_obvious_outlier() -> None:
    layers = _make_normal_layers(30, 5)
    layers["layer_outlier"] = np.array([500.0, 500.0, 500.0, 500.0, 500.0])

    summary = detect_layer_anomalies(layers)
    by_name = {r.layer_name: r for r in summary.results}

    assert by_name["layer_outlier"].is_outlier is True
    # the extreme outlier should have the highest anomaly score of the batch
    max_score_layer = max(summary.results, key=lambda r: r.anomaly_score)
    assert max_score_layer.layer_name == "layer_outlier"


def test_statistical_fallback_flags_obvious_outlier() -> None:
    layers = _make_normal_layers(4, 3)
    layers["layer_outlier"] = np.array([1000.0, 1000.0, 1000.0])

    summary = detect_layer_anomalies(layers, min_samples_for_ml=DEFAULT_MIN_SAMPLES_FOR_ML)
    by_name = {r.layer_name: r for r in summary.results}

    assert summary.status == "statistical_fallback"
    assert by_name["layer_outlier"].is_outlier is True


def test_deterministic_ordering() -> None:
    layers = _make_normal_layers(20, 4)
    s1 = detect_layer_anomalies(layers, random_seed=42)
    s2 = detect_layer_anomalies(layers, random_seed=42)
    assert [r.layer_name for r in s1.results] == [r.layer_name for r in s2.results]
    assert [r.anomaly_score for r in s1.results] == [r.anomaly_score for r in s2.results]


def test_non_finite_features_do_not_crash() -> None:
    layers = _make_normal_layers(15, 4)
    layers["layer_bad"] = np.array([np.nan, np.inf, -np.inf, 1.0])

    summary = detect_layer_anomalies(layers)
    assert all(np.isfinite(r.anomaly_score) for r in summary.results)


def test_all_identical_layers_no_false_positive_storm() -> None:
    # If every layer is identical, nothing should be flagged as an outlier
    # (there is no "normal" to deviate from).
    layers = {f"l{i}": np.array([1.0, 2.0, 3.0, 4.0]) for i in range(20)}
    summary = detect_layer_anomalies(layers)
    flagged = [r for r in summary.results if r.is_outlier]
    assert len(flagged) == 0


def test_single_layer() -> None:
    summary = detect_layer_anomalies({"only_layer": np.array([1.0, 2.0, 3.0])})
    assert summary.status == "statistical_fallback"
    assert len(summary.results) == 1
    assert summary.results[0].is_outlier is False


def test_contamination_clipped_to_valid_range() -> None:
    layers = _make_normal_layers(30, 5)
    # Should not raise even with an out-of-range contamination value.
    summary = detect_layer_anomalies(layers, contamination=0.9)
    assert summary.status == "ml_based"
