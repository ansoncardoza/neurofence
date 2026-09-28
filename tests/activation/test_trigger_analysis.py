from __future__ import annotations

import numpy as np

from neurofence.activation.trigger_analysis import MIN_BASELINE_SAMPLES, analyze_trigger_activations
from neurofence.weight_forensics.statistics import compute_tensor_statistics


def _make_stats_dict(case_ids: list[str], layer_name: str, values_by_case: dict[str, np.ndarray]):
    return {cid: {layer_name: compute_tensor_statistics(values_by_case[cid])} for cid in case_ids}


def test_trigger_prompts_clearly_separated() -> None:
    rng = np.random.default_rng(0)
    baseline_ids = [f"b{i}" for i in range(12)]
    trigger_ids = [f"t{i}" for i in range(4)]

    baseline_values = {cid: rng.normal(0.0, 1.0, size=32) for cid in baseline_ids}
    trigger_values = {cid: rng.normal(0.0, 1.0, size=32) + 500.0 for cid in trigger_ids}

    baseline_stats = _make_stats_dict(baseline_ids, "layerA", baseline_values)
    trigger_stats = _make_stats_dict(trigger_ids, "layerA", trigger_values)

    result = analyze_trigger_activations(baseline_stats, trigger_stats, "layerA")

    assert result.status == "analyzed"
    assert result.consistent_separation is True
    assert all(e.elevated for e in result.evidence)
    assert result.cluster_note is not None


def test_inert_trigger_not_separated() -> None:
    # Baseline and "trigger" prompts drawn from the identical distribution
    # -- with enough baseline samples for the per-feature MAD/median
    # reference to be a reliable estimate, this must not spuriously
    # separate. (Regression: at baseline n=6 this produced false
    # "consistent separation" purely from small-sample MAD noise, which is
    # why MIN_BASELINE_SAMPLES was raised to match
    # weight_forensics.anomaly.DEFAULT_MIN_SAMPLES_FOR_ML.)
    rng = np.random.default_rng(0)
    baseline_ids = [f"b{i}" for i in range(12)]
    trigger_ids = [f"t{i}" for i in range(4)]

    baseline_values = {cid: rng.normal(0.0, 1.0, size=32) for cid in baseline_ids}
    trigger_values = {cid: rng.normal(0.0, 1.0, size=32) for cid in trigger_ids}

    baseline_stats = _make_stats_dict(baseline_ids, "layerA", baseline_values)
    trigger_stats = _make_stats_dict(trigger_ids, "layerA", trigger_values)

    result = analyze_trigger_activations(baseline_stats, trigger_stats, "layerA")

    assert result.status == "analyzed"
    assert result.consistent_separation is False


def test_too_few_baseline_samples_skipped() -> None:
    rng = np.random.default_rng(0)
    baseline_stats = _make_stats_dict(["b0"], "layerA", {"b0": rng.standard_normal(16)})
    trigger_stats = _make_stats_dict(["t0"], "layerA", {"t0": rng.standard_normal(16)})

    result = analyze_trigger_activations(baseline_stats, trigger_stats, "layerA")

    assert result.status == "skipped"
    assert "baseline" in result.skip_reason


def test_below_min_baseline_threshold_skipped() -> None:
    rng = np.random.default_rng(0)
    ids = [f"b{i}" for i in range(MIN_BASELINE_SAMPLES - 1)]
    baseline_stats = _make_stats_dict(ids, "layerA", {cid: rng.standard_normal(16) for cid in ids})
    trigger_stats = _make_stats_dict(["t0"], "layerA", {"t0": rng.standard_normal(16)})

    result = analyze_trigger_activations(baseline_stats, trigger_stats, "layerA")

    assert result.status == "skipped"


def test_at_min_baseline_threshold_analyzed() -> None:
    rng = np.random.default_rng(0)
    ids = [f"b{i}" for i in range(MIN_BASELINE_SAMPLES)]
    baseline_stats = _make_stats_dict(ids, "layerA", {cid: rng.standard_normal(16) for cid in ids})
    trigger_stats = _make_stats_dict(["t0"], "layerA", {"t0": rng.standard_normal(16)})

    result = analyze_trigger_activations(baseline_stats, trigger_stats, "layerA")

    assert result.status == "analyzed"


def test_no_trigger_samples_skipped() -> None:
    rng = np.random.default_rng(0)
    baseline_ids = [f"b{i}" for i in range(MIN_BASELINE_SAMPLES)]
    baseline_stats = _make_stats_dict(
        baseline_ids, "layerA", {cid: rng.standard_normal(16) for cid in baseline_ids}
    )

    result = analyze_trigger_activations(baseline_stats, {}, "layerA")

    assert result.status == "skipped"


def test_missing_layer_in_some_samples_filtered_not_crashed() -> None:
    rng = np.random.default_rng(0)
    baseline_stats = {
        f"b{i}": {"layerA": compute_tensor_statistics(rng.standard_normal(16))}
        for i in range(MIN_BASELINE_SAMPLES)
    }
    baseline_stats["b_missing"] = {"layerB": compute_tensor_statistics(rng.standard_normal(16))}
    trigger_stats = {
        "t0": {"layerA": compute_tensor_statistics(rng.standard_normal(16))},
    }

    result = analyze_trigger_activations(baseline_stats, trigger_stats, "layerA")

    assert result.status == "analyzed"
    assert result.num_baseline_samples == MIN_BASELINE_SAMPLES  # b_missing filtered out


def test_layer_missing_entirely_skipped() -> None:
    rng = np.random.default_rng(0)
    baseline_ids = [f"b{i}" for i in range(MIN_BASELINE_SAMPLES)]
    baseline_stats = _make_stats_dict(
        baseline_ids, "layerA", {cid: rng.standard_normal(16) for cid in baseline_ids}
    )
    trigger_stats = _make_stats_dict(["t0"], "layerA", {"t0": rng.standard_normal(16)})

    result = analyze_trigger_activations(baseline_stats, trigger_stats, "nonexistent_layer")

    assert result.status == "skipped"
