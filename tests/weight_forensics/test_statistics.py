from __future__ import annotations

import numpy as np

from neurofence.weight_forensics.statistics import compute_tensor_statistics


def test_normal_tensor_statistics() -> None:
    rng = np.random.default_rng(0)
    arr = rng.standard_normal((100, 100)).astype(np.float32)

    stats = compute_tensor_statistics(arr)

    assert stats.status == "ok"
    assert stats.element_count == 10_000
    assert stats.finite_count == 10_000
    assert stats.mean is not None and abs(stats.mean) < 0.1
    assert stats.std is not None and 0.9 < stats.std < 1.1
    assert stats.l2_norm is not None and stats.l2_norm > 0
    assert stats.notes == []


def test_empty_tensor() -> None:
    stats = compute_tensor_statistics(np.zeros((0,)))
    assert stats.status == "empty"
    assert stats.element_count == 0
    assert stats.mean is None


def test_zero_length_dimension_tensor() -> None:
    stats = compute_tensor_statistics(np.zeros((5, 0)))
    assert stats.status == "empty"


def test_all_nan_tensor() -> None:
    stats = compute_tensor_statistics(np.full((10,), np.nan))
    assert stats.status == "all_non_finite"
    assert stats.nan_count == 10
    assert stats.mean is None


def test_all_inf_tensor() -> None:
    stats = compute_tensor_statistics(np.full((5,), np.inf))
    assert stats.status == "all_non_finite"
    assert stats.inf_count == 5


def test_mixed_nan_inf_and_finite_values() -> None:
    arr = np.array([1.0, 2.0, np.nan, np.inf, -np.inf, 3.0])
    stats = compute_tensor_statistics(arr)
    assert stats.status == "ok"
    assert stats.finite_count == 3
    assert stats.nan_count == 1
    assert stats.inf_count == 2
    assert stats.mean == 2.0
    assert len(stats.notes) == 2


def test_constant_tensor() -> None:
    stats = compute_tensor_statistics(np.full((50,), 7.0))
    assert stats.status == "ok"
    assert stats.std == 0.0
    assert stats.skewness == 0.0
    assert stats.kurtosis == 0.0
    assert any("constant tensor" in n for n in stats.notes)


def test_zero_tensor() -> None:
    stats = compute_tensor_statistics(np.zeros((20, 20)))
    assert stats.status == "ok"
    assert stats.mean == 0.0
    assert stats.sparsity == 1.0
    assert stats.zero_ratio == 1.0
    assert stats.l1_norm == 0.0
    assert stats.l2_norm == 0.0


def test_tiny_tensor_single_element() -> None:
    stats = compute_tensor_statistics(np.array([42.0]))
    assert stats.status == "ok"
    assert stats.mean == 42.0
    assert stats.std == 0.0
    assert stats.median == 42.0


def test_two_element_tensor() -> None:
    stats = compute_tensor_statistics(np.array([1.0, 3.0]))
    assert stats.status == "ok"
    assert stats.mean == 2.0


def test_float16_no_overflow() -> None:
    # float16 max is ~65504; sum-of-squares in a naive float16 implementation
    # would overflow. Statistics must upcast internally.
    arr = np.full((1000,), 1000.0, dtype=np.float16)
    stats = compute_tensor_statistics(arr)
    assert stats.status == "ok"
    assert stats.l2_norm is not None and np.isfinite(stats.l2_norm)
    assert stats.mean == 1000.0


def test_extremely_large_values() -> None:
    arr = np.array([1e300, 2e300, 3e300])
    stats = compute_tensor_statistics(arr)
    assert stats.status == "ok"
    assert stats.mean is not None and np.isfinite(stats.mean)
    # Regression: sum-of-squares/cubes on values this large overflows
    # float64. Overflowing statistics must be reported as None with an
    # explanatory note, never as a silent inf/NaN that could poison
    # downstream anomaly scoring.
    assert stats.std is None
    assert stats.variance is None
    assert stats.l2_norm is None
    assert stats.skewness is None
    assert stats.kurtosis is None
    assert any("overflow" in n for n in stats.notes)
    for field_name in ("std", "variance", "l2_norm", "skewness", "kurtosis", "l1_norm", "mean"):
        value = getattr(stats, field_name)
        assert value is None or np.isfinite(value), f"{field_name} leaked a non-finite value"


def test_extremely_small_values() -> None:
    arr = np.array([1e-300, 2e-300, 3e-300])
    stats = compute_tensor_statistics(arr)
    assert stats.status == "ok"
    assert stats.mean is not None and stats.mean > 0


def test_sparsity_partial() -> None:
    arr = np.array([0.0, 0.0, 1.0, 2.0, 0.0])
    stats = compute_tensor_statistics(arr)
    assert stats.sparsity == 3 / 5


def test_percentiles_present_and_ordered() -> None:
    arr = np.arange(1, 101, dtype=np.float64)
    stats = compute_tensor_statistics(arr)
    assert stats.percentile_1 <= stats.percentile_5 <= stats.percentile_25
    assert stats.percentile_25 <= stats.percentile_75 <= stats.percentile_95 <= stats.percentile_99


def test_float64_dtype_recorded() -> None:
    arr = np.zeros((3, 3), dtype=np.float64)
    stats = compute_tensor_statistics(arr)
    assert stats.dtype == "float64"
    assert stats.shape == [3, 3]
