from __future__ import annotations

import numpy as np

from neurofence.activation.reduction import reduce_dimensionality


def test_reduce_dimensionality_normal_case() -> None:
    rng = np.random.default_rng(0)
    x = rng.standard_normal((20, 10))
    result = reduce_dimensionality(x, n_components=2)

    assert result.status == "computed"
    assert result.n_components == 2
    assert len(result.coordinates) == 20
    assert len(result.explained_variance_ratio) == 2


def test_reduce_dimensionality_too_few_samples() -> None:
    result = reduce_dimensionality(np.array([[1.0, 2.0, 3.0]]))
    assert result.status == "skipped"
    assert "samples" in result.skip_reason


def test_reduce_dimensionality_zero_samples() -> None:
    result = reduce_dimensionality(np.zeros((0, 5)))
    assert result.status == "skipped"


def test_reduce_dimensionality_1d_input_skipped() -> None:
    result = reduce_dimensionality(np.array([1.0, 2.0, 3.0]))
    assert result.status == "skipped"


def test_reduce_dimensionality_zero_features() -> None:
    result = reduce_dimensionality(np.zeros((5, 0)))
    assert result.status == "skipped"


def test_reduce_dimensionality_nan_input_skipped() -> None:
    x = np.array([[1.0, np.nan], [2.0, 3.0], [4.0, 5.0]])
    result = reduce_dimensionality(x)
    assert result.status == "skipped"
    assert "NaN" in result.skip_reason or "Inf" in result.skip_reason


def test_reduce_dimensionality_requested_components_exceeds_features() -> None:
    x = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    result = reduce_dimensionality(x, n_components=10)
    assert result.status == "computed"
    assert result.n_components == 2  # clipped to n_features


def test_reduce_dimensionality_constant_data() -> None:
    # Regression: sklearn's explained_variance_ratio_ is variance/total_variance,
    # a 0/0 NaN when every sample is identical (zero total variance).
    x = np.ones((5, 4))
    result = reduce_dimensionality(x, n_components=2)
    assert result.status == "computed"
    assert all(np.isfinite(v) for v in result.explained_variance_ratio)
    assert result.explained_variance_ratio == [0.0, 0.0]
