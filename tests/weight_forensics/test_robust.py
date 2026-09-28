from __future__ import annotations

import numpy as np

from neurofence.weight_forensics.robust import mad_robust_zscore, median_absolute_deviation


def test_mad_normal_data() -> None:
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    assert median_absolute_deviation(x) == 1.0


def test_mad_empty_array_is_zero() -> None:
    assert median_absolute_deviation(np.array([])) == 0.0


def test_mad_all_nan_is_zero() -> None:
    assert median_absolute_deviation(np.array([np.nan, np.nan])) == 0.0


def test_mad_constant_array_is_zero() -> None:
    assert median_absolute_deviation(np.full(10, 5.0)) == 0.0


def test_robust_zscore_flags_outlier() -> None:
    x = np.array([1.0, 1.0, 1.0, 1.0, 100.0])
    z = mad_robust_zscore(x)
    assert abs(z[-1]) > abs(z[0])
    assert np.all(np.isfinite(z))


def test_robust_zscore_constant_array_all_zero() -> None:
    x = np.full(10, 3.0)
    z = mad_robust_zscore(x)
    assert np.all(z == 0.0)


def test_robust_zscore_empty_array() -> None:
    z = mad_robust_zscore(np.array([]))
    assert z.shape == (0,)


def test_robust_zscore_all_nan_returns_zeros_no_crash() -> None:
    z = mad_robust_zscore(np.array([np.nan, np.nan, np.nan]))
    assert np.all(z == 0.0)
    assert np.all(np.isfinite(z))


def test_robust_zscore_mixed_nan_and_finite_no_nan_leak() -> None:
    x = np.array([1.0, 2.0, np.nan, 3.0, 100.0])
    z = mad_robust_zscore(x)
    assert np.all(np.isfinite(z))
    assert z[2] == 0.0  # the NaN position itself carries no signal


def test_robust_zscore_falls_back_to_std_when_mad_zero_but_std_nonzero() -> None:
    # >=50% of values equal the median -> MAD is 0, but there is real spread.
    x = np.array([5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 100.0])
    z = mad_robust_zscore(x)
    assert np.all(np.isfinite(z))
    assert z[-1] > 0  # the 100.0 outlier still gets flagged via std fallback
