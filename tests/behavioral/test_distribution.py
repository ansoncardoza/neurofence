from __future__ import annotations

import numpy as np
import pytest

from neurofence.behavioral.distribution import jensen_shannon_divergence, kl_divergence


def test_kl_identical_distributions_near_zero() -> None:
    p = np.array([0.2, 0.3, 0.5])
    assert kl_divergence(p, p) == pytest.approx(0.0, abs=1e-6)


def test_kl_different_distributions_positive() -> None:
    p = np.array([0.9, 0.1])
    q = np.array([0.1, 0.9])
    assert kl_divergence(p, q) > 0


def test_kl_is_not_symmetric() -> None:
    p = np.array([0.9, 0.1])
    q = np.array([0.5, 0.5])
    assert kl_divergence(p, q) != kl_divergence(q, p)


def test_kl_zero_probability_in_q_does_not_raise_or_return_inf() -> None:
    p = np.array([0.5, 0.5, 0.0])
    q = np.array([0.0, 0.5, 0.5])
    value = kl_divergence(p, q)
    assert np.isfinite(value)
    assert value > 0


def test_kl_unnormalized_input_is_normalized() -> None:
    p = np.array([2.0, 2.0])  # sums to 4, not 1
    q = np.array([1.0, 1.0])
    assert kl_divergence(p, q) == pytest.approx(0.0, abs=1e-6)


def test_kl_mismatched_shapes_raises() -> None:
    with pytest.raises(ValueError):
        kl_divergence(np.array([0.5, 0.5]), np.array([0.3, 0.3, 0.4]))


def test_kl_negative_values_raises() -> None:
    with pytest.raises(ValueError):
        kl_divergence(np.array([-0.1, 1.1]), np.array([0.5, 0.5]))


def test_kl_nan_raises() -> None:
    with pytest.raises(ValueError):
        kl_divergence(np.array([np.nan, 1.0]), np.array([0.5, 0.5]))


def test_kl_inf_raises() -> None:
    with pytest.raises(ValueError):
        kl_divergence(np.array([np.inf, 1.0]), np.array([0.5, 0.5]))


def test_kl_all_zero_raises() -> None:
    with pytest.raises(ValueError):
        kl_divergence(np.array([0.0, 0.0]), np.array([0.5, 0.5]))


def test_kl_empty_raises() -> None:
    with pytest.raises(ValueError):
        kl_divergence(np.array([]), np.array([]))


def test_kl_2d_input_raises() -> None:
    with pytest.raises(ValueError):
        kl_divergence(np.array([[0.5, 0.5]]), np.array([[0.5, 0.5]]))


def test_js_identical_distributions_zero() -> None:
    p = np.array([0.2, 0.3, 0.5])
    assert jensen_shannon_divergence(p, p) == pytest.approx(0.0, abs=1e-6)


def test_js_is_symmetric() -> None:
    p = np.array([0.9, 0.1])
    q = np.array([0.1, 0.9])
    forward = jensen_shannon_divergence(p, q)
    backward = jensen_shannon_divergence(q, p)
    assert forward == pytest.approx(backward, abs=1e-9)


def test_js_bounded_by_ln2() -> None:
    p = np.array([1.0, 0.0])
    q = np.array([0.0, 1.0])
    value = jensen_shannon_divergence(p, q)
    assert 0 <= value <= np.log(2) + 1e-6


def test_js_disjoint_support_finite() -> None:
    p = np.array([1.0, 0.0, 0.0])
    q = np.array([0.0, 0.0, 1.0])
    value = jensen_shannon_divergence(p, q)
    assert np.isfinite(value)
    assert value > 0


def test_js_mismatched_shapes_raises() -> None:
    with pytest.raises(ValueError):
        jensen_shannon_divergence(np.array([0.5, 0.5]), np.array([1.0]))
