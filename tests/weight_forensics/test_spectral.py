from __future__ import annotations

import numpy as np

from neurofence.weight_forensics.spectral import spectral_analysis


def test_full_svd_on_small_matrix() -> None:
    rng = np.random.default_rng(0)
    m = rng.standard_normal((20, 20))
    result = spectral_analysis(m)

    assert result.algorithm == "full_svd"
    assert result.computation_status == "computed"
    assert result.spectral_norm is not None and result.spectral_norm > 0
    assert result.condition_number is not None
    assert len(result.top_singular_values) > 0
    assert result.effective_rank is not None and result.effective_rank > 0


def test_non_2d_tensor_skipped() -> None:
    result = spectral_analysis(np.zeros((3, 3, 3)))
    assert result.computation_status == "skipped"
    assert "2D" in result.skip_reason


def test_1d_tensor_skipped() -> None:
    result = spectral_analysis(np.zeros((10,)))
    assert result.computation_status == "skipped"


def test_nan_matrix_skipped() -> None:
    m = np.full((5, 5), np.nan)
    result = spectral_analysis(m)
    assert result.computation_status == "skipped"
    assert "NaN" in result.skip_reason or "Inf" in result.skip_reason


def test_inf_matrix_skipped() -> None:
    m = np.full((5, 5), np.inf)
    result = spectral_analysis(m)
    assert result.computation_status == "skipped"


def test_zero_matrix() -> None:
    m = np.zeros((10, 10))
    result = spectral_analysis(m)

    assert result.computation_status == "computed"
    assert result.spectral_norm == 0.0
    assert result.effective_rank == 0.0
    assert result.condition_number is None
    assert result.condition_number_note is not None


def test_zero_length_dimension_skipped() -> None:
    result = spectral_analysis(np.zeros((0, 5)))
    assert result.computation_status == "skipped"


def test_identity_matrix_full_rank() -> None:
    m = np.eye(10)
    result = spectral_analysis(m)

    assert result.computation_status == "computed"
    assert result.condition_number == 1.0  # perfectly conditioned
    assert abs(result.effective_rank - 10.0) < 1e-6  # entropy is maximal: all singular values equal


def test_rank_deficient_matrix_condition_number_undefined() -> None:
    # rank-1 matrix: all but one singular value are 0
    v = np.array([[1.0], [2.0], [3.0]])
    m = v @ v.T  # 3x3, rank 1

    result = spectral_analysis(m)

    assert result.computation_status == "computed"
    assert result.condition_number is None
    assert "rank-deficient" in result.condition_number_note


def test_tiny_1x1_matrix() -> None:
    result = spectral_analysis(np.array([[5.0]]))
    assert result.computation_status == "computed"
    assert result.spectral_norm == 5.0


def test_large_matrix_uses_truncated_svd() -> None:
    rng = np.random.default_rng(0)
    # exceeds a small forced threshold to exercise the truncated path without
    # actually needing a huge (slow) matrix in the test suite.
    m = rng.standard_normal((300, 300))
    result = spectral_analysis(m, max_full_svd_elements=1000, truncated_k=5)

    assert result.algorithm == "truncated_svd_topk"
    assert result.computation_status == "computed"
    assert len(result.top_singular_values) == 5
    assert result.condition_number is None
    assert result.condition_number_note is not None
    assert result.spectral_norm is not None and result.spectral_norm > 0


def test_truncated_svd_singular_values_descending() -> None:
    rng = np.random.default_rng(1)
    m = rng.standard_normal((200, 200))
    result = spectral_analysis(m, max_full_svd_elements=100, truncated_k=8)

    values = result.top_singular_values
    assert values == sorted(values, reverse=True)


def test_non_square_matrix() -> None:
    rng = np.random.default_rng(0)
    m = rng.standard_normal((50, 10))
    result = spectral_analysis(m)
    assert result.computation_status == "computed"
    assert len(result.top_singular_values) <= 10


def test_low_rank_perturbation_detectable_via_spectral_norm() -> None:
    """Ground-truth check: injecting a low-rank spike should increase the
    top singular value relative to the clean baseline -- this is the exact
    signal spectral forensics is meant to catch (e.g. a rank-1 backdoor
    patch added to a weight matrix)."""
    rng = np.random.default_rng(0)
    clean = rng.standard_normal((100, 100)) * 0.01

    spike = np.outer(rng.standard_normal(100), rng.standard_normal(100)) * 5.0
    poisoned = clean + spike

    clean_result = spectral_analysis(clean)
    poisoned_result = spectral_analysis(poisoned)

    assert poisoned_result.spectral_norm > clean_result.spectral_norm
