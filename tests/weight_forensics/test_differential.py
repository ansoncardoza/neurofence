from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from safetensors.numpy import save_file

from neurofence.exceptions import AcquisitionError
from neurofence.weight_forensics.differential import compare_models


def _make_model(dir_path: Path, tensors: dict[str, np.ndarray]) -> None:
    dir_path.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(dir_path / "model.safetensors"))
    (dir_path / "config.json").write_text(json.dumps({"architectures": ["Test"]}), encoding="utf-8")


def test_identical_models_zero_diff(tmp_path: Path) -> None:
    tensors = {"w": np.random.default_rng(0).standard_normal((10, 10)).astype(np.float32)}
    _make_model(tmp_path / "clean", tensors)
    _make_model(tmp_path / "suspect", tensors)

    result = compare_models(tmp_path / "clean", tmp_path / "suspect")

    assert result.status == "compared"
    assert result.compared_count == 1
    diff = result.tensor_diffs[0]
    assert diff.l2_diff == 0.0
    assert diff.percent_changed == 0.0


def test_localized_modification_detected(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    clean_w = rng.standard_normal((20, 20)).astype(np.float32)
    suspect_w = clean_w.copy()
    suspect_w[0, 0] += 50.0  # single-element localized modification

    _make_model(tmp_path / "clean", {"w": clean_w})
    _make_model(tmp_path / "suspect", {"w": suspect_w})

    result = compare_models(tmp_path / "clean", tmp_path / "suspect")

    diff = result.tensor_diffs[0]
    assert diff.l2_diff == pytest.approx(50.0, rel=1e-4)
    assert diff.percent_changed == pytest.approx(1 / 400, rel=1e-6)
    assert diff.max_abs_diff == pytest.approx(50.0, rel=1e-4)


def test_distributed_modification_detected(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    clean_w = rng.standard_normal((20, 20)).astype(np.float32)
    suspect_w = clean_w + rng.normal(scale=0.1, size=(20, 20)).astype(np.float32)

    _make_model(tmp_path / "clean", {"w": clean_w})
    _make_model(tmp_path / "suspect", {"w": suspect_w})

    result = compare_models(tmp_path / "clean", tmp_path / "suspect")

    diff = result.tensor_diffs[0]
    assert diff.percent_changed > 0.5  # most elements perturbed
    assert diff.l2_diff > 0


def test_no_overlapping_tensors_reference_incompatible(tmp_path: Path) -> None:
    _make_model(tmp_path / "clean", {"encoder.w": np.zeros((5, 5), dtype=np.float32)})
    _make_model(tmp_path / "suspect", {"decoder.w": np.zeros((5, 5), dtype=np.float32)})

    result = compare_models(tmp_path / "clean", tmp_path / "suspect")

    assert result.status == "reference_incompatible"
    assert "encoder.w" in result.missing_in_suspect
    assert "decoder.w" in result.missing_in_clean


def test_shape_mismatch_not_forced(tmp_path: Path) -> None:
    _make_model(tmp_path / "clean", {"w": np.zeros((10, 10), dtype=np.float32)})
    _make_model(tmp_path / "suspect", {"w": np.zeros((20, 5), dtype=np.float32)})

    result = compare_models(tmp_path / "clean", tmp_path / "suspect")

    assert result.status == "compared"
    assert result.compared_count == 0
    assert len(result.shape_mismatches) == 1
    assert result.shape_mismatches[0].clean_shape == [10, 10]
    assert result.shape_mismatches[0].suspect_shape == [20, 5]


def test_nan_tensor_skipped_not_crashed(tmp_path: Path) -> None:
    clean_w = np.ones((5, 5), dtype=np.float32)
    suspect_w = np.ones((5, 5), dtype=np.float32)
    suspect_w[0, 0] = np.nan

    _make_model(tmp_path / "clean", {"w": clean_w})
    _make_model(tmp_path / "suspect", {"w": suspect_w})

    result = compare_models(tmp_path / "clean", tmp_path / "suspect")

    assert result.status == "compared"
    assert result.tensor_diffs[0].status == "skipped_non_finite"
    assert result.tensor_diffs[0].l2_diff is None


def test_missing_clean_directory_raises(tmp_path: Path) -> None:
    _make_model(tmp_path / "suspect", {"w": np.zeros((3, 3), dtype=np.float32)})
    with pytest.raises(AcquisitionError):
        compare_models(tmp_path / "does_not_exist", tmp_path / "suspect")


def test_most_affected_layers_ranked_descending(tmp_path: Path) -> None:
    clean = {
        "small_change": np.zeros((5, 5), dtype=np.float32),
        "big_change": np.zeros((5, 5), dtype=np.float32),
    }
    suspect = {
        "small_change": clean["small_change"] + 0.01,
        "big_change": clean["big_change"] + 100.0,
    }
    _make_model(tmp_path / "clean", clean)
    _make_model(tmp_path / "suspect", suspect)

    result = compare_models(tmp_path / "clean", tmp_path / "suspect")

    assert result.most_affected_layers[0] == "big_change"


def test_zero_norm_clean_tensor_relative_diff_none(tmp_path: Path) -> None:
    clean = {"w": np.zeros((4, 4), dtype=np.float32)}
    suspect = {"w": np.ones((4, 4), dtype=np.float32)}
    _make_model(tmp_path / "clean", clean)
    _make_model(tmp_path / "suspect", suspect)

    result = compare_models(tmp_path / "clean", tmp_path / "suspect")

    diff = result.tensor_diffs[0]
    assert diff.relative_l2_diff is None  # ||clean|| == 0, relative diff undefined
    assert diff.l2_diff is not None and diff.l2_diff > 0
