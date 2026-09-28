from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from safetensors.numpy import save_file

from neurofence.acquisition.metadata import extract_model_metadata
from neurofence.exceptions import AcquisitionError


def test_extracts_architecture_and_param_count(tmp_model_dir: Path) -> None:
    meta = extract_model_metadata(tmp_model_dir)

    assert meta.architecture == "TinyTestModel"
    assert meta.model_type == "tiny_test"
    assert meta.layer_count == 1
    assert meta.weight_format == "safetensors"
    # layer0.weight (16*16) + layer0.bias (16) = 272
    assert meta.parameter_count == 272
    assert meta.tensor_count == 2
    assert meta.shard_count == 1
    assert meta.warnings == []


def test_missing_config_json_reports_warning_not_crash(tmp_path: Path) -> None:
    model_dir = tmp_path / "no_config"
    model_dir.mkdir()
    save_file({"w": np.zeros((2, 2), dtype=np.float32)}, str(model_dir / "model.safetensors"))

    meta = extract_model_metadata(model_dir)

    assert meta.architecture is None
    assert meta.parameter_count == 4
    assert any("config.json" in w for w in meta.warnings)


def test_malformed_config_json_does_not_crash(tmp_path: Path) -> None:
    model_dir = tmp_path / "bad_config"
    model_dir.mkdir()
    (model_dir / "config.json").write_text("{not valid json", encoding="utf-8")
    save_file({"w": np.zeros((2, 2), dtype=np.float32)}, str(model_dir / "model.safetensors"))

    meta = extract_model_metadata(model_dir)

    assert meta.config is None
    assert any("not valid JSON" in w for w in meta.warnings)


def test_pickle_only_model_does_not_unpickle(tmp_path: Path) -> None:
    model_dir = tmp_path / "pickle_model"
    model_dir.mkdir()
    (model_dir / "pytorch_model.bin").write_bytes(b"not really a pickle, just bytes")

    meta = extract_model_metadata(model_dir)

    assert meta.weight_format == "pickle_weights"
    assert meta.parameter_count is None
    assert any("unpickle" in w for w in meta.warnings)


def test_mixed_safetensors_and_pickle_flags_mixed(tmp_path: Path) -> None:
    model_dir = tmp_path / "mixed_model"
    model_dir.mkdir()
    save_file({"w": np.zeros((2, 2), dtype=np.float32)}, str(model_dir / "model.safetensors"))
    (model_dir / "legacy.bin").write_bytes(b"legacy pickle bytes")

    meta = extract_model_metadata(model_dir)

    assert meta.weight_format == "mixed"
    assert meta.parameter_count == 4  # only from safetensors
    assert any("pickle-format" in w for w in meta.warnings)


def test_no_weight_files_at_all(tmp_path: Path) -> None:
    model_dir = tmp_path / "no_weights"
    model_dir.mkdir()
    (model_dir / "readme.txt").write_text("hi")

    meta = extract_model_metadata(model_dir)

    assert meta.weight_format == "none"
    assert meta.parameter_count is None


def test_missing_directory_raises(tmp_path: Path) -> None:
    with pytest.raises(AcquisitionError):
        extract_model_metadata(tmp_path / "nope")


def test_multi_shard_param_count_sums_across_shards(tmp_path: Path) -> None:
    model_dir = tmp_path / "sharded"
    model_dir.mkdir()
    shard1 = model_dir / "model-00001-of-00002.safetensors"
    shard2 = model_dir / "model-00002-of-00002.safetensors"
    save_file({"a": np.zeros((3, 3), dtype=np.float32)}, str(shard1))
    save_file({"b": np.zeros((5,), dtype=np.float32)}, str(shard2))

    meta = extract_model_metadata(model_dir)

    assert meta.parameter_count == 9 + 5
    assert meta.shard_count == 2


def test_zero_dim_tensor_counts_as_one_scalar_param(tmp_path: Path) -> None:
    model_dir = tmp_path / "scalar_model"
    model_dir.mkdir()
    save_file({"scale": np.array(3.0, dtype=np.float32)}, str(model_dir / "model.safetensors"))

    meta = extract_model_metadata(model_dir)

    assert meta.parameter_count == 1
