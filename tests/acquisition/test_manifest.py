from __future__ import annotations

from pathlib import Path

import pytest

from neurofence.acquisition.manifest import build_manifest, verify_manifest
from neurofence.exceptions import AcquisitionError


def test_build_manifest_on_clean_model(tmp_model_dir: Path) -> None:
    manifest = build_manifest(tmp_model_dir)

    assert manifest.file_count == 3
    assert manifest.total_size_bytes > 0
    paths = {f.path for f in manifest.files}
    assert paths == {"model.safetensors", "config.json", "tokenizer_config.json"}


def test_manifest_files_sorted_deterministically(tmp_model_dir: Path) -> None:
    m1 = build_manifest(tmp_model_dir)
    m2 = build_manifest(tmp_model_dir)

    assert [f.path for f in m1.files] == [f.path for f in m2.files]
    assert [f.sha256 for f in m1.files] == [f.sha256 for f in m2.files]


def test_missing_directory_raises(tmp_path: Path) -> None:
    with pytest.raises(AcquisitionError):
        build_manifest(tmp_path / "does_not_exist")


def test_empty_directory_raises(empty_model_dir: Path) -> None:
    with pytest.raises(AcquisitionError):
        build_manifest(empty_model_dir)


def test_path_given_is_a_file_not_directory(tmp_path: Path) -> None:
    f = tmp_path / "notadir.txt"
    f.write_text("hi")
    with pytest.raises(AcquisitionError):
        build_manifest(f)


def test_verify_manifest_matches_unmodified_model(tmp_model_dir: Path) -> None:
    manifest = build_manifest(tmp_model_dir)
    result = verify_manifest(manifest, tmp_model_dir)

    assert result.matched is True
    assert result.discrepancies == []


def test_verify_manifest_detects_modified_weight_file(tmp_model_dir: Path) -> None:
    manifest = build_manifest(tmp_model_dir)

    weight_file = tmp_model_dir / "model.safetensors"
    data = bytearray(weight_file.read_bytes())
    data[-1] ^= 0xFF  # flip last byte
    weight_file.write_bytes(bytes(data))

    result = verify_manifest(manifest, tmp_model_dir)

    assert result.matched is False
    kinds = {d.kind for d in result.discrepancies}
    assert "hash_mismatch" in kinds


def test_verify_manifest_detects_modified_config(tmp_model_dir: Path) -> None:
    manifest = build_manifest(tmp_model_dir)

    config_file = tmp_model_dir / "config.json"
    config_file.write_text('{"architectures": ["Evil"]}', encoding="utf-8")

    result = verify_manifest(manifest, tmp_model_dir)

    assert result.matched is False
    assert any(d.path == "config.json" for d in result.discrepancies)


def test_verify_manifest_detects_missing_file(tmp_model_dir: Path) -> None:
    manifest = build_manifest(tmp_model_dir)

    (tmp_model_dir / "tokenizer_config.json").unlink()

    result = verify_manifest(manifest, tmp_model_dir)

    assert result.matched is False
    assert any(d.kind == "missing_file" for d in result.discrepancies)


def test_verify_manifest_detects_extra_file(tmp_model_dir: Path) -> None:
    manifest = build_manifest(tmp_model_dir)

    (tmp_model_dir / "unexpected_extra.bin").write_bytes(b"surprise")

    result = verify_manifest(manifest, tmp_model_dir)

    assert result.matched is False
    assert any(d.kind == "extra_file" for d in result.discrepancies)


def test_duplicate_content_files_get_independent_records(tmp_model_dir: Path) -> None:
    # Two files with identical content should still both appear, with equal
    # hashes but distinct paths -- the manifest must not silently dedupe.
    content = '{"architectures": ["Evil"]}'
    (tmp_model_dir / "duplicate.json").write_text(content, encoding="utf-8")
    (tmp_model_dir / "duplicate_copy.json").write_text(content, encoding="utf-8")

    manifest = build_manifest(tmp_model_dir)

    dup_records = [f for f in manifest.files if f.path.startswith("duplicate")]
    assert len(dup_records) == 2
    assert dup_records[0].sha256 == dup_records[1].sha256
    assert dup_records[0].path != dup_records[1].path


def test_multiple_shards_all_included(tmp_model_dir: Path) -> None:
    import numpy as np
    from safetensors.numpy import save_file

    shard2 = tmp_model_dir / "model-00002-of-00002.safetensors"
    save_file({"layer1.weight": np.zeros((4, 4), dtype=np.float32)}, str(shard2))
    (tmp_model_dir / "model.safetensors").rename(tmp_model_dir / "model-00001-of-00002.safetensors")

    manifest = build_manifest(tmp_model_dir)
    shard_paths = [f.path for f in manifest.files if f.path.endswith(".safetensors")]
    assert len(shard_paths) == 2


def test_nested_subdirectories_are_walked(tmp_path: Path) -> None:
    model_dir = tmp_path / "nested_model"
    model_dir.mkdir()
    (model_dir / "sub").mkdir()
    (model_dir / "sub" / "file.txt").write_text("data")
    (model_dir / "top.txt").write_text("data")

    manifest = build_manifest(model_dir)

    paths = {f.path for f in manifest.files}
    assert paths == {"sub/file.txt", "top.txt"}
