from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from safetensors.numpy import save_file

from neurofence.weight_forensics.pipeline import analyze_model_weights


def _make_model(dir_path: Path, tensors: dict[str, np.ndarray]) -> None:
    dir_path.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(dir_path / "model.safetensors"))
    (dir_path / "config.json").write_text(json.dumps({"architectures": ["Test"]}), encoding="utf-8")


def test_end_to_end_clean_model(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    tensors = {
        f"layer{i}.weight": rng.standard_normal((16, 16)).astype(np.float32) for i in range(12)
    }
    model_dir = tmp_path / "model"
    _make_model(model_dir, tensors)

    result = analyze_model_weights(model_dir)

    assert result.tensor_count == 12
    assert len(result.tensors) == 12
    assert all(t.statistics.status == "ok" for t in result.tensors)
    assert all(
        t.spectral is not None and t.spectral.computation_status == "computed"
        for t in result.tensors
    )
    assert result.anomaly_detection.status == "ml_based"


def test_end_to_end_poisoned_model_flags_outlier_layer(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    tensors = {
        f"layer{i}.weight": rng.standard_normal((16, 16)).astype(np.float32) for i in range(15)
    }
    tensors["layer_poisoned.weight"] = rng.standard_normal((16, 16)).astype(np.float32) * 50 + 100
    model_dir = tmp_path / "model"
    _make_model(model_dir, tensors)

    result = analyze_model_weights(model_dir)

    by_name = {t.name: t for t in result.tensors}
    anomaly_by_name = {r.layer_name: r for r in result.anomaly_detection.results}
    assert anomaly_by_name["layer_poisoned.weight"].is_outlier is True
    assert by_name["layer_poisoned.weight"].statistics.mean > 50


def test_max_layers_sampled_caps_spectral_analysis(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    tensors = {
        f"layer{i}.weight": rng.standard_normal((8, 8)).astype(np.float32) for i in range(10)
    }
    model_dir = tmp_path / "model"
    _make_model(model_dir, tensors)

    result = analyze_model_weights(model_dir, max_layers_sampled=3)

    spectral_computed = [t for t in result.tensors if t.spectral is not None]
    assert len(spectral_computed) == 3
    # statistics still run on every tensor regardless of the spectral cap
    assert all(t.statistics.status == "ok" for t in result.tensors)


def test_1d_tensors_skip_spectral_but_still_get_statistics(tmp_path: Path) -> None:
    tensors = {"bias": np.zeros(64, dtype=np.float32)}
    model_dir = tmp_path / "model"
    _make_model(model_dir, tensors)

    result = analyze_model_weights(model_dir)

    assert result.tensors[0].statistics.status == "ok"
    assert result.tensors[0].spectral is None
