from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from safetensors.numpy import save_file


@pytest.fixture
def tmp_model_dir(tmp_path: Path) -> Path:
    """A minimal, valid safetensors model directory."""
    model_dir = tmp_path / "clean_model"
    model_dir.mkdir()

    tensors = {
        "layer0.weight": np.random.default_rng(0).standard_normal((16, 16)).astype(np.float32),
        "layer0.bias": np.zeros(16, dtype=np.float32),
    }
    save_file(tensors, str(model_dir / "model.safetensors"))

    config = {
        "architectures": ["TinyTestModel"],
        "model_type": "tiny_test",
        "num_hidden_layers": 1,
    }
    (model_dir / "config.json").write_text(json.dumps(config), encoding="utf-8")
    (model_dir / "tokenizer_config.json").write_text("{}", encoding="utf-8")

    return model_dir


@pytest.fixture
def empty_model_dir(tmp_path: Path) -> Path:
    d = tmp_path / "empty_model"
    d.mkdir()
    return d
