"""Safe model metadata extraction.

Reads config.json and safetensors headers only -- both are structured,
non-executable formats. Never unpickles `.bin`/`.pt` checkpoints: if a model
ships only pickle-format weights, parameter counting is reported as
unavailable rather than performed unsafely.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field
from safetensors import safe_open

from neurofence.acquisition.formats import FileFormat, classify
from neurofence.exceptions import AcquisitionError
from neurofence.logging_setup import get_logger

logger = get_logger(__name__)


class ModelMetadata(BaseModel):
    architecture: str | None = None
    model_type: str | None = None
    parameter_count: int | None = None
    tensor_count: int | None = None
    layer_count: int | None = None
    weight_format: str  # "safetensors" | "pickle_weights" | "mixed" | "none"
    shard_count: int
    config: dict[str, Any] | None = None
    warnings: list[str] = Field(default_factory=list)


def _load_config_json(model_dir: Path) -> tuple[dict[str, Any] | None, list[str]]:
    config_path = model_dir / "config.json"
    if not config_path.exists():
        return None, ["No config.json found; architecture metadata unavailable."]

    try:
        raw = config_path.read_text(encoding="utf-8")
    except OSError as e:
        return None, [f"Could not read config.json: {e}"]

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as e:
        return None, [f"config.json is not valid JSON: {e}"]

    if not isinstance(parsed, dict):
        return None, ["config.json top-level value is not an object."]

    return parsed, []


def _count_safetensors_params(paths: list[Path]) -> tuple[int, int, list[str]]:
    """Return (parameter_count, tensor_count, warnings) by reading only
    safetensors headers -- tensor data is memory-mapped, not materialized.
    """
    total_params = 0
    total_tensors = 0
    warnings: list[str] = []

    for path in paths:
        try:
            with safe_open(str(path), framework="numpy") as f:
                for key in f.keys():
                    try:
                        shape = f.get_slice(key).get_shape()
                    except Exception as e:  # malformed tensor entry
                        warnings.append(
                            f"{path.name}: could not read shape for tensor '{key}': {e}"
                        )
                        continue
                    total_params += math.prod(shape) if shape else 1
                    total_tensors += 1
        except Exception as e:
            warnings.append(f"Could not parse safetensors header for {path.name}: {e}")

    return total_params, total_tensors, warnings


def extract_model_metadata(model_dir: str | Path) -> ModelMetadata:
    root = Path(model_dir)
    if not root.exists() or not root.is_dir():
        raise AcquisitionError(f"Model directory does not exist: {root}")

    config, config_warnings = _load_config_json(root)
    warnings = list(config_warnings)

    architecture = None
    model_type = None
    layer_count = None
    if config:
        archs = config.get("architectures")
        if isinstance(archs, list) and archs:
            architecture = str(archs[0])
        model_type = config.get("model_type")
        for key in ("num_hidden_layers", "n_layer", "num_layers"):
            if key in config and isinstance(config[key], int):
                layer_count = config[key]
                break

    safetensor_files = sorted(
        p for p in root.rglob("*") if p.is_file() and classify(p) == FileFormat.SAFETENSORS
    )
    pickle_files = [
        p for p in root.rglob("*") if p.is_file() and classify(p) == FileFormat.PICKLE_WEIGHTS
    ]

    parameter_count: int | None = None
    tensor_count: int | None = None

    if safetensor_files:
        parameter_count, tensor_count, st_warnings = _count_safetensors_params(safetensor_files)
        warnings.extend(st_warnings)
        weight_format = "mixed" if pickle_files else "safetensors"
        if pickle_files:
            warnings.append(
                f"{len(pickle_files)} pickle-format weight file(s) present alongside safetensors; "
                "these are hashed but not parsed (unpickling untrusted files is unsafe)."
            )
    elif pickle_files:
        weight_format = "pickle_weights"
        warnings.append(
            "Only pickle-format (.bin/.pt/.pth) weight files found; parameter count and tensor "
            "statistics are unavailable because NeuroFence will not unpickle untrusted files. "
            "Convert to safetensors for full weight forensics."
        )
    else:
        weight_format = "none"
        warnings.append("No recognized weight files (safetensors or pickle) found.")

    return ModelMetadata(
        architecture=architecture,
        model_type=model_type,
        parameter_count=parameter_count,
        tensor_count=tensor_count,
        layer_count=layer_count,
        weight_format=weight_format,
        shard_count=len(safetensor_files),
        config=config,
        warnings=warnings,
    )
