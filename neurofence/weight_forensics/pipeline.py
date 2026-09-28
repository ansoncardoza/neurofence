"""Ties together per-tensor statistics, spectral analysis, and layer-level
anomaly detection into a single weight-forensics pass over a model
directory. This is the entry point the CLI/API call; the individual
modules (statistics, spectral, anomaly, differential) remain independently
testable and usable on their own.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from pydantic import BaseModel

from neurofence.acquisition.formats import FileFormat, classify
from neurofence.weight_forensics.anomaly import AnomalyDetectionSummary, detect_layer_anomalies
from neurofence.weight_forensics.spectral import SpectralResult, spectral_analysis
from neurofence.weight_forensics.statistics import TensorStatistics, compute_tensor_statistics

# Order matters: this defines the feature vector layout fed to layer anomaly
# detection. None/degenerate values become 0.0 -- documented, not silent,
# since detect_layer_anomalies operates purely on numbers.
#
# l1_norm/l2_norm are NOT used directly: raw norms scale with tensor size
# (a 2048-element layer has a far larger L2 norm than a 32-element bias
# vector even when both are drawn from the identical distribution), which
# swamped every other signal and flagged every differently-shaped layer in
# a model as "anomalous" purely for being a different size -- caught via
# the attack_lab benchmark producing 0% true-negative rate on an
# all-clean synthetic dataset containing mixed 2D weights and a 1D bias.
# Per-element-normalized norms (mean absolute value, RMS) are scale-
# invariant and comparable across differently-shaped layers instead.
FEATURE_NAMES = (
    "mean",
    "std",
    "skewness",
    "kurtosis",
    "l1_norm_per_element",
    "l2_norm_rms",
    "sparsity",
)


class TensorForensicsRecord(BaseModel):
    name: str
    statistics: TensorStatistics
    spectral: SpectralResult | None = None


class WeightForensicsResult(BaseModel):
    tensor_count: int
    tensors: list[TensorForensicsRecord]
    anomaly_detection: AnomalyDetectionSummary


def _safe_float(v: float | None) -> float:
    # `v or 0.0` would leave a NaN in place (NaN is truthy in Python), so
    # check explicitly rather than relying on truthiness.
    return 0.0 if v is None or not np.isfinite(v) else float(v)


def build_layer_feature_vector(stats: TensorStatistics) -> np.ndarray:
    n = stats.finite_count
    l1_per_element = (stats.l1_norm / n) if (stats.l1_norm is not None and n > 0) else None
    l2_rms = (stats.l2_norm / np.sqrt(n)) if (stats.l2_norm is not None and n > 0) else None

    by_name = {
        "mean": stats.mean,
        "std": stats.std,
        "skewness": stats.skewness,
        "kurtosis": stats.kurtosis,
        "l1_norm_per_element": l1_per_element,
        "l2_norm_rms": l2_rms,
        "sparsity": stats.sparsity,
    }
    return np.array([_safe_float(by_name[name]) for name in FEATURE_NAMES], dtype=np.float64)


def _load_safetensors_arrays(model_dir: str | Path) -> dict[str, np.ndarray]:
    from safetensors import safe_open

    root = Path(model_dir)
    shard_paths = sorted(
        p for p in root.rglob("*") if p.is_file() and classify(p) == FileFormat.SAFETENSORS
    )
    tensors: dict[str, np.ndarray] = {}
    for path in shard_paths:
        with safe_open(str(path), framework="numpy") as f:
            for key in f.keys():
                tensors[key] = f.get_tensor(key)
    return tensors


def analyze_model_weights(
    model_dir: str | Path,
    max_layers_sampled: int | None = None,
    run_spectral: bool = True,
) -> WeightForensicsResult:
    """Run weight forensics over every safetensors tensor in `model_dir`.

    `max_layers_sampled` caps how many tensors get the (potentially
    expensive) spectral analysis, in sorted-name order for determinism;
    statistics and anomaly detection always run over every tensor.
    """
    tensors = _load_safetensors_arrays(model_dir)
    names = sorted(tensors)

    records: list[TensorForensicsRecord] = []
    features: dict[str, np.ndarray] = {}

    spectral_budget = len(names) if max_layers_sampled is None else max_layers_sampled

    for i, name in enumerate(names):
        arr = tensors[name]
        stats = compute_tensor_statistics(arr)
        features[name] = build_layer_feature_vector(stats)

        spectral: SpectralResult | None = None
        if run_spectral and arr.ndim == 2 and i < spectral_budget:
            spectral = spectral_analysis(arr)

        records.append(TensorForensicsRecord(name=name, statistics=stats, spectral=spectral))

    anomaly_summary = detect_layer_anomalies(features)

    return WeightForensicsResult(
        tensor_count=len(names),
        tensors=records,
        anomaly_detection=anomaly_summary,
    )
