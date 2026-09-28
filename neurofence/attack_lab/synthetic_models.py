"""Generates synthetic "models" (small dicts of named weight tensors) and
applies documented, deterministic perturbations to them, with a
PerturbationRecord describing exactly what was changed.

These are not real trained models -- they exist purely to give the
weight-forensics detectors known-clean and known-poisoned ground truth to
be measured against (Milestone 7 of the project spec). Every perturbation
kind is a distinct, named attack pattern so evaluation can report
per-attack-type detection performance, not just an aggregate number.
"""

from __future__ import annotations

import numpy as np
from pydantic import BaseModel

# Layer shapes loosely mimicking a tiny transformer block: a few
# projection matrices and bias vectors. Deliberately >= 8 tensors
# (weight_forensics.anomaly.DEFAULT_MIN_SAMPLES_FOR_ML) so a clean
# synthetic model routes through the ML-based (Isolation Forest/LOF/
# Mahalanobis) detection path rather than the small-sample statistical
# fallback -- the fallback's MAD-based z-score is far more sensitive to
# ordinary sampling noise across only a handful of layers, which produced
# spurious false positives on clean models here during benchmarking.
DEFAULT_LAYER_SHAPES: dict[str, tuple[int, ...]] = {
    "attn.q_proj.weight": (32, 32),
    "attn.k_proj.weight": (32, 32),
    "attn.v_proj.weight": (32, 32),
    "attn.out_proj.weight": (32, 32),
    "mlp.fc1.weight": (32, 64),
    "mlp.fc2.weight": (64, 32),
    "attn.out_proj.bias": (32,),
    "mlp.bias": (32,),
}

PERTURBATION_KINDS = (
    "localized",
    "distributed",
    "layer_level",
    "neuron_level",
    "low_magnitude",
)


class PerturbationRecord(BaseModel):
    kind: str
    layer_name: str
    magnitude: float
    fraction_elements_affected: float
    detail: str


def generate_clean_tensors(
    shapes: dict[str, tuple[int, ...]] | None = None,
    seed: int = 0,
    scale: float = 0.05,
) -> dict[str, np.ndarray]:
    """A deterministic, plausible-looking set of small weight tensors.
    `scale` mimics typical small-init transformer weight magnitudes.
    """
    shapes = shapes or DEFAULT_LAYER_SHAPES
    rng = np.random.default_rng(seed)
    return {
        name: (rng.standard_normal(shape) * scale).astype(np.float32)
        for name, shape in shapes.items()
    }


def _pick_layer(tensors: dict[str, np.ndarray], rng: np.random.Generator) -> str:
    # 1D (bias) tensors are excluded from perturbations that assume a 2D
    # weight matrix (layer_level and neuron_level); localized/distributed/
    # low_magnitude work on any shape.
    return str(rng.choice(list(tensors)))


def apply_localized_perturbation(
    tensors: dict[str, np.ndarray],
    rng: np.random.Generator,
    layer_name: str | None = None,
    magnitude: float = 50.0,
) -> tuple[dict[str, np.ndarray], PerturbationRecord]:
    """Flip a single element by a large, fixed magnitude -- the classic
    "single weight tampering" case.
    """
    layer_name = layer_name or _pick_layer(tensors, rng)
    poisoned = {name: arr.copy() for name, arr in tensors.items()}
    arr = poisoned[layer_name]
    flat_index = int(rng.integers(0, arr.size))
    idx = np.unravel_index(flat_index, arr.shape)
    arr[idx] += magnitude

    record = PerturbationRecord(
        kind="localized",
        layer_name=layer_name,
        magnitude=magnitude,
        fraction_elements_affected=1.0 / arr.size,
        detail=f"Single element at index {idx} shifted by {magnitude}.",
    )
    return poisoned, record


def apply_distributed_perturbation(
    tensors: dict[str, np.ndarray],
    rng: np.random.Generator,
    layer_name: str | None = None,
    fraction: float = 0.3,
    magnitude: float = 0.5,
) -> tuple[dict[str, np.ndarray], PerturbationRecord]:
    """Add noise of a given magnitude to a fraction of elements, spread
    across the tensor -- simulates a broader, less concentrated tamper.
    """
    layer_name = layer_name or _pick_layer(tensors, rng)
    poisoned = {name: arr.copy() for name, arr in tensors.items()}
    arr = poisoned[layer_name]
    n_affected = max(1, int(arr.size * fraction))
    flat = arr.reshape(-1)
    indices = rng.choice(arr.size, size=n_affected, replace=False)
    flat[indices] += rng.normal(scale=magnitude, size=n_affected)

    record = PerturbationRecord(
        kind="distributed",
        layer_name=layer_name,
        magnitude=magnitude,
        fraction_elements_affected=n_affected / arr.size,
        detail=f"{n_affected}/{arr.size} elements perturbed with noise scale {magnitude}.",
    )
    return poisoned, record


def apply_layer_level_perturbation(
    tensors: dict[str, np.ndarray],
    rng: np.random.Generator,
    layer_name: str | None = None,
    scale_factor: float = 3.0,
) -> tuple[dict[str, np.ndarray], PerturbationRecord]:
    """Scale an entire layer's weights -- simulates a layer-wide tamper
    (e.g. a rescaling attack) rather than a localized change.
    """
    eligible = [name for name, arr in tensors.items() if arr.ndim == 2] or list(tensors)
    layer_name = layer_name or str(rng.choice(eligible))
    poisoned = {name: arr.copy() for name, arr in tensors.items()}
    poisoned[layer_name] = poisoned[layer_name] * scale_factor

    record = PerturbationRecord(
        kind="layer_level",
        layer_name=layer_name,
        magnitude=scale_factor,
        fraction_elements_affected=1.0,
        detail=f"Entire layer scaled by factor {scale_factor}.",
    )
    return poisoned, record


def apply_neuron_level_perturbation(
    tensors: dict[str, np.ndarray],
    rng: np.random.Generator,
    layer_name: str | None = None,
    magnitude: float = 10.0,
) -> tuple[dict[str, np.ndarray], PerturbationRecord]:
    """Perturb a single row ("neuron") of a 2D weight matrix -- simulates
    a targeted single-neuron backdoor implant.
    """
    eligible = [name for name, arr in tensors.items() if arr.ndim == 2]
    if not eligible:
        raise ValueError("neuron_level perturbation requires at least one 2D weight tensor.")
    layer_name = layer_name or str(rng.choice(eligible))
    poisoned = {name: arr.copy() for name, arr in tensors.items()}
    arr = poisoned[layer_name]
    row = int(rng.integers(0, arr.shape[0]))
    arr[row, :] += magnitude

    record = PerturbationRecord(
        kind="neuron_level",
        layer_name=layer_name,
        magnitude=magnitude,
        fraction_elements_affected=arr.shape[1] / arr.size,
        detail=f"Row {row} (one neuron) shifted by {magnitude}.",
    )
    return poisoned, record


def apply_low_magnitude_perturbation(
    tensors: dict[str, np.ndarray],
    rng: np.random.Generator,
    layer_name: str | None = None,
    fraction: float = 0.5,
    magnitude: float = 0.01,
) -> tuple[dict[str, np.ndarray], PerturbationRecord]:
    """A distributed perturbation at a magnitude comparable to normal
    weight noise -- the deliberately hard-to-detect case. Detectors are
    *expected* to miss some fraction of these; that expectation is exactly
    what evaluation.metrics measures (false negative rate), not a bug to
    "fix" by tuning thresholds until this case always triggers.
    """
    poisoned, record = apply_distributed_perturbation(
        tensors, rng, layer_name=layer_name, fraction=fraction, magnitude=magnitude
    )
    return poisoned, record.model_copy(update={"kind": "low_magnitude"})


_PERTURBATION_FUNCS = {
    "localized": apply_localized_perturbation,
    "distributed": apply_distributed_perturbation,
    "layer_level": apply_layer_level_perturbation,
    "neuron_level": apply_neuron_level_perturbation,
    "low_magnitude": apply_low_magnitude_perturbation,
}


def apply_perturbation(
    kind: str,
    tensors: dict[str, np.ndarray],
    rng: np.random.Generator,
    **kwargs: object,
) -> tuple[dict[str, np.ndarray], PerturbationRecord]:
    if kind not in _PERTURBATION_FUNCS:
        raise ValueError(
            f"Unknown perturbation kind {kind!r}; expected one of {PERTURBATION_KINDS}."
        )
    return _PERTURBATION_FUNCS[kind](tensors, rng, **kwargs)  # type: ignore[operator]
