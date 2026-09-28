"""Differential weight analysis: compare a suspect model against a trusted
reference model, tensor by tensor.

Only safetensors-format weights are compared (see neurofence.acquisition
for why pickle-format weights are never loaded). If the two models share no
tensors with matching names, or none with matching shapes, this is reported
as REFERENCE_INCOMPATIBLE rather than forced into a misleading comparison.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from pydantic import BaseModel
from safetensors import safe_open

from neurofence.acquisition.formats import FileFormat, classify
from neurofence.exceptions import AcquisitionError


class TensorDiff(BaseModel):
    name: str
    shape: list[int]
    status: str  # "compared" | "skipped_non_finite"
    l2_diff: float | None = None  # == Frobenius norm of (suspect - clean)
    max_abs_diff: float | None = None
    mean_abs_diff: float | None = None
    percent_changed: float | None = None  # fraction of elements with |diff| > tolerance
    relative_l2_diff: float | None = None  # l2_diff / ||clean||_2, None if clean tensor is all-zero
    note: str | None = None


class ShapeMismatch(BaseModel):
    name: str
    clean_shape: list[int]
    suspect_shape: list[int]


class DifferentialResult(BaseModel):
    status: str  # "compared" | "reference_incompatible"
    reason: str | None = None
    clean_tensor_count: int = 0
    suspect_tensor_count: int = 0
    compared_count: int = 0
    shape_mismatches: list[ShapeMismatch] = []
    missing_in_suspect: list[str] = []
    missing_in_clean: list[str] = []
    tensor_diffs: list[TensorDiff] = []
    most_affected_layers: list[str] = []  # top tensor_diffs by l2_diff, descending


def _load_safetensors_tensors(model_dir: str | Path) -> dict[str, np.ndarray]:
    root = Path(model_dir)
    if not root.exists() or not root.is_dir():
        raise AcquisitionError(f"Model directory does not exist: {root}")

    shard_paths = sorted(
        p for p in root.rglob("*") if p.is_file() and classify(p) == FileFormat.SAFETENSORS
    )

    tensors: dict[str, np.ndarray] = {}
    for path in shard_paths:
        with safe_open(str(path), framework="numpy") as f:
            for key in f.keys():
                tensors[key] = f.get_tensor(key)
    return tensors


def compare_models(
    clean_model_dir: str | Path,
    suspect_model_dir: str | Path,
    diff_tolerance: float = 1e-6,
    top_n_affected: int = 10,
) -> DifferentialResult:
    clean = _load_safetensors_tensors(clean_model_dir)
    suspect = _load_safetensors_tensors(suspect_model_dir)

    clean_names = set(clean)
    suspect_names = set(suspect)
    common = sorted(clean_names & suspect_names)

    if not common:
        return DifferentialResult(
            status="reference_incompatible",
            reason=(
                "No tensor names are shared between the reference and suspect models "
                "(different architectures or naming conventions). Differential analysis "
                "cannot proceed -- forcing a comparison would produce meaningless results."
            ),
            clean_tensor_count=len(clean_names),
            suspect_tensor_count=len(suspect_names),
            missing_in_suspect=sorted(clean_names - suspect_names),
            missing_in_clean=sorted(suspect_names - clean_names),
        )

    shape_mismatches: list[ShapeMismatch] = []
    tensor_diffs: list[TensorDiff] = []

    for name in common:
        c = clean[name]
        s = suspect[name]
        if c.shape != s.shape:
            shape_mismatches.append(
                ShapeMismatch(name=name, clean_shape=list(c.shape), suspect_shape=list(s.shape))
            )
            continue

        c64 = c.astype(np.float64, copy=False)
        s64 = s.astype(np.float64, copy=False)

        if not (np.all(np.isfinite(c64)) and np.all(np.isfinite(s64))):
            tensor_diffs.append(
                TensorDiff(
                    name=name,
                    shape=list(c.shape),
                    status="skipped_non_finite",
                    note="Clean or suspect tensor contains NaN/Inf; diff is not well-defined.",
                )
            )
            continue

        diff = s64 - c64
        abs_diff = np.abs(diff)
        l2_diff = float(np.linalg.norm(diff))
        clean_norm = float(np.linalg.norm(c64))
        percent_changed = float(np.count_nonzero(abs_diff > diff_tolerance)) / diff.size

        tensor_diffs.append(
            TensorDiff(
                name=name,
                shape=list(c.shape),
                status="compared",
                l2_diff=l2_diff,
                max_abs_diff=float(np.max(abs_diff)),
                mean_abs_diff=float(np.mean(abs_diff)),
                percent_changed=percent_changed,
                relative_l2_diff=(l2_diff / clean_norm) if clean_norm > 0 else None,
            )
        )

    compared = [d for d in tensor_diffs if d.status == "compared"]
    ranked = sorted(compared, key=lambda d: d.l2_diff or 0.0, reverse=True)
    most_affected = [d.name for d in ranked[:top_n_affected]]

    return DifferentialResult(
        status="compared",
        clean_tensor_count=len(clean_names),
        suspect_tensor_count=len(suspect_names),
        compared_count=len(compared),
        shape_mismatches=shape_mismatches,
        missing_in_suspect=sorted(clean_names - suspect_names),
        missing_in_clean=sorted(suspect_names - clean_names),
        tensor_diffs=tensor_diffs,
        most_affected_layers=most_affected,
    )
