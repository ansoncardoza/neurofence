"""Builds labeled synthetic datasets (clean=0, poisoned=1) for measuring
weight-forensics detector performance against known ground truth.
"""

from __future__ import annotations

import numpy as np
from pydantic import BaseModel

from neurofence.attack_lab.synthetic_models import (
    DEFAULT_LAYER_SHAPES,
    PERTURBATION_KINDS,
    PerturbationRecord,
    apply_perturbation,
    generate_clean_tensors,
)


class LabeledSample(BaseModel):
    sample_id: str
    label: int  # 0 = clean, 1 = poisoned
    tensors: dict[str, list]  # numpy arrays are not JSON-native; serialized as nested lists
    perturbation: PerturbationRecord | None = None

    def tensor_arrays(self) -> dict[str, np.ndarray]:
        return {name: np.array(values, dtype=np.float32) for name, values in self.tensors.items()}


class LabeledDataset(BaseModel):
    samples: list[LabeledSample]
    seed: int
    layer_shapes: dict[str, list[int]]

    def labels(self) -> list[int]:
        return [s.label for s in self.samples]


def build_labeled_dataset(
    n_clean: int = 15,
    n_poisoned_per_kind: int = 3,
    shapes: dict[str, tuple[int, ...]] | None = None,
    seed: int = 1337,
    perturbation_kinds: tuple[str, ...] = PERTURBATION_KINDS,
) -> LabeledDataset:
    """Build a deterministic labeled dataset: `n_clean` unperturbed
    synthetic models, plus `n_poisoned_per_kind` poisoned models for each
    requested perturbation kind. Each sample uses its own model seed
    (derived from `seed`) so clean and poisoned samples are all distinct
    synthetic models, not repeats of the same one.
    """
    shapes = shapes or DEFAULT_LAYER_SHAPES
    samples: list[LabeledSample] = []
    sample_seed_counter = 0

    for i in range(n_clean):
        model_seed = seed * 1000 + sample_seed_counter
        sample_seed_counter += 1
        tensors = generate_clean_tensors(shapes, seed=model_seed)
        samples.append(
            LabeledSample(
                sample_id=f"clean_{i:03d}",
                label=0,
                tensors={name: arr.tolist() for name, arr in tensors.items()},
            )
        )

    for kind in perturbation_kinds:
        for i in range(n_poisoned_per_kind):
            model_seed = seed * 1000 + sample_seed_counter
            sample_seed_counter += 1
            base_tensors = generate_clean_tensors(shapes, seed=model_seed)
            rng = np.random.default_rng(seed * 7919 + sample_seed_counter)
            poisoned_tensors, record = apply_perturbation(kind, base_tensors, rng)
            samples.append(
                LabeledSample(
                    sample_id=f"{kind}_{i:03d}",
                    label=1,
                    tensors={name: arr.tolist() for name, arr in poisoned_tensors.items()},
                    perturbation=record,
                )
            )

    return LabeledDataset(
        samples=samples,
        seed=seed,
        layer_shapes={name: list(shape) for name, shape in shapes.items()},
    )
