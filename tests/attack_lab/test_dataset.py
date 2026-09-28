from __future__ import annotations

from neurofence.attack_lab.dataset import build_labeled_dataset
from neurofence.attack_lab.synthetic_models import PERTURBATION_KINDS


def test_dataset_has_expected_sample_counts() -> None:
    dataset = build_labeled_dataset(n_clean=5, n_poisoned_per_kind=2)
    labels = dataset.labels()
    assert labels.count(0) == 5
    assert labels.count(1) == 2 * len(PERTURBATION_KINDS)


def test_dataset_deterministic_given_seed() -> None:
    a = build_labeled_dataset(n_clean=3, n_poisoned_per_kind=1, seed=42)
    b = build_labeled_dataset(n_clean=3, n_poisoned_per_kind=1, seed=42)
    for sa, sb in zip(a.samples, b.samples, strict=True):
        assert sa.tensors == sb.tensors
        assert sa.label == sb.label


def test_dataset_different_seeds_differ() -> None:
    a = build_labeled_dataset(n_clean=3, n_poisoned_per_kind=1, seed=1)
    b = build_labeled_dataset(n_clean=3, n_poisoned_per_kind=1, seed=2)
    assert a.samples[0].tensors != b.samples[0].tensors


def test_poisoned_samples_carry_perturbation_record() -> None:
    dataset = build_labeled_dataset(n_clean=1, n_poisoned_per_kind=1)
    poisoned = [s for s in dataset.samples if s.label == 1]
    assert all(s.perturbation is not None for s in poisoned)


def test_clean_samples_have_no_perturbation_record() -> None:
    dataset = build_labeled_dataset(n_clean=3, n_poisoned_per_kind=0)
    clean = [s for s in dataset.samples if s.label == 0]
    assert all(s.perturbation is None for s in clean)


def test_tensor_arrays_round_trip() -> None:
    dataset = build_labeled_dataset(n_clean=1, n_poisoned_per_kind=0)
    sample = dataset.samples[0]
    arrays = sample.tensor_arrays()
    assert set(arrays) == set(dataset.layer_shapes)
    for name, shape in dataset.layer_shapes.items():
        assert list(arrays[name].shape) == shape


def test_restricted_perturbation_kinds() -> None:
    dataset = build_labeled_dataset(
        n_clean=0, n_poisoned_per_kind=2, perturbation_kinds=("localized",)
    )
    kinds = {s.perturbation.kind for s in dataset.samples}
    assert kinds == {"localized"}


def test_zero_samples_produces_empty_dataset() -> None:
    dataset = build_labeled_dataset(n_clean=0, n_poisoned_per_kind=0)
    assert dataset.samples == []
