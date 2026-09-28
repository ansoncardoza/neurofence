from __future__ import annotations

import numpy as np
import pytest

from neurofence.attack_lab.synthetic_models import (
    DEFAULT_LAYER_SHAPES,
    PERTURBATION_KINDS,
    apply_localized_perturbation,
    apply_perturbation,
    generate_clean_tensors,
)


def test_generate_clean_tensors_deterministic() -> None:
    a = generate_clean_tensors(seed=0)
    b = generate_clean_tensors(seed=0)
    for name in a:
        assert np.array_equal(a[name], b[name])


def test_generate_clean_tensors_different_seeds_differ() -> None:
    a = generate_clean_tensors(seed=0)
    b = generate_clean_tensors(seed=1)
    assert not all(np.array_equal(a[name], b[name]) for name in a)


def test_generate_clean_tensors_matches_requested_shapes() -> None:
    tensors = generate_clean_tensors(DEFAULT_LAYER_SHAPES, seed=0)
    for name, shape in DEFAULT_LAYER_SHAPES.items():
        assert tensors[name].shape == shape


def test_localized_perturbation_changes_exactly_one_element() -> None:
    tensors = generate_clean_tensors(seed=0)
    rng = np.random.default_rng(1)
    poisoned, record = apply_localized_perturbation(tensors, rng, layer_name="attn.q_proj.weight")

    diff_count = np.count_nonzero(poisoned["attn.q_proj.weight"] != tensors["attn.q_proj.weight"])
    assert diff_count == 1
    assert record.kind == "localized"
    assert record.layer_name == "attn.q_proj.weight"

    for name in tensors:
        if name != "attn.q_proj.weight":
            assert np.array_equal(tensors[name], poisoned[name])  # other layers untouched


def test_all_perturbation_kinds_run_without_error() -> None:
    for kind in PERTURBATION_KINDS:
        tensors = generate_clean_tensors(seed=0)
        rng = np.random.default_rng(0)
        poisoned, record = apply_perturbation(kind, tensors, rng)
        assert record.kind == kind
        assert any(
            not np.array_equal(tensors[name], poisoned[name]) for name in tensors
        ), f"{kind} produced no change"


def test_unknown_perturbation_kind_raises() -> None:
    tensors = generate_clean_tensors(seed=0)
    with pytest.raises(ValueError):
        apply_perturbation("not_a_real_kind", tensors, np.random.default_rng(0))


def test_perturbation_does_not_mutate_input() -> None:
    tensors = generate_clean_tensors(seed=0)
    original = {name: arr.copy() for name, arr in tensors.items()}
    apply_perturbation("distributed", tensors, np.random.default_rng(0))
    for name in tensors:
        assert np.array_equal(tensors[name], original[name])


def test_neuron_level_requires_2d_tensor() -> None:
    from neurofence.attack_lab.synthetic_models import apply_neuron_level_perturbation

    tensors = {"bias_only": np.zeros(10, dtype=np.float32)}
    with pytest.raises(ValueError):
        apply_neuron_level_perturbation(tensors, np.random.default_rng(0))


def test_low_magnitude_smaller_effect_than_localized() -> None:
    tensors = generate_clean_tensors(seed=0)
    rng1 = np.random.default_rng(0)
    rng2 = np.random.default_rng(0)
    poisoned_low, _ = apply_perturbation("low_magnitude", tensors, rng1)
    poisoned_local, _ = apply_perturbation("localized", tensors, rng2)

    low_diff = sum(
        np.sum(np.abs(poisoned_low[n] - tensors[n])) for n in tensors
    )
    local_diff = sum(
        np.sum(np.abs(poisoned_local[n] - tensors[n])) for n in tensors
    )
    assert low_diff < local_diff
