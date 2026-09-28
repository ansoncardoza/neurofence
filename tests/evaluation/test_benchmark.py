from __future__ import annotations

from neurofence.attack_lab.dataset import build_labeled_dataset
from neurofence.evaluation.benchmark import (
    evaluate_trigger_detector,
    evaluate_weight_anomaly_detector,
)


def test_weight_anomaly_benchmark_runs_and_measures_something() -> None:
    dataset = build_labeled_dataset(n_clean=10, n_poisoned_per_kind=2, seed=1)
    result = evaluate_weight_anomaly_detector(dataset)

    assert result.n_samples == len(dataset.samples)
    assert result.runtime_seconds >= 0.0
    assert 0.0 <= result.metrics.accuracy <= 1.0
    assert len(result.per_sample_predictions) == len(dataset.samples)


def test_weight_anomaly_benchmark_deterministic() -> None:
    dataset = build_labeled_dataset(n_clean=8, n_poisoned_per_kind=1, seed=5)
    result_a = evaluate_weight_anomaly_detector(dataset)
    result_b = evaluate_weight_anomaly_detector(dataset)
    assert result_a.metrics.accuracy == result_b.metrics.accuracy
    assert result_a.metrics.confusion_matrix == result_b.metrics.confusion_matrix


def test_weight_anomaly_benchmark_detects_obvious_localized_attack() -> None:
    # A large single-element shift (localized, default magnitude 50) on a
    # small-scale (~0.05) tensor should be an easy catch for the detector
    # -- this is a sanity check that the pipeline is actually wired
    # correctly end-to-end, not a claim about real-world detection rates.
    dataset = build_labeled_dataset(n_clean=10, n_poisoned_per_kind=0, seed=2)
    # Manually add one obviously-poisoned sample alongside the clean ones.
    import numpy as np

    from neurofence.attack_lab.dataset import LabeledSample
    from neurofence.attack_lab.synthetic_models import (
        apply_localized_perturbation,
        generate_clean_tensors,
    )

    tensors = generate_clean_tensors(seed=999)
    rng = np.random.default_rng(999)
    poisoned, record = apply_localized_perturbation(
        tensors, rng, layer_name="attn.q_proj.weight", magnitude=500.0
    )
    dataset.samples.append(
        LabeledSample(
            sample_id="obvious_poison",
            label=1,
            tensors={name: arr.tolist() for name, arr in poisoned.items()},
            perturbation=record,
        )
    )

    result = evaluate_weight_anomaly_detector(dataset)
    assert result.per_sample_predictions["obvious_poison"]["predicted_label"] == 1


def test_weight_anomaly_benchmark_localizes_obvious_attacks() -> None:
    # The four large/structural perturbation kinds should have their
    # actual attacked layer specifically flagged, not just "something"
    # flagged -- this is the meaningful signal, since the binary
    # predicted_label criterion (any layer flagged) is permissive enough
    # that a contamination-based detector flags roughly one layer
    # regardless of ground truth.
    dataset = build_labeled_dataset(
        n_clean=0,
        n_poisoned_per_kind=3,
        perturbation_kinds=("localized", "distributed", "layer_level", "neuron_level"),
        seed=7,
    )
    result = evaluate_weight_anomaly_detector(dataset)
    assert result.layer_localization_rate == 1.0


def test_weight_anomaly_benchmark_localization_not_evaluated_with_no_poisoned_samples() -> None:
    dataset = build_labeled_dataset(n_clean=5, n_poisoned_per_kind=0, seed=8)
    result = evaluate_weight_anomaly_detector(dataset)
    assert result.layer_localization_rate is None
    assert result.layer_localization_note is not None


def test_weight_anomaly_benchmark_roc_auc_reported_when_both_classes_present() -> None:
    dataset = build_labeled_dataset(n_clean=10, n_poisoned_per_kind=2, seed=3)
    result = evaluate_weight_anomaly_detector(dataset)
    assert result.roc_auc is not None
    assert 0.0 <= result.roc_auc <= 1.0


def test_weight_anomaly_benchmark_single_class_auc_not_evaluated() -> None:
    dataset = build_labeled_dataset(n_clean=5, n_poisoned_per_kind=0, seed=4)
    result = evaluate_weight_anomaly_detector(dataset)
    assert result.roc_auc is None
    assert result.roc_auc_note is not None


def test_trigger_detector_benchmark_runs() -> None:
    result = evaluate_trigger_detector()
    assert result.n_samples == 2
    assert result.detector == "trigger_discovery"
    assert set(result.per_sample_predictions) == {"backdoored", "clean"}


def test_trigger_detector_benchmark_correctly_labels_clean_case() -> None:
    result = evaluate_trigger_detector()
    assert result.per_sample_predictions["clean"]["true_label"] == 0
    assert result.per_sample_predictions["clean"]["predicted_label"] == 0


def test_trigger_detector_benchmark_correctly_labels_backdoored_case() -> None:
    result = evaluate_trigger_detector()
    assert result.per_sample_predictions["backdoored"]["true_label"] == 1
    assert result.per_sample_predictions["backdoored"]["predicted_label"] == 1


def test_trigger_detector_benchmark_perfect_accuracy_on_ground_truth() -> None:
    # With a clearly-signaled scripted backdoor vs a clean runner, the
    # detector should get both ground-truth cases right -- this exercises
    # the full discover_trigger_candidates pipeline, not just a mock.
    result = evaluate_trigger_detector()
    assert result.metrics.accuracy == 1.0
