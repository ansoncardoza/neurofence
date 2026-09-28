from __future__ import annotations

import pytest

from neurofence.evaluation.metrics import (
    compute_binary_classification_metrics,
    compute_pr_auc,
    compute_roc_auc,
)


def test_perfect_classifier() -> None:
    y_true = [0, 0, 1, 1]
    y_pred = [0, 0, 1, 1]
    metrics = compute_binary_classification_metrics(y_true, y_pred)
    assert metrics.accuracy == 1.0
    assert metrics.precision == 1.0
    assert metrics.recall == 1.0
    assert metrics.f1 == 1.0
    assert metrics.specificity == 1.0
    assert metrics.false_positive_rate == 0.0
    assert metrics.false_negative_rate == 0.0
    assert metrics.confusion_matrix.true_positive == 2
    assert metrics.confusion_matrix.true_negative == 2


def test_worst_classifier() -> None:
    y_true = [0, 0, 1, 1]
    y_pred = [1, 1, 0, 0]
    metrics = compute_binary_classification_metrics(y_true, y_pred)
    assert metrics.accuracy == 0.0
    assert metrics.precision == 0.0
    assert metrics.recall == 0.0


def test_no_positive_predictions_precision_undefined() -> None:
    y_true = [0, 1, 1]
    y_pred = [0, 0, 0]
    metrics = compute_binary_classification_metrics(y_true, y_pred)
    assert metrics.precision is None
    assert any("precision undefined" in n for n in metrics.notes)
    assert metrics.recall == 0.0  # recall IS defined here (there are positives in y_true)


def test_all_negative_ground_truth_recall_undefined() -> None:
    y_true = [0, 0, 0]
    y_pred = [0, 1, 0]
    metrics = compute_binary_classification_metrics(y_true, y_pred)
    assert metrics.recall is None
    assert any("recall undefined" in n for n in metrics.notes)


def test_all_positive_ground_truth_specificity_undefined() -> None:
    y_true = [1, 1, 1]
    y_pred = [1, 0, 1]
    metrics = compute_binary_classification_metrics(y_true, y_pred)
    assert metrics.specificity is None
    assert any("specificity undefined" in n for n in metrics.notes)


def test_mismatched_length_raises() -> None:
    with pytest.raises(ValueError):
        compute_binary_classification_metrics([0, 1], [0, 1, 1])


def test_empty_input_raises() -> None:
    with pytest.raises(ValueError):
        compute_binary_classification_metrics([], [])


def test_non_binary_labels_raise() -> None:
    with pytest.raises(ValueError):
        compute_binary_classification_metrics([0, 2], [0, 1])


def test_single_sample() -> None:
    metrics = compute_binary_classification_metrics([1], [1])
    assert metrics.accuracy == 1.0
    assert metrics.n_samples == 1


def test_confusion_matrix_sums_to_n_samples() -> None:
    y_true = [0, 0, 1, 1, 1, 0]
    y_pred = [0, 1, 1, 0, 1, 0]
    metrics = compute_binary_classification_metrics(y_true, y_pred)
    cm = metrics.confusion_matrix
    assert cm.true_positive + cm.false_positive + cm.true_negative + cm.false_negative == 6


def test_roc_auc_perfect_separation() -> None:
    y_true = [0, 0, 1, 1]
    y_scores = [0.1, 0.2, 0.8, 0.9]
    auc, note = compute_roc_auc(y_true, y_scores)
    assert auc == 1.0
    assert note is None


def test_roc_auc_single_class_undefined() -> None:
    auc, note = compute_roc_auc([1, 1, 1], [0.1, 0.5, 0.9])
    assert auc is None
    assert note is not None and "only one class" in note


def test_pr_auc_single_class_undefined() -> None:
    auc, note = compute_pr_auc([0, 0, 0], [0.1, 0.5, 0.9])
    assert auc is None
    assert note is not None


def test_pr_auc_perfect_separation() -> None:
    auc, note = compute_pr_auc([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9])
    assert auc == 1.0
    assert note is None


def test_roc_auc_random_scores_near_half() -> None:
    # Not a strict assertion on the exact value (that would be flaky) --
    # just confirms it's computed and within valid [0,1] bounds.
    y_true = [0, 1, 0, 1, 0, 1, 0, 1]
    y_scores = [0.5] * 8  # ties everywhere -> AUC == 0.5 exactly
    auc, note = compute_roc_auc(y_true, y_scores)
    assert auc == 0.5
    assert note is None
