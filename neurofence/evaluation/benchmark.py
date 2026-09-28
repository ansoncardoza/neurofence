"""Benchmarks NeuroFence detectors against synthetic ground-truth data.

Every number here is measured by actually running the detector against
the synthetic dataset at benchmark time -- nothing is invented, cached
from a prior "typical" run, or interpolated. Re-running with the same
seed reproduces the same numbers (weight forensics' Isolation Forest/LOF
are seeded; the synthetic dataset generation is seeded).
"""

from __future__ import annotations

import time

from pydantic import BaseModel

from neurofence.attack_lab.dataset import LabeledDataset
from neurofence.attack_lab.trigger_experiments import build_trigger_experiments
from neurofence.evaluation.metrics import (
    ClassificationMetrics,
    compute_binary_classification_metrics,
    compute_pr_auc,
    compute_roc_auc,
)
from neurofence.fuzzing.trigger_discovery import discover_trigger_candidates
from neurofence.weight_forensics.anomaly import detect_layer_anomalies
from neurofence.weight_forensics.pipeline import build_layer_feature_vector
from neurofence.weight_forensics.statistics import compute_tensor_statistics


class BenchmarkResult(BaseModel):
    detector: str
    n_samples: int
    metrics: ClassificationMetrics
    roc_auc: float | None
    roc_auc_note: str | None
    pr_auc: float | None
    pr_auc_note: str | None
    runtime_seconds: float
    per_sample_predictions: dict[str, dict[str, float | int | str]]
    layer_localization_rate: float | None = None
    layer_localization_note: str | None = None


def _predict_sample(tensors: dict) -> tuple[int, float, list[str]]:
    """One synthetic model -> (predicted_label, continuous_score, flagged_layer_names).

    Predicted label is 1 iff detect_layer_anomalies flags at least one
    layer as an outlier within that sample's own set of layers (there is
    no cross-sample baseline here -- each synthetic "model" is judged
    against its own other layers, consistent with how weight forensics
    runs against a single real model with no reference). The continuous
    score (for ROC/PR-AUC) is the maximum per-layer anomaly_score.

    Note this binary criterion is intentionally permissive (ANY flagged
    layer counts as "predicted poisoned"), which is exactly what makes
    accuracy/recall from `metrics` alone potentially misleading for small
    layer counts: an unsupervised contamination-based detector (Isolation
    Forest/LOF with contamination=0.1) flags roughly one layer out of ~10
    "by construction," largely independent of whether anything is
    actually anomalous. `layer_localization_rate` below is the more
    meaningful signal -- whether the *specific* attacked layer was among
    those flagged, not just whether something, anything, was flagged.
    """
    features = {
        name: build_layer_feature_vector(compute_tensor_statistics(arr))
        for name, arr in tensors.items()
    }
    summary = detect_layer_anomalies(features)
    if not summary.results:
        return 0, 0.0, []
    flagged = [r.layer_name for r in summary.results if r.is_outlier]
    score = max(r.anomaly_score for r in summary.results)
    return int(bool(flagged)), score, flagged


def evaluate_weight_anomaly_detector(dataset: LabeledDataset) -> BenchmarkResult:
    y_true: list[int] = []
    y_pred: list[int] = []
    y_scores: list[float] = []
    per_sample: dict[str, dict[str, float | int | str]] = {}
    localization_hits = 0
    localization_total = 0

    start = time.perf_counter()
    for sample in dataset.samples:
        predicted, score, flagged = _predict_sample(sample.tensor_arrays())
        y_true.append(sample.label)
        y_pred.append(predicted)
        y_scores.append(score)

        correctly_localized: bool | None = None
        if sample.perturbation is not None:
            localization_total += 1
            correctly_localized = sample.perturbation.layer_name in flagged
            localization_hits += int(correctly_localized)

        per_sample[sample.sample_id] = {
            "true_label": sample.label,
            "predicted_label": predicted,
            "score": score,
            "perturbation_kind": sample.perturbation.kind if sample.perturbation else "none",
            "flagged_layers": ",".join(flagged),
            "correctly_localized": (
                "" if correctly_localized is None else str(correctly_localized)
            ),
        }
    runtime = time.perf_counter() - start

    metrics = compute_binary_classification_metrics(y_true, y_pred)
    roc_auc, roc_note = compute_roc_auc(y_true, y_scores)
    pr_auc, pr_note = compute_pr_auc(y_true, y_scores)

    if localization_total > 0:
        localization_rate: float | None = localization_hits / localization_total
        localization_note = None
    else:
        localization_rate = None
        localization_note = "No poisoned samples in dataset; layer localization not evaluated."

    return BenchmarkResult(
        detector="weight_anomaly (isolation_forest+lof+mahalanobis consensus)",
        n_samples=len(dataset.samples),
        metrics=metrics,
        roc_auc=roc_auc,
        roc_auc_note=roc_note,
        pr_auc=pr_auc,
        pr_auc_note=pr_note,
        runtime_seconds=runtime,
        per_sample_predictions=per_sample,
        layer_localization_rate=localization_rate,
        layer_localization_note=localization_note,
    )


def evaluate_trigger_detector(
    planted_trigger: str = "zzz_backdoor_zzz",
    decoy_triggers: tuple[str, ...] = ("please", "hello there", "urgent request"),
) -> BenchmarkResult:
    """Runs trigger discovery against the two ground-truth experiments
    (one backdoored runner, one clean runner) built by
    neurofence.attack_lab.trigger_experiments, and checks whether it
    correctly identifies the planted trigger's phrase specifically (not
    just "some trigger was flagged") in the backdoored case, and stays
    quiet in the clean case.
    """
    from neurofence.behavioral.prompts import DEFAULT_PROMPTS

    experiments = build_trigger_experiments(planted_trigger, decoy_triggers)

    y_true: list[int] = []
    y_pred: list[int] = []
    y_scores: list[float] = []
    per_sample: dict[str, dict[str, float | int | str]] = {}

    start = time.perf_counter()
    for case, runner in experiments:
        candidates = list(case.candidate_triggers_tested)
        results = discover_trigger_candidates(runner, DEFAULT_PROMPTS, candidates)
        consistent = [r for r in results if r.consistent]
        # Correct positive detection requires flagging the *actual* planted
        # trigger consistently, not merely flagging something.
        if case.has_planted_trigger:
            predicted = int(any(r.trigger == case.planted_trigger for r in consistent))
        else:
            predicted = int(len(consistent) > 0)
        score = max((r.mean_anomaly_score for r in results), default=0.0)

        y_true.append(int(case.has_planted_trigger))
        y_pred.append(predicted)
        y_scores.append(score)
        per_sample[case.experiment_id] = {
            "true_label": int(case.has_planted_trigger),
            "predicted_label": predicted,
            "score": score,
            "num_consistent_candidates": len(consistent),
        }
    runtime = time.perf_counter() - start

    metrics = compute_binary_classification_metrics(y_true, y_pred)
    roc_auc, roc_note = compute_roc_auc(y_true, y_scores)
    pr_auc, pr_note = compute_pr_auc(y_true, y_scores)

    return BenchmarkResult(
        detector="trigger_discovery",
        n_samples=len(experiments),
        metrics=metrics,
        roc_auc=roc_auc,
        roc_auc_note=roc_note,
        pr_auc=pr_auc,
        pr_auc_note=pr_note,
        runtime_seconds=runtime,
        per_sample_predictions=per_sample,
    )
