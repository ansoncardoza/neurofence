"""Binary classification metrics for detector evaluation.

Every metric that is mathematically undefined for the given input (e.g.
precision when there are no positive predictions at all) is reported as
`None` with a note -- never coerced to 0.0, which would misreport
"undefined" as "measured and zero." Per the project's rule against faking
results, nothing here is estimated or interpolated: only what can be
computed from the given labels is reported.
"""

from __future__ import annotations

from pydantic import BaseModel


class ConfusionMatrix(BaseModel):
    true_positive: int
    false_positive: int
    true_negative: int
    false_negative: int


class ClassificationMetrics(BaseModel):
    n_samples: int
    confusion_matrix: ConfusionMatrix
    accuracy: float
    precision: float | None
    recall: float | None
    f1: float | None
    specificity: float | None
    false_positive_rate: float | None
    false_negative_rate: float | None
    notes: list[str] = []


def compute_binary_classification_metrics(
    y_true: list[int], y_pred: list[int]
) -> ClassificationMetrics:
    """`y_true`/`y_pred` are 0/1 labels (0 = clean/negative, 1 = poisoned/
    positive), equal length, non-empty.
    """
    if len(y_true) != len(y_pred):
        raise ValueError(
            f"y_true and y_pred must have equal length, got {len(y_true)} vs {len(y_pred)}."
        )
    if len(y_true) == 0:
        raise ValueError("y_true/y_pred must not be empty.")
    if not all(v in (0, 1) for v in y_true) or not all(v in (0, 1) for v in y_pred):
        raise ValueError("y_true and y_pred must contain only 0/1 labels.")

    tp = sum(1 for t, p in zip(y_true, y_pred, strict=True) if t == 1 and p == 1)
    fp = sum(1 for t, p in zip(y_true, y_pred, strict=True) if t == 0 and p == 1)
    tn = sum(1 for t, p in zip(y_true, y_pred, strict=True) if t == 0 and p == 0)
    fn = sum(1 for t, p in zip(y_true, y_pred, strict=True) if t == 1 and p == 0)

    notes: list[str] = []
    accuracy = (tp + tn) / len(y_true)

    precision = tp / (tp + fp) if (tp + fp) > 0 else None
    if precision is None:
        notes.append("precision undefined: no positive predictions were made.")

    recall = tp / (tp + fn) if (tp + fn) > 0 else None
    if recall is None:
        notes.append("recall undefined: no positive ground-truth samples present.")

    f1 = (
        (2 * precision * recall / (precision + recall))
        if (precision is not None and recall is not None and (precision + recall) > 0)
        else None
    )
    if f1 is None and precision is not None and recall is not None:
        notes.append("f1 undefined: precision and recall both zero.")

    specificity = tn / (tn + fp) if (tn + fp) > 0 else None
    if specificity is None:
        notes.append("specificity undefined: no negative ground-truth samples present.")

    fpr = fp / (fp + tn) if (fp + tn) > 0 else None
    fnr = fn / (fn + tp) if (fn + tp) > 0 else None

    return ClassificationMetrics(
        n_samples=len(y_true),
        confusion_matrix=ConfusionMatrix(
            true_positive=tp, false_positive=fp, true_negative=tn, false_negative=fn
        ),
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1=f1,
        specificity=specificity,
        false_positive_rate=fpr,
        false_negative_rate=fnr,
        notes=notes,
    )


def compute_roc_auc(y_true: list[int], y_scores: list[float]) -> tuple[float | None, str | None]:
    """Returns (auc, note). ROC-AUC is undefined when y_true has only one
    class -- reported as (None, reason) rather than raising or guessing.
    """
    if len(set(y_true)) < 2:
        return None, "ROC-AUC undefined: y_true contains only one class."
    from sklearn.metrics import roc_auc_score

    return float(roc_auc_score(y_true, y_scores)), None


def compute_pr_auc(y_true: list[int], y_scores: list[float]) -> tuple[float | None, str | None]:
    if len(set(y_true)) < 2:
        return None, "PR-AUC undefined: y_true contains only one class."
    from sklearn.metrics import average_precision_score

    return float(average_precision_score(y_true, y_scores)), None
