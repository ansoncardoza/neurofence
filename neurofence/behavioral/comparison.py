"""Pairwise behavioral comparison between two model outputs.

A comparison reports *evidence* (semantic distance, length change, refusal
change) -- it does not itself decide "anomalous" or "not anomalous"; that
judgment belongs to the anomaly-detection layer that consumes many of these
comparisons together (see neurofence.behavioral.anomaly, added alongside
the fuzzer in a later milestone) or to evidence fusion.
"""

from __future__ import annotations

from pydantic import BaseModel

from neurofence.behavioral.embeddings import SimilarityBackend
from neurofence.behavioral.refusal import RefusalResult, detect_refusal


class BehavioralComparison(BaseModel):
    exact_match: bool
    semantic_similarity: float  # 1.0 = identical meaning (per the backend used), 0.0 = unrelated
    semantic_distance: float  # 1 - semantic_similarity
    length_baseline: int
    length_test: int
    length_ratio: float | None  # test/baseline; None when baseline length is 0 and test isn't
    length_ratio_note: str | None = None
    refusal_baseline: RefusalResult
    refusal_test: RefusalResult
    refusal_changed: bool


def compare_outputs(
    baseline_text: str,
    test_text: str,
    backend: SimilarityBackend,
) -> BehavioralComparison:
    len_baseline = len(baseline_text)
    len_test = len(test_text)

    if len_baseline > 0:
        length_ratio = len_test / len_baseline
        length_ratio_note = None
    elif len_test == 0:
        length_ratio = 1.0
        length_ratio_note = "Both outputs are empty."
    else:
        length_ratio = None
        length_ratio_note = "Baseline output is empty; ratio is undefined (division by zero)."

    similarity = backend.similarity(baseline_text, test_text)

    refusal_baseline = detect_refusal(baseline_text)
    refusal_test = detect_refusal(test_text)

    return BehavioralComparison(
        exact_match=baseline_text == test_text,
        semantic_similarity=similarity,
        semantic_distance=1.0 - similarity,
        length_baseline=len_baseline,
        length_test=len_test,
        length_ratio=length_ratio,
        length_ratio_note=length_ratio_note,
        refusal_baseline=refusal_baseline,
        refusal_test=refusal_test,
        refusal_changed=refusal_baseline.detected != refusal_test.detected,
    )
