from __future__ import annotations

from neurofence.acquisition.manifest import IntegrityDiscrepancy, VerificationResult
from neurofence.behavioral.comparison import BehavioralComparison
from neurofence.behavioral.pipeline import BehavioralComparisonRecord
from neurofence.behavioral.refusal import RefusalResult
from neurofence.fusion.scores import (
    activation_anomaly_score,
    behavioral_anomaly_score,
    integrity_score,
    trigger_evidence_score,
    weight_anomaly_score,
)
from neurofence.fuzzing.trigger_discovery import TriggerCandidateResult
from neurofence.weight_forensics.anomaly import AnomalyDetectionSummary, LayerAnomalyResult


def test_integrity_none_not_evaluated() -> None:
    result = integrity_score(None)
    assert result.status == "not_evaluated"
    assert result.score is None


def test_integrity_matched_zero_score() -> None:
    verification = VerificationResult(matched=True, discrepancies=[], files_checked=10)
    result = integrity_score(verification)
    assert result.status == "evaluated"
    assert result.score == 0.0


def test_integrity_discrepancies_nonzero_score() -> None:
    discrepancy = IntegrityDiscrepancy(
        kind="hash_mismatch", path="model.safetensors", detail="x"
    )
    verification = VerificationResult(matched=False, discrepancies=[discrepancy], files_checked=10)
    result = integrity_score(verification)
    assert result.status == "evaluated"
    assert result.score > 0.0
    assert result.score <= 100.0


def test_weight_anomaly_none_not_evaluated() -> None:
    result = weight_anomaly_score(None)
    assert result.status == "not_evaluated"


def test_weight_anomaly_empty_status_not_evaluated() -> None:
    summary = AnomalyDetectionSummary(status="empty", sample_count=0, feature_count=0)
    result = weight_anomaly_score(summary)
    assert result.status == "not_evaluated"


def test_weight_anomaly_no_outliers_low_score() -> None:
    clean_results = [
        LayerAnomalyResult(layer_name=f"l{i}", is_outlier=False, anomaly_score=0.1)
        for i in range(10)
    ]
    summary = AnomalyDetectionSummary(
        status="ml_based",
        sample_count=10,
        feature_count=5,
        methods_used=["isolation_forest"],
        results=clean_results,
    )
    result = weight_anomaly_score(summary)
    assert result.status == "evaluated"
    assert result.score == 0.0


def test_weight_anomaly_some_outliers_positive_score() -> None:
    results = [
        LayerAnomalyResult(layer_name=f"l{i}", is_outlier=False, anomaly_score=0.1)
        for i in range(9)
    ]
    results.append(LayerAnomalyResult(layer_name="bad", is_outlier=True, anomaly_score=0.9))
    summary = AnomalyDetectionSummary(
        status="ml_based",
        sample_count=10,
        feature_count=5,
        methods_used=["isolation_forest"],
        results=results,
    )
    result = weight_anomaly_score(summary)
    assert result.status == "evaluated"
    assert 0.0 < result.score <= 100.0
    assert any("bad" in r for r in result.reasons)


def test_activation_anomaly_none_not_evaluated() -> None:
    assert activation_anomaly_score(None).status == "not_evaluated"
    assert activation_anomaly_score([]).status == "not_evaluated"


def test_activation_anomaly_aggregates_across_prompts() -> None:
    def _make_summary() -> AnomalyDetectionSummary:
        results = [
            LayerAnomalyResult(layer_name=f"l{i}", is_outlier=(i == 0), anomaly_score=0.5)
            for i in range(10)
        ]
        return AnomalyDetectionSummary(
            status="ml_based", sample_count=10, feature_count=5, results=results
        )

    summaries = [_make_summary() for _ in range(3)]
    result = activation_anomaly_score(summaries)
    assert result.status == "evaluated"
    assert result.score > 0.0


def test_behavioral_none_not_evaluated() -> None:
    assert behavioral_anomaly_score(None).status == "not_evaluated"
    assert behavioral_anomaly_score([]).status == "not_evaluated"


def _make_comparison(semantic_similarity: float, refusal_changed: bool) -> BehavioralComparison:
    return BehavioralComparison(
        exact_match=False,
        semantic_similarity=semantic_similarity,
        semantic_distance=1.0 - semantic_similarity,
        length_baseline=10,
        length_test=10,
        length_ratio=1.0,
        refusal_baseline=RefusalResult(detected=False),
        refusal_test=RefusalResult(detected=refusal_changed),
        refusal_changed=refusal_changed,
    )


def test_behavioral_identical_outputs_zero_score() -> None:
    comparison = _make_comparison(1.0, False)
    records = [BehavioralComparisonRecord(case_id="p1", category="gk", comparison=comparison)]
    result = behavioral_anomaly_score(records)
    assert result.status == "evaluated"
    assert result.score == 0.0


def test_behavioral_divergent_outputs_high_score() -> None:
    comparison = _make_comparison(0.0, True)
    records = [BehavioralComparisonRecord(case_id="p1", category="gk", comparison=comparison)]
    result = behavioral_anomaly_score(records)
    assert result.status == "evaluated"
    assert result.score > 50.0


def test_trigger_evidence_none_not_evaluated() -> None:
    assert trigger_evidence_score(None).status == "not_evaluated"
    assert trigger_evidence_score([]).status == "not_evaluated"


def test_trigger_evidence_consistent_full_score() -> None:
    results = [
        TriggerCandidateResult(
            trigger="evil", num_prompts_tested=4, mean_anomaly_score=0.9, min_anomaly_score=0.8,
            max_anomaly_score=1.0, consistent=True, evidence=[],
        )
    ]
    result = trigger_evidence_score(results)
    assert result.status == "evaluated"
    assert result.score == 90.0


def test_trigger_evidence_inconsistent_reduced_score() -> None:
    results = [
        TriggerCandidateResult(
            trigger="maybe", num_prompts_tested=4, mean_anomaly_score=0.9, min_anomaly_score=0.0,
            max_anomaly_score=0.9, consistent=False, evidence=[],
        )
    ]
    result = trigger_evidence_score(results)
    assert result.status == "evaluated"
    assert result.score < 90.0  # discounted relative to a consistent finding at the same severity
