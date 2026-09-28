from __future__ import annotations

from neurofence.fusion.scores import SubScore
from neurofence.fuzzing.trigger_discovery import TriggerCandidateResult
from neurofence.reporting.report import (
    LIMITATIONS,
    generate_findings,
    generate_recommendations,
)
from neurofence.weight_forensics.anomaly import AnomalyDetectionSummary, LayerAnomalyResult
from neurofence.weight_forensics.differential import DifferentialResult, TensorDiff
from neurofence.weight_forensics.pipeline import TensorForensicsRecord, WeightForensicsResult
from neurofence.weight_forensics.statistics import compute_tensor_statistics
from tests.reporting.conftest import make_sub_scores


def test_generate_findings_all_none_returns_empty() -> None:
    assert generate_findings(None, None, None) == []


def test_weight_findings_from_flagged_layer() -> None:
    import numpy as np

    stats = compute_tensor_statistics(np.zeros((4, 4)))
    summary = AnomalyDetectionSummary(
        status="ml_based",
        sample_count=10,
        feature_count=5,
        methods_used=["isolation_forest", "local_outlier_factor", "mahalanobis"],
        results=[
            LayerAnomalyResult(
                layer_name="bad_layer",
                is_outlier=True,
                anomaly_score=0.9,
                votes={
                    "isolation_forest": True,
                    "local_outlier_factor": True,
                    "mahalanobis": False,
                },
            ),
            LayerAnomalyResult(layer_name="ok_layer", is_outlier=False, anomaly_score=0.1),
        ],
    )
    weight_result = WeightForensicsResult(
        tensor_count=2,
        tensors=[TensorForensicsRecord(name="bad_layer", statistics=stats)],
        anomaly_detection=summary,
    )

    findings = generate_findings(weight_result, None, None)

    assert len(findings) == 1
    assert findings[0].affected == "bad_layer"
    assert findings[0].finding_id == "WEIGHT-001"
    assert "Multiple independent methods" in findings[0].confidence


def test_weight_findings_no_outliers_no_findings() -> None:
    summary = AnomalyDetectionSummary(
        status="ml_based",
        sample_count=10,
        feature_count=5,
        results=[LayerAnomalyResult(layer_name="l1", is_outlier=False, anomaly_score=0.1)],
    )
    weight_result = WeightForensicsResult(tensor_count=1, tensors=[], anomaly_detection=summary)
    assert generate_findings(weight_result, None, None) == []


def test_differential_findings_from_most_affected() -> None:
    diff = DifferentialResult(
        status="compared",
        clean_tensor_count=2,
        suspect_tensor_count=2,
        compared_count=2,
        tensor_diffs=[
            TensorDiff(
                name="layer1",
                shape=[4, 4],
                status="compared",
                l2_diff=50.0,
                max_abs_diff=50.0,
                mean_abs_diff=1.0,
                percent_changed=0.1,
                relative_l2_diff=1.0,
            ),
        ],
        most_affected_layers=["layer1"],
    )
    findings = generate_findings(None, diff, None)
    assert len(findings) == 1
    assert findings[0].affected == "layer1"
    assert findings[0].finding_id == "DIFF-001"


def test_differential_reference_incompatible_no_findings() -> None:
    diff = DifferentialResult(status="reference_incompatible", reason="no overlap")
    assert generate_findings(None, diff, None) == []


def test_trigger_findings_only_consistent() -> None:
    results = [
        TriggerCandidateResult(
            trigger="evil",
            num_prompts_tested=4,
            mean_anomaly_score=0.8,
            min_anomaly_score=0.5,
            max_anomaly_score=1.0,
            consistent=True,
            evidence=[],
        ),
        TriggerCandidateResult(
            trigger="benign",
            num_prompts_tested=4,
            mean_anomaly_score=0.1,
            min_anomaly_score=0.0,
            max_anomaly_score=0.2,
            consistent=False,
            evidence=[],
        ),
    ]
    findings = generate_findings(None, None, results)
    assert len(findings) == 1
    assert findings[0].affected == "evil"
    assert findings[0].finding_id == "TRIGGER-001"


def test_recommendations_not_evaluated() -> None:
    from neurofence.config.schema import RiskWeights
    from neurofence.fusion.combine import fuse_evidence

    fusion = fuse_evidence(make_sub_scores(), RiskWeights())
    recs = generate_recommendations(fusion)
    assert len(recs) == 1
    assert "No detectors were run" in recs[0]


def test_recommendations_per_risk_label() -> None:
    from neurofence.config.schema import RiskWeights
    from neurofence.fusion.combine import fuse_evidence

    for label, score in [("LOW", 5.0), ("MEDIUM", 30.0), ("HIGH", 60.0), ("CRITICAL", 90.0)]:
        weight_score = SubScore(name="weight_anomaly", status="evaluated", score=score)
        sub_scores = make_sub_scores(weight_anomaly=weight_score)
        fusion = fuse_evidence(sub_scores, RiskWeights())
        recs = generate_recommendations(fusion)
        assert len(recs) > 0, label


def test_limitations_non_empty_and_static() -> None:
    assert len(LIMITATIONS) >= 3
    assert all(isinstance(item, str) for item in LIMITATIONS)
