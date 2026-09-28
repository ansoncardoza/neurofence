from __future__ import annotations

import json

from neurofence.reporting.json_report import render_json_report, write_json_report
from neurofence.reporting.report import ScanReport, build_report
from tests.reporting.conftest import minimal_report


def test_build_report_minimal(manifest, metadata, not_evaluated_fusion) -> None:
    report = build_report(
        model_path="/fake/path",
        manifest=manifest,
        metadata=metadata,
        evidence_fusion=not_evaluated_fusion,
    )
    assert isinstance(report, ScanReport)
    assert report.weight_forensics is None
    assert report.findings == []
    assert len(report.recommendations) == 1
    assert len(report.limitations) >= 3


def test_build_report_findings_sorted_by_severity_descending(manifest, metadata) -> None:
    import numpy as np

    from neurofence.config.schema import RiskWeights
    from neurofence.fusion.combine import fuse_evidence
    from neurofence.fusion.scores import SubScore
    from neurofence.weight_forensics.anomaly import AnomalyDetectionSummary, LayerAnomalyResult
    from neurofence.weight_forensics.pipeline import TensorForensicsRecord, WeightForensicsResult
    from neurofence.weight_forensics.statistics import compute_tensor_statistics

    stats = compute_tensor_statistics(np.zeros((4, 4)))
    summary = AnomalyDetectionSummary(
        status="ml_based",
        sample_count=10,
        feature_count=5,
        results=[
            LayerAnomalyResult(layer_name="low_sev", is_outlier=True, anomaly_score=0.1),
            LayerAnomalyResult(layer_name="high_sev", is_outlier=True, anomaly_score=0.95),
        ],
    )
    weight_result = WeightForensicsResult(
        tensor_count=2,
        tensors=[TensorForensicsRecord(name="x", statistics=stats)],
        anomaly_detection=summary,
    )
    fusion = fuse_evidence(
        {
            "integrity": SubScore(name="integrity", status="not_evaluated"),
            "weight_anomaly": SubScore(name="weight_anomaly", status="evaluated", score=50.0),
            "behavioral": SubScore(name="behavioral", status="not_evaluated"),
            "activation": SubScore(name="activation", status="not_evaluated"),
            "trigger": SubScore(name="trigger", status="not_evaluated"),
        },
        RiskWeights(),
    )

    report = build_report(
        model_path="/fake/path",
        manifest=manifest,
        metadata=metadata,
        evidence_fusion=fusion,
        weight_forensics=weight_result,
    )

    severities = [f.severity for f in report.findings]
    order = {"LOW": 0, "MEDIUM": 1, "HIGH": 2, "CRITICAL": 3}
    assert [order[s] for s in severities] == sorted([order[s] for s in severities], reverse=True)


def test_render_json_report_is_valid_json(manifest, metadata, not_evaluated_fusion) -> None:
    report = minimal_report(manifest, metadata, not_evaluated_fusion)
    text = render_json_report(report)
    parsed = json.loads(text)
    assert parsed["model_path"] == "/fake/path"
    assert "findings" in parsed
    assert "limitations" in parsed


def test_write_json_report_creates_file(tmp_path, manifest, metadata, not_evaluated_fusion) -> None:
    report = minimal_report(manifest, metadata, not_evaluated_fusion)
    path = write_json_report(report, tmp_path / "report.json")
    assert path.exists()
    parsed = json.loads(path.read_text(encoding="utf-8"))
    assert parsed["neurofence_version"]


def test_json_report_round_trips_through_pydantic(manifest, metadata, not_evaluated_fusion) -> None:
    report = minimal_report(manifest, metadata, not_evaluated_fusion)
    text = render_json_report(report)
    reloaded = ScanReport.model_validate_json(text)
    assert reloaded.model_path == report.model_path
    assert reloaded.evidence_fusion.status == report.evidence_fusion.status
