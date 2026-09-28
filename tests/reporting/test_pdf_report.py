from __future__ import annotations

from pathlib import Path

from neurofence.reporting.pdf_report import render_pdf_report
from neurofence.reporting.report import build_report
from tests.reporting.conftest import minimal_report


def _extract_text(pdf_path: Path) -> str:
    from pypdf import PdfReader

    reader = PdfReader(str(pdf_path))
    return "\n".join(page.extract_text() for page in reader.pages)


def test_render_pdf_minimal_report_produces_valid_pdf(
    tmp_path, manifest, metadata, not_evaluated_fusion
) -> None:
    report = minimal_report(manifest, metadata, not_evaluated_fusion)
    out = render_pdf_report(report, tmp_path / "report.pdf")

    assert out.exists()
    assert out.stat().st_size > 0
    assert out.read_bytes()[:5] == b"%PDF-"


def test_render_pdf_minimal_report_has_expected_sections(
    tmp_path, manifest, metadata, not_evaluated_fusion
) -> None:
    report = minimal_report(manifest, metadata, not_evaluated_fusion)
    out = render_pdf_report(report, tmp_path / "report.pdf")
    text = _extract_text(out)

    for marker in (
        "Executive Summary",
        "Model Identity",
        "Integrity Verification",
        "Model Metadata",
        "Weight Analysis",
        "Differential Analysis",
        "Behavioral Analysis",
        "Findings",
        "Recommendations",
        "Limitations",
        "Technical Appendix",
    ):
        assert marker in text, f"missing section: {marker}"


def test_render_pdf_reports_not_evaluated_sections_when_disabled(
    tmp_path, manifest, metadata, not_evaluated_fusion
) -> None:
    report = minimal_report(manifest, metadata, not_evaluated_fusion)
    out = render_pdf_report(report, tmp_path / "report.pdf")
    text = _extract_text(out)
    assert "Not evaluated" in text
    assert "No detectors were evaluated" in text


def test_render_pdf_with_full_report_includes_findings(tmp_path, manifest, metadata) -> None:
    import numpy as np

    from neurofence.config.schema import RiskWeights
    from neurofence.fusion.combine import fuse_evidence
    from neurofence.fusion.scores import SubScore
    from neurofence.weight_forensics.anomaly import AnomalyDetectionSummary, LayerAnomalyResult
    from neurofence.weight_forensics.pipeline import TensorForensicsRecord, WeightForensicsResult
    from neurofence.weight_forensics.statistics import compute_tensor_statistics

    stats = compute_tensor_statistics(np.zeros((4, 4)))
    outlier = LayerAnomalyResult(layer_name="suspicious_layer", is_outlier=True, anomaly_score=0.9)
    summary = AnomalyDetectionSummary(
        status="ml_based", sample_count=10, feature_count=5, results=[outlier]
    )
    weight_result = WeightForensicsResult(
        tensor_count=1,
        tensors=[TensorForensicsRecord(name="suspicious_layer", statistics=stats)],
        anomaly_detection=summary,
    )
    fusion = fuse_evidence(
        {
            "integrity": SubScore(name="integrity", status="not_evaluated"),
            "weight_anomaly": SubScore(name="weight_anomaly", status="evaluated", score=80.0),
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

    out = render_pdf_report(report, tmp_path / "report.pdf")
    text = _extract_text(out)

    assert "suspicious_layer" in text
    assert "WEIGHT-001" in text
    assert fusion.risk_label in text


def test_render_pdf_special_characters_in_trigger_do_not_crash(
    tmp_path, manifest, metadata, not_evaluated_fusion
) -> None:
    # ReportLab's Paragraph uses a minimal XML/HTML-like markup -- text
    # containing '<', '>', '&' must be escaped or it will raise, not just
    # render oddly. This exercises that path via a trigger phrase.
    from neurofence.fuzzing.trigger_discovery import TriggerCandidateResult

    report = build_report(
        model_path="/fake/path",
        manifest=manifest,
        metadata=metadata,
        evidence_fusion=not_evaluated_fusion,
        trigger_candidates=[
            TriggerCandidateResult(
                trigger="<script>alert('x')</script> & more",
                num_prompts_tested=2,
                mean_anomaly_score=0.9,
                min_anomaly_score=0.8,
                max_anomaly_score=1.0,
                consistent=True,
                evidence=[],
            )
        ],
    )
    out = render_pdf_report(report, tmp_path / "report.pdf")
    assert out.exists()


def test_render_pdf_unbalanced_markup_in_architecture_does_not_crash(
    tmp_path, manifest, metadata, not_evaluated_fusion
) -> None:
    # Regression: an untrusted model's config.json can put arbitrary text
    # in "architectures" -- this reached ReportLab's Paragraph markup
    # parser unescaped and raised ValueError("Parse error: saw </para>
    # instead of expected </b>") on unbalanced '<'/'&', aborting the
    # entire report. Model-controlled text must never be able to crash
    # report generation.
    metadata_with_hostile_arch = metadata.model_copy(
        update={"architecture": "Model<3 & unclosed <b>bold", "model_type": "type & <tag"}
    )
    report = build_report(
        model_path="/fake/path",
        manifest=manifest,
        metadata=metadata_with_hostile_arch,
        evidence_fusion=not_evaluated_fusion,
    )
    out = render_pdf_report(report, tmp_path / "report.pdf")
    assert out.exists()
    text = _extract_text(out)
    assert "Model<3" in text or "Model" in text  # rendered as literal text, not parsed as markup
