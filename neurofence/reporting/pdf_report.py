"""PDF rendering of a ScanReport via ReportLab.

Reads from the same ScanReport the JSON renderer uses -- see
neurofence.reporting.json_report -- so the two report formats can never
disagree about content, only presentation. Sections follow the project
spec: executive summary, model identity, integrity, metadata, weight
analysis, differential analysis, behavioral analysis, fuzzing results,
activation analysis, evidence fusion (anomaly score / threat confidence /
risk classification), findings, recommendations, limitations, technical
appendix.

Security note: ReportLab's Paragraph text is a small XML-like markup
language (<b>, <font>, ...), not plain text -- feeding it unescaped
arbitrary text raises a hard ValueError and aborts the whole report.
Everything in a ScanReport that ultimately originates from outside this
process (a scanned model's config.json architecture/model_type string, a
user-supplied --trigger phrase, a tensor/layer name) must be escaped with
`_esc()` before being interpolated into any Paragraph text. Only markup
this module writes itself (the literal <b>/<font> tags below) is safe to
leave unescaped.
"""

from __future__ import annotations

from pathlib import Path
from xml.sax.saxutils import escape as _xml_escape

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    ListFlowable,
    ListItem,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from neurofence.reporting.report import ScanReport

_RISK_COLORS = {
    "LOW": colors.HexColor("#2e7d32"),
    "MEDIUM": colors.HexColor("#f9a825"),
    "HIGH": colors.HexColor("#ef6c00"),
    "CRITICAL": colors.HexColor("#c62828"),
    "NOT_EVALUATED": colors.HexColor("#616161"),
}


def _esc(value: object) -> str:
    """Escape a value for safe interpolation into ReportLab Paragraph
    markup. Must be applied to every value that isn't a literal string
    this module wrote itself -- see module docstring.
    """
    return _xml_escape(str(value))


def _styles() -> dict:
    base = getSampleStyleSheet()
    base.add(ParagraphStyle("SectionHeading", parent=base["Heading1"], spaceBefore=18))
    base.add(ParagraphStyle("SubHeading", parent=base["Heading2"], spaceBefore=10))
    return base


def _kv_table(rows: list[tuple[str, str]], styles: dict) -> Table:
    # `k` (the label) is always a literal string this module wrote; `v`
    # (the value) usually originates from report data and is escaped.
    data = [
        [Paragraph(f"<b>{k}</b>", styles["BodyText"]), Paragraph(_esc(v), styles["BodyText"])]
        for k, v in rows
    ]
    table = Table(data, colWidths=[2.0 * inch, 4.3 * inch])
    table.setStyle(
        TableStyle(
            [
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cccccc")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#f5f5f5")),
            ]
        )
    )
    return table


def _bullets(items: list[str], styles: dict) -> ListFlowable:
    return ListFlowable(
        [ListItem(Paragraph(_esc(item), styles["BodyText"])) for item in items],
        bulletType="bullet",
    )


def _executive_summary(report: ScanReport, styles: dict) -> list:
    fusion = report.evidence_fusion
    flow: list = [Paragraph("1. Executive Summary", styles["SectionHeading"])]

    if fusion.status == "not_evaluated":
        flow.append(Paragraph("No detectors were run against this model.", styles["BodyText"]))
        return flow

    risk_color = _RISK_COLORS.get(fusion.risk_label, colors.black)
    flow.append(
        Paragraph(
            f'Risk classification: <font color="{risk_color.hexval()}">'
            f"<b>{_esc(fusion.risk_label)}</b></font>",
            styles["BodyText"],
        )
    )
    flow.append(
        _kv_table(
            [
                ("Anomaly Score", f"{fusion.anomaly_score:.1f} / 100"),
                ("Threat Confidence", f"{fusion.threat_confidence:.1f} / 100"),
                ("Detectors evaluated", str(fusion.num_signals_evaluated)),
                ("Detectors elevated", str(fusion.num_signals_elevated)),
            ],
            styles,
        )
    )
    flow.append(Spacer(1, 8))
    flow.append(
        Paragraph(
            "Anomaly Score measures how statistically unusual the model appears overall. "
            "Threat Confidence measures how strongly the combined evidence, across "
            "independent detectors, is consistent with deliberate malicious modification -- "
            "it is deliberately lower than the Anomaly Score when only one detector is "
            "elevated and nothing corroborates it. <b>Neither score is proof of anything; "
            "both are inputs to a human decision.</b>",
            styles["BodyText"],
        )
    )
    return flow


def _model_identity(report: ScanReport, styles: dict) -> list:
    m = report.metadata
    return [
        Paragraph("2. Model Identity", styles["SectionHeading"]),
        _kv_table(
            [
                ("Model path", report.model_path),
                ("Architecture", m.architecture or "unknown"),
                ("Model type", m.model_type or "unknown"),
                ("Parameter count", f"{m.parameter_count:,}" if m.parameter_count else "unknown"),
                ("Layer count", str(m.layer_count) if m.layer_count is not None else "unknown"),
                ("Weight format", m.weight_format),
            ],
            styles,
        ),
    ]


def _integrity_section(report: ScanReport, styles: dict) -> list:
    manifest = report.manifest
    flow = [
        Paragraph("3. Integrity Verification", styles["SectionHeading"]),
        _kv_table(
            [
                ("Files in manifest", str(manifest.file_count)),
                ("Total size", f"{manifest.total_size_bytes:,} bytes"),
                ("Manifest generated at", manifest.generated_at),
            ],
            styles,
        ),
    ]
    integrity_score = report.evidence_fusion.sub_scores.get("integrity")
    if integrity_score is not None:
        flow.append(Spacer(1, 6))
        flow.append(Paragraph(f"<b>Status:</b> {_esc(integrity_score.status)}", styles["BodyText"]))
        if integrity_score.reasons:
            flow.append(_bullets(integrity_score.reasons, styles))
    return flow


def _metadata_section(report: ScanReport, styles: dict) -> list:
    flow = [Paragraph("4. Model Metadata", styles["SectionHeading"])]
    if report.metadata.warnings:
        flow.append(Paragraph("Warnings raised during metadata extraction:", styles["BodyText"]))
        flow.append(_bullets(report.metadata.warnings, styles))
    else:
        flow.append(
            Paragraph("No warnings raised during metadata extraction.", styles["BodyText"])
        )
    return flow


def _weight_analysis_section(report: ScanReport, styles: dict) -> list:
    flow = [Paragraph("5. Weight Analysis", styles["SectionHeading"])]
    wf = report.weight_forensics
    if wf is None:
        flow.append(
            Paragraph(
                "Not evaluated (weight forensics was disabled for this scan).", styles["BodyText"]
            )
        )
        return flow

    flow.append(
        _kv_table(
            [
                ("Tensors analyzed", str(wf.tensor_count)),
                ("Anomaly detection method", wf.anomaly_detection.status),
                (
                    "Layers flagged",
                    str(sum(1 for r in wf.anomaly_detection.results if r.is_outlier)),
                ),
            ],
            styles,
        )
    )
    if wf.anomaly_detection.reason:
        flow.append(Spacer(1, 6))
        flow.append(Paragraph(_esc(wf.anomaly_detection.reason), styles["BodyText"]))
    return flow


def _differential_section(report: ScanReport, styles: dict) -> list:
    flow = [Paragraph("6. Differential Analysis", styles["SectionHeading"])]
    diff = report.differential
    if diff is None:
        flow.append(
            Paragraph("Not evaluated (no --reference model supplied).", styles["BodyText"])
        )
        return flow
    if diff.status == "reference_incompatible":
        flow.append(Paragraph(f"REFERENCE_INCOMPATIBLE: {_esc(diff.reason)}", styles["BodyText"]))
        return flow

    flow.append(
        _kv_table(
            [
                ("Tensors compared", str(diff.compared_count)),
                ("Shape mismatches", str(len(diff.shape_mismatches))),
                ("Most affected layers", ", ".join(diff.most_affected_layers[:5]) or "none"),
            ],
            styles,
        )
    )
    return flow


def _behavioral_section(report: ScanReport, styles: dict) -> list:
    flow = [Paragraph("7. Behavioral Analysis", styles["SectionHeading"])]
    if report.behavioral is None:
        flow.append(Paragraph("Not evaluated (--behavioral was not enabled).", styles["BodyText"]))
        return flow
    flow.append(Paragraph(f"{len(report.behavioral)} prompt(s) tested.", styles["BodyText"]))
    if report.behavioral_comparison is not None:
        changed = sum(1 for c in report.behavioral_comparison if c.comparison.refusal_changed)
        flow.append(
            Paragraph(
                f"Compared against --reference model output: refusal behavior changed on "
                f"{changed}/{len(report.behavioral_comparison)} prompt(s).",
                styles["BodyText"],
            )
        )
    return flow


def _fuzzing_section(report: ScanReport, styles: dict) -> list:
    flow = [Paragraph("8. Fuzzing / Trigger Discovery Results", styles["SectionHeading"])]
    if not report.trigger_candidates:
        flow.append(
            Paragraph("Not evaluated (no --trigger phrases supplied).", styles["BodyText"])
        )
        return flow
    header = ("Trigger", "Mean score", "Consistent")
    body_rows = [
        (r.trigger, f"{r.mean_anomaly_score:.2f}", "yes" if r.consistent else "no")
        for r in report.trigger_candidates[:10]
    ]
    table = Table(
        [[Paragraph(f"<b>{c}</b>", styles["BodyText"]) for c in header]]
        + [[Paragraph(_esc(c), styles["BodyText"]) for c in row] for row in body_rows],
        colWidths=[2.5 * inch, 1.5 * inch, 1.5 * inch],
    )
    table.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cccccc"))]))
    flow.append(table)
    return flow


def _activation_section(report: ScanReport, styles: dict) -> list:
    flow = [Paragraph("9. Activation Analysis", styles["SectionHeading"])]
    if not report.activation_summary:
        flow.append(Paragraph("Not evaluated (--activations was not enabled).", styles["BodyText"]))
        return flow
    for key, value in report.activation_summary.items():
        flow.append(Paragraph(f"<b>{_esc(key)}:</b> {_esc(value)}", styles["BodyText"]))
    return flow


def _evidence_fusion_section(report: ScanReport, styles: dict) -> list:
    fusion = report.evidence_fusion
    flow = [
        Paragraph(
            "10-13. Evidence Fusion, Anomaly Score, Threat Confidence, Risk",
            styles["SectionHeading"],
        )
    ]
    if fusion.status == "not_evaluated":
        flow.append(Paragraph("No detectors were evaluated.", styles["BodyText"]))
        return flow
    for line in fusion.explanation:
        flow.append(Paragraph(_esc(line), styles["BodyText"]))
        flow.append(Spacer(1, 4))
    return flow


def _findings_section(report: ScanReport, styles: dict) -> list:
    flow = [Paragraph("14. Findings", styles["SectionHeading"])]
    if not report.findings:
        flow.append(Paragraph("No findings were raised.", styles["BodyText"]))
        return flow
    for f in report.findings:
        color = _RISK_COLORS.get(f.severity, colors.black)
        flow.append(
            Paragraph(
                f"[{_esc(f.finding_id)}] "
                f'<font color="{color.hexval()}"><b>{_esc(f.severity)}</b></font> '
                f"-- {_esc(f.summary)}",
                styles["SubHeading"],
            )
        )
        flow.append(Paragraph(f"<b>Affected:</b> {_esc(f.affected)}", styles["BodyText"]))
        flow.append(Paragraph(f"<b>Confidence:</b> {_esc(f.confidence)}", styles["BodyText"]))
        flow.append(_bullets(f.evidence, styles))
        flow.append(
            Paragraph(f"<b>Recommendation:</b> {_esc(f.recommendation)}", styles["BodyText"])
        )
        flow.append(Spacer(1, 6))
    return flow


def _recommendations_section(report: ScanReport, styles: dict) -> list:
    flow = [Paragraph("15. Recommendations", styles["SectionHeading"])]
    flow.append(_bullets(report.recommendations, styles))
    return flow


def _limitations_section(report: ScanReport, styles: dict) -> list:
    flow = [Paragraph("16. Limitations", styles["SectionHeading"])]
    flow.append(_bullets(report.limitations, styles))
    return flow


def _appendix_section(report: ScanReport, styles: dict) -> list:
    return [
        Paragraph("17. Technical Appendix", styles["SectionHeading"]),
        _kv_table(
            [
                ("NeuroFence version", report.neurofence_version),
                ("Report generated at", report.generated_at),
                (
                    "Manifest SHA-256 of first file",
                    report.manifest.files[0].sha256 if report.manifest.files else "n/a",
                ),
            ],
            styles,
        ),
    ]


def render_pdf_report(report: ScanReport, output_path: str | Path) -> Path:
    target = Path(output_path)
    styles = _styles()
    doc = SimpleDocTemplate(
        str(target),
        pagesize=letter,
        title="NeuroFence Scan Report",
        author="NeuroFence",
    )

    story: list = [Paragraph("NeuroFence Scan Report", styles["Title"])]
    for section_builder in (
        _executive_summary,
        _model_identity,
        _integrity_section,
        _metadata_section,
        _weight_analysis_section,
        _differential_section,
        _behavioral_section,
        _fuzzing_section,
        _activation_section,
        _evidence_fusion_section,
        _findings_section,
        _recommendations_section,
        _limitations_section,
        _appendix_section,
    ):
        story.extend(section_builder(report, styles))

    doc.build(story)
    return target
