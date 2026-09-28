"""The canonical report data model: everything a NeuroFence scan produced,
assembled into one structure that both the JSON and PDF renderers read
from. Building this is the CLI's job (it already has every live object in
scope after running `scan`); this module only defines the shape and
derives findings/recommendations/limitations from it.
"""

from __future__ import annotations

from datetime import UTC, datetime

from pydantic import BaseModel

from neurofence import __version__
from neurofence.acquisition.manifest import ModelManifest
from neurofence.acquisition.metadata import ModelMetadata
from neurofence.behavioral.pipeline import BehavioralComparisonRecord, BehavioralTestResult
from neurofence.fusion.combine import FusionResult
from neurofence.fuzzing.trigger_discovery import TriggerCandidateResult
from neurofence.weight_forensics.differential import DifferentialResult
from neurofence.weight_forensics.pipeline import WeightForensicsResult


class Finding(BaseModel):
    finding_id: str
    severity: str  # "LOW" | "MEDIUM" | "HIGH" | "CRITICAL"
    summary: str
    evidence: list[str]
    affected: str  # layer name, test id, or trigger phrase
    confidence: str  # human-readable, not a fabricated number beyond what's measured
    recommendation: str


class ScanReport(BaseModel):
    generated_at: str
    neurofence_version: str
    model_path: str

    manifest: ModelManifest
    metadata: ModelMetadata

    weight_forensics: WeightForensicsResult | None = None
    differential: DifferentialResult | None = None
    behavioral: list[BehavioralTestResult] | None = None
    behavioral_comparison: list[BehavioralComparisonRecord] | None = None
    trigger_candidates: list[TriggerCandidateResult] | None = None
    activation_summary: dict[str, object] | None = None

    evidence_fusion: FusionResult

    findings: list[Finding] = []
    recommendations: list[str] = []
    limitations: list[str] = []


_SEVERITY_BY_SCORE_THRESHOLDS = (
    (76, "CRITICAL"),
    (51, "HIGH"),
    (26, "MEDIUM"),
    (0, "LOW"),
)


def _severity_for_score(score: float) -> str:
    for threshold, label in _SEVERITY_BY_SCORE_THRESHOLDS:
        if score >= threshold:
            return label
    return "LOW"  # unreachable given the 0 floor above, kept for exhaustiveness


def _weight_findings(result: WeightForensicsResult | None) -> list[Finding]:
    if result is None:
        return []
    findings = []
    for i, layer_result in enumerate(
        r for r in result.anomaly_detection.results if r.is_outlier
    ):
        findings.append(
            Finding(
                finding_id=f"WEIGHT-{i + 1:03d}",
                severity=_severity_for_score(layer_result.anomaly_score * 100),
                summary=f"Layer '{layer_result.layer_name}' has an anomalous weight profile.",
                evidence=[
                    f"Detection method: {result.anomaly_detection.status} "
                    f"({', '.join(result.anomaly_detection.methods_used)}).",
                    (
                        f"Votes: {layer_result.votes}"
                        if layer_result.votes
                        else "Statistical fallback flagged this layer."
                    ),
                ],
                affected=layer_result.layer_name,
                confidence=(
                    "Multiple independent methods agree"
                    if sum(layer_result.votes.values()) >= 2
                    else "Single method / statistical fallback -- weaker evidence"
                ),
                recommendation=(
                    f"Manually inspect layer '{layer_result.layer_name}' and, if a trusted "
                    "reference model is available, run differential analysis against it."
                ),
            )
        )
    return findings


def _differential_findings(result: DifferentialResult | None) -> list[Finding]:
    if result is None or result.status != "compared":
        return []
    findings = []
    for i, name in enumerate(result.most_affected_layers[:5]):
        diff = next((d for d in result.tensor_diffs if d.name == name), None)
        if diff is None or diff.l2_diff is None:
            continue
        findings.append(
            Finding(
                finding_id=f"DIFF-{i + 1:03d}",
                severity=_severity_for_score(min(100.0, (diff.percent_changed or 0) * 100 * 3)),
                summary=f"Tensor '{name}' differs substantially from the reference model.",
                evidence=[
                    f"L2 diff: {diff.l2_diff:.4g}",
                    f"Elements changed: {(diff.percent_changed or 0):.1%}",
                    f"Max absolute diff: {diff.max_abs_diff:.4g}" if diff.max_abs_diff else "",
                ],
                affected=name,
                confidence="Direct comparison against a trusted reference model.",
                recommendation=(
                    f"Confirm whether the change to '{name}' is an expected part of fine-tuning "
                    "or update history; if not, treat as evidence of tampering."
                ),
            )
        )
    return findings


def _trigger_findings(results: list[TriggerCandidateResult] | None) -> list[Finding]:
    if not results:
        return []
    findings = []
    for i, r in enumerate(res for res in results if res.consistent):
        findings.append(
            Finding(
                finding_id=f"TRIGGER-{i + 1:03d}",
                severity=_severity_for_score(r.mean_anomaly_score * 100),
                summary=f"Candidate trigger phrase '{r.trigger}' shows consistent divergence.",
                evidence=[
                    f"Mean anomaly score: {r.mean_anomaly_score:.2f} over "
                    f"{r.num_prompts_tested} prompt(s).",
                    f"Range: {r.min_anomaly_score:.2f}-{r.max_anomaly_score:.2f}.",
                ],
                affected=r.trigger,
                confidence=(
                    "Consistent across a majority of tested prompts -- moderate evidence, "
                    "not proof."
                ),
                recommendation=(
                    f"Manually test the model with the phrase '{r.trigger}' across more diverse "
                    "prompts to confirm whether this is a genuine trigger-conditioned behavior."
                ),
            )
        )
    return findings


def generate_findings(
    weight_forensics: WeightForensicsResult | None,
    differential: DifferentialResult | None,
    trigger_candidates: list[TriggerCandidateResult] | None,
) -> list[Finding]:
    return [
        *_weight_findings(weight_forensics),
        *_differential_findings(differential),
        *_trigger_findings(trigger_candidates),
    ]


def generate_recommendations(fusion: FusionResult) -> list[str]:
    if fusion.status == "not_evaluated":
        return [
            "No detectors were run. Re-scan with --weights, --behavioral, --trigger, and/or "
            "--activations enabled before drawing any conclusion about this model."
        ]

    by_label = {
        "CRITICAL": [
            "Do not deploy this model until every finding below has been manually reviewed.",
            "Obtain a trusted reference model (original checkpoint, vendor-signed release) and "
            "re-run with --reference for differential analysis if not already done.",
            "Treat this model as potentially compromised until proven otherwise.",
        ],
        "HIGH": [
            "Manually review all findings below before deploying this model.",
            "Consider re-running with additional detectors enabled (--behavioral, --trigger, "
            "--activations) if not all were used, to corroborate or rule out the current evidence.",
        ],
        "MEDIUM": [
            "Review the findings below; the evidence is not yet strong enough for a confident "
            "determination either way.",
            "Consider testing with a broader prompt set or additional candidate triggers.",
        ],
        "LOW": [
            "No significant evidence of tampering was found by the detectors that ran.",
            "This is not proof the model is safe -- see Limitations. Continue routine monitoring.",
        ],
    }
    return by_label.get(fusion.risk_label, [])


_SEVERITY_ORDER = {"LOW": 0, "MEDIUM": 1, "HIGH": 2, "CRITICAL": 3}


def build_report(
    model_path: str,
    manifest: ModelManifest,
    metadata: ModelMetadata,
    evidence_fusion: FusionResult,
    weight_forensics: WeightForensicsResult | None = None,
    differential: DifferentialResult | None = None,
    behavioral: list[BehavioralTestResult] | None = None,
    behavioral_comparison: list[BehavioralComparisonRecord] | None = None,
    trigger_candidates: list[TriggerCandidateResult] | None = None,
    activation_summary: dict[str, object] | None = None,
) -> ScanReport:
    findings = generate_findings(weight_forensics, differential, trigger_candidates)
    findings.sort(key=lambda f: _SEVERITY_ORDER.get(f.severity, 0), reverse=True)

    return ScanReport(
        generated_at=datetime.now(UTC).isoformat(),
        neurofence_version=__version__,
        model_path=model_path,
        manifest=manifest,
        metadata=metadata,
        weight_forensics=weight_forensics,
        differential=differential,
        behavioral=behavioral,
        behavioral_comparison=behavioral_comparison,
        trigger_candidates=trigger_candidates,
        activation_summary=activation_summary,
        evidence_fusion=evidence_fusion,
        findings=findings,
        recommendations=generate_recommendations(evidence_fusion),
        limitations=list(LIMITATIONS),
    )


LIMITATIONS = [
    "Absence of a detected anomaly does not prove a model is safe -- it means the detectors "
    "that ran found no evidence, which is a different (weaker) claim.",
    "Unknown attack techniques may evade every detector implemented here.",
    "Behavioral and trigger-discovery analysis depend entirely on the prompt/trigger coverage "
    "used for this scan; an untested prompt category or trigger phrase is not evaluated.",
    "Statistical weight anomalies can have entirely benign explanations (unusual but legitimate "
    "initialization, architecture-specific layers, fine-tuning artifacts).",
    "Reference-free detection (no --reference model supplied) is inherently weaker than "
    "differential comparison against a trusted baseline.",
    "Anomaly Score and Threat Confidence are project-defined heuristic scores, not calibrated "
    "probabilities and not derived from a labeled real-world dataset -- see "
    "neurofence.evaluation for measured detector performance on synthetic ground truth.",
]
