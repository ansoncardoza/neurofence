"""Evidence fusion: combines independent SubScores into an Anomaly Score
and a Threat Confidence -- two deliberately different numbers.

- **Anomaly Score**: how statistically unusual the model looks overall. A
  weighted average of every sub-score that was actually evaluated
  (weights renormalized over just those, so a detector that wasn't run
  does not silently count as "clean"). One very unusual signal can push
  this high on its own.

- **Threat Confidence**: how strong the *combined* evidence is that the
  anomalies reflect deliberate malicious modification, per the project's
  core principle of never deciding from a single detector. It rewards
  agreement: several independent detectors elevated together yields
  materially higher confidence than one isolated elevated detector at the
  same severity, even though both cases could produce the same Anomaly
  Score. Neither score is proof of anything; both are inputs to a human
  decision.

Neither formula is derived from a labeled dataset -- both are explainable,
documented heuristics (see docs/limitations, added in a later milestone).
"""

from __future__ import annotations

from pydantic import BaseModel

from neurofence.config.schema import RiskConfig, RiskWeights
from neurofence.fusion.scores import SubScore

# A sub-score at or above this is "elevated" for the purpose of measuring
# agreement across detectors. Below it, a detector is treated as reporting
# nothing noteworthy.
ELEVATED_THRESHOLD = 40.0

# Threat confidence scales the mean of elevated scores by a factor between
# CONFIDENCE_MIN_FACTOR (only a small minority of evaluated detectors
# agree) and 1.0 (all evaluated detectors agree) -- i.e. even full
# agreement never *inflates* confidence above the elevated detectors' own
# mean severity, but partial/isolated agreement discounts it.
_CONFIDENCE_MIN_FACTOR = 0.4


class FusionResult(BaseModel):
    status: str  # "evaluated" | "not_evaluated"
    anomaly_score: float | None = None
    threat_confidence: float | None = None
    risk_label: str
    num_signals_evaluated: int
    num_signals_elevated: int
    agreement_ratio: float | None = None
    sub_scores: dict[str, SubScore]
    explanation: list[str] = []


def fuse_evidence(
    sub_scores: dict[str, SubScore],
    weights: RiskWeights,
    risk_config: RiskConfig | None = None,
) -> FusionResult:
    risk_config = risk_config or RiskConfig(weights=weights)
    weight_by_name = {
        "integrity": weights.integrity,
        "weight_anomaly": weights.weight_anomaly,
        "behavioral": weights.behavioral,
        "activation": weights.activation,
        "trigger": weights.trigger,
    }

    evaluated = {name: s for name, s in sub_scores.items() if s.status == "evaluated"}
    # Invariant: SubScore.score is non-None whenever status == "evaluated"
    # (enforced by the neurofence.fusion.scores extractors); narrow the
    # type here once instead of ignoring the Optional at every use site.
    evaluated_score: dict[str, float] = {
        name: s.score for name, s in evaluated.items() if s.score is not None
    }

    if not evaluated:
        return FusionResult(
            status="not_evaluated",
            risk_label="NOT_EVALUATED",
            num_signals_evaluated=0,
            num_signals_elevated=0,
            sub_scores=sub_scores,
            explanation=["No detectors were evaluated -- nothing to fuse."],
        )

    total_weight = sum(weight_by_name.get(name, 0.0) for name in evaluated_score)
    if total_weight <= 0:
        # All evaluated detectors happen to have zero configured weight;
        # fall back to an unweighted average rather than dividing by zero.
        anomaly_score = sum(evaluated_score.values()) / len(evaluated_score)
    else:
        anomaly_score = (
            sum(weight_by_name.get(name, 0.0) * score for name, score in evaluated_score.items())
            / total_weight
        )

    elevated = {
        name: score for name, score in evaluated_score.items() if score >= ELEVATED_THRESHOLD
    }
    agreement_ratio = len(elevated) / len(evaluated_score)

    if not elevated:
        threat_confidence = 0.0
    else:
        mean_elevated = sum(elevated.values()) / len(elevated)
        remaining = 1.0 - _CONFIDENCE_MIN_FACTOR
        confidence_factor = _CONFIDENCE_MIN_FACTOR + remaining * agreement_ratio
        threat_confidence = mean_elevated * confidence_factor

    risk_label = risk_config.classify(threat_confidence)

    explanation = [
        f"{len(evaluated)}/{len(sub_scores)} detector(s) evaluated; "
        f"{len(elevated)} elevated (score >= {ELEVATED_THRESHOLD:.0f}).",
    ]
    for name, s in sorted(evaluated.items(), key=lambda kv: kv[1].score or 0.0, reverse=True):
        explanation.append(f"[{name}] score={s.score:.1f}: {'; '.join(s.reasons)}")

    return FusionResult(
        status="evaluated",
        anomaly_score=float(anomaly_score),
        threat_confidence=float(threat_confidence),
        risk_label=risk_label,
        num_signals_evaluated=len(evaluated),
        num_signals_elevated=len(elevated),
        agreement_ratio=agreement_ratio,
        sub_scores=sub_scores,
        explanation=explanation,
    )
