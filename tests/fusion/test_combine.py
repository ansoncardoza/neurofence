from __future__ import annotations

from neurofence.config.schema import RiskWeights
from neurofence.fusion.combine import fuse_evidence
from neurofence.fusion.scores import SubScore

# defaults: integrity .15, weight_anomaly .25, behavioral .25, activation .20, trigger .15
WEIGHTS = RiskWeights()
DETECTORS = ("integrity", "weight_anomaly", "behavioral", "activation", "trigger")


def _score(name: str, value: float | None) -> SubScore:
    if value is None:
        return SubScore(name=name, status="not_evaluated")
    return SubScore(name=name, status="evaluated", score=value, reasons=[f"{name} test reason"])


def _uniform(value: float | None) -> dict[str, SubScore]:
    return {name: _score(name, value) for name in DETECTORS}


def test_nothing_evaluated_returns_not_evaluated() -> None:
    result = fuse_evidence(_uniform(None), WEIGHTS)
    assert result.status == "not_evaluated"
    assert result.anomaly_score is None
    assert result.threat_confidence is None
    assert result.risk_label == "NOT_EVALUATED"


def test_single_isolated_high_signal_low_confidence() -> None:
    """The spec's key example: one detector screams (weight_anomaly=90),
    everything else evaluated and quiet -> anomaly score should be
    moderately high (one strong contributor) but threat confidence should
    be clearly discounted relative to that 90, since nothing corroborates
    it.
    """
    scores = {
        "integrity": _score("integrity", 5.0),
        "weight_anomaly": _score("weight_anomaly", 90.0),
        "behavioral": _score("behavioral", 3.0),
        "activation": _score("activation", 2.0),
        "trigger": _score("trigger", 0.0),
    }
    result = fuse_evidence(scores, WEIGHTS)

    assert result.status == "evaluated"
    assert result.num_signals_elevated == 1
    assert result.threat_confidence < result.anomaly_score * 2  # sanity: not wildly inflated
    assert result.threat_confidence < 90.0 * 0.6  # discounted well below the raw elevated score
    assert result.anomaly_score > 15.0  # the one strong signal still visibly moves it


def test_multiple_corroborating_signals_high_confidence() -> None:
    """All five detectors independently elevated at similar severity ->
    threat confidence should be close to that shared severity (full
    agreement), clearly higher than the isolated-signal case above even
    though the anomaly score is similar in shape.
    """
    result = fuse_evidence(_uniform(85.0), WEIGHTS)

    assert result.status == "evaluated"
    assert result.num_signals_elevated == 5
    assert result.agreement_ratio == 1.0
    assert result.anomaly_score == 85.0
    assert result.threat_confidence == 85.0  # full agreement: confidence == mean elevated severity


def test_isolated_vs_corroborated_same_severity_different_confidence() -> None:
    isolated = {
        "integrity": _score("integrity", 0.0),
        "weight_anomaly": _score("weight_anomaly", 80.0),
        "behavioral": _score("behavioral", 0.0),
        "activation": _score("activation", 0.0),
        "trigger": _score("trigger", 0.0),
    }

    isolated_result = fuse_evidence(isolated, WEIGHTS)
    corroborated_result = fuse_evidence(_uniform(80.0), WEIGHTS)

    assert isolated_result.threat_confidence < corroborated_result.threat_confidence


def test_no_elevated_signals_zero_confidence() -> None:
    result = fuse_evidence(_uniform(10.0), WEIGHTS)
    assert result.threat_confidence == 0.0
    assert result.anomaly_score == 10.0
    assert result.risk_label == "LOW"


def test_partial_evaluation_renormalizes_weights() -> None:
    # Only weight_anomaly evaluated, at 60 -- anomaly_score should equal 60
    # exactly (100% of the renormalized weight), not 60 * 0.25 diluted by
    # missing detectors as if they were zero.
    scores = _uniform(None)
    scores["weight_anomaly"] = _score("weight_anomaly", 60.0)
    result = fuse_evidence(scores, WEIGHTS)
    assert result.anomaly_score == 60.0
    assert result.num_signals_evaluated == 1


def test_zero_weight_detectors_fallback_unweighted_average() -> None:
    zero_weights = RiskWeights(
        integrity=0.0, weight_anomaly=1.0, behavioral=0.0, activation=0.0, trigger=0.0
    )
    # Only "integrity" is evaluated but its configured weight is 0 -- total
    # weight among evaluated detectors is 0, so this must fall back to an
    # unweighted average instead of dividing by zero.
    scores = _uniform(None)
    scores["integrity"] = _score("integrity", 100.0)
    result = fuse_evidence(scores, zero_weights)
    assert result.anomaly_score == 100.0


def test_risk_label_matches_risk_config_bands() -> None:
    result = fuse_evidence(_uniform(95.0), WEIGHTS)
    assert result.risk_label == "CRITICAL"


def test_scores_never_exceed_bounds() -> None:
    result = fuse_evidence(_uniform(100.0), WEIGHTS)
    assert 0.0 <= result.anomaly_score <= 100.0
    assert 0.0 <= result.threat_confidence <= 100.0


def test_explanation_includes_all_evaluated_detectors() -> None:
    scores = _uniform(None)
    scores["integrity"] = _score("integrity", 50.0)
    scores["weight_anomaly"] = _score("weight_anomaly", 60.0)
    result = fuse_evidence(scores, WEIGHTS)
    joined = " ".join(result.explanation)
    assert "integrity" in joined
    assert "weight_anomaly" in joined
