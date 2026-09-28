# Detection Methodology: Scoring and Fusion

## Sub-scores

Each detector's raw output is converted to a `SubScore` (0-100, or
`not_evaluated` with a reason) by `neurofence.fusion.scores`:

| Sub-score | Source | Not evaluated when |
|---|---|---|
| `integrity` | manifest re-verification against a stored baseline | no baseline manifest supplied (currently always, from the CLI -- see `docs/limitations.md`) |
| `weight_anomaly` | `weight_forensics.anomaly.AnomalyDetectionSummary` | weight forensics disabled or produced no results |
| `behavioral` | `behavioral` comparison records vs. `--reference` | `--behavioral` not combined with `--reference` |
| `activation` | per-prompt `activation.anomaly` summaries | `--activations` not enabled |
| `trigger` | `fuzzing.trigger_discovery` results | no `--trigger` phrases supplied |

Every scaling factor (e.g. "weight_anomaly = 100 * outlier_fraction * 5")
is a documented, inspectable heuristic in `neurofence/fusion/scores.py` --
not derived from a labeled dataset. They are deliberately conservative:
even a small fraction of flagged layers pushes the score well above LOW,
since a genuinely healthy model is expected to have few or no outlier
layers.

## Anomaly Score

```
anomaly_score = sum(weight[d] * score[d] for d in evaluated_detectors)
                / sum(weight[d] for d in evaluated_detectors)
```

A plain weighted average, **renormalized over only the detectors that
actually ran** -- a detector that wasn't run does not silently count as
"clean" (weight 0 in the numerator without being removed from the
denominator would do exactly that). This answers: *how statistically
unusual does this model look overall, given what was checked?* One very
loud detector can push this high on its own.

## Threat Confidence

```
elevated = {d : score[d] >= 40 for d in evaluated_detectors}
agreement_ratio = len(elevated) / len(evaluated_detectors)

if not elevated:
    threat_confidence = 0
else:
    mean_elevated = mean(score[d] for d in elevated)
    confidence_factor = 0.4 + 0.6 * agreement_ratio
    threat_confidence = mean_elevated * confidence_factor
```

This is the numeric expression of the project's core rule: **never make a
security determination from a single detector.** `confidence_factor`
ranges from 0.4 (only a small minority of evaluated detectors agree) to
1.0 (all of them do) -- so an isolated elevated signal is discounted to
well below its own severity, while several independent detectors elevated
together score close to their shared severity.

Worked example (from `tests/fusion/test_combine.py`):

- Five detectors evaluated, only `weight_anomaly` elevated at 90, the
  rest quiet (0-5): `anomaly_score` is still meaningfully raised by the
  one strong signal, but `threat_confidence` is discounted well below 54
  (90 * 0.6) because nothing corroborates it.
- All five detectors elevated at 85: `anomaly_score = 85`,
  `threat_confidence = 85` exactly (full agreement, `confidence_factor =
  1.0`).

## Risk Classification

`threat_confidence` (not `anomaly_score`) drives the LOW/MEDIUM/HIGH/
CRITICAL classification, via `neurofence.config.schema.RiskConfig.classify`
-- an explicit design choice: the risk *label* should track "how strong
is the combined evidence for malicious modification," which is what
Threat Confidence measures, while Anomaly Score is reported alongside for
context (a model can be highly unusual with weak evidence of malice, or
vice versa -- both are useful, distinct signals for a human reviewer).
Risk bands are project-defined defaults (`neurofence/config/default.yaml`),
not a universal cybersecurity standard, and are fully configurable.

## Findings

`neurofence.reporting.report.generate_findings` derives human-readable
findings from the same underlying detector outputs used for scoring (not
from the scores themselves) -- each finding cites specific evidence
(layer name, L2 diff, mean anomaly score, consensus votes) and a targeted
recommendation, sorted most-severe-first. Findings are the "why" behind a
risk classification; the score alone is never presented without them.
