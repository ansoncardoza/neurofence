# Evaluation

All numbers below are **measured**, reproducible with:

```bash
neurofence benchmark
# or, for the exact numbers below:
neurofence benchmark --n-clean 15 --n-poisoned-per-kind 3
```

(seed 1337, the default). Nothing here is invented or estimated -- see
`neurofence.evaluation.benchmark`, which times and scores the detector
against freshly generated synthetic data on every call. If you get
different numbers, that's real: report an issue, don't paper over it.

## Weight anomaly detector

Dataset: 30 synthetic models (15 clean, 3 poisoned per perturbation kind
x 5 kinds), 8 tensors each (see `neurofence.attack_lab.synthetic_models`).

| Metric | Value |
|---|---|
| Accuracy (binary: any layer flagged) | 0.567 |
| Precision | 0.536 |
| Recall | 1.0 |
| F1 | 0.698 |
| ROC-AUC | 0.698 |
| PR-AUC | 0.629 |
| **Layer localization rate** (correct layer flagged, poisoned samples only) | **0.8** |
| Runtime | ~5.4s for 30 samples |

### Reading these numbers honestly

The binary accuracy figure is a **permissive, misleading-if-read-alone**
metric: an unsupervised, `contamination=0.1` Isolation Forest/LOF flags
roughly one layer out of eight "by construction," largely independent of
whether anything is actually anomalous (see `docs/limitations.md`). That
mechanically produces perfect recall (poisoned samples always have
*something* flagged) alongside mediocre precision (so do many clean
samples).

**`layer_localization_rate` is the metric that actually answers "does the
detector work":** does it flag the *specific* tensor that was tampered
with, not just "something"? Breaking it down by attack type:

| Perturbation kind | Correctly localized |
|---|---|
| `localized` (single-element, magnitude 50) | 3/3 (100%) |
| `distributed` (30% of elements, noise scale 0.5) | 3/3 (100%) |
| `layer_level` (whole layer x3 scale) | 3/3 (100%) |
| `neuron_level` (one row, magnitude 10) | 3/3 (100%) |
| `low_magnitude` (50% of elements, noise scale 0.01 -- comparable to normal weight scale) | 0/3 (0%) |

This is exactly the expected shape of result: large, structural attacks
are reliably localized; a perturbation deliberately sized to blend into
normal weight noise is not caught by statistical layer-comparison alone.
That is a genuine, disclosed limitation of this detection method (see
`docs/limitations.md`), not a bug to be tuned away by lowering
detection thresholds until it "passes" -- doing so would just move false
negatives into false positives on clean models instead.

## Trigger detector

Two ground-truth scripted experiments (`neurofence.attack_lab.trigger_experiments`):
one deterministic backdoor simulator with a planted trigger phrase, one
clean simulator with none.

| Metric | Value |
|---|---|
| Accuracy | 1.0 (2/2) |
| Runtime | ~0.02s |

The backdoored case's planted trigger was found and correctly named among
the candidates tested; the clean case correctly stayed quiet across the
same candidate list (including decoy phrases). This validates the
evidence-combination logic in `discover_trigger_candidates`, not
real-world recall against a genuinely trained, adversarially-optimized
backdoor -- see `docs/limitations.md`.

## What would make this evaluation stronger

- A labeled dataset of real clean vs. genuinely poisoned production-scale
  model checkpoints (none exists publicly at the scale this would
  require).
- Testing against a real trained backdoor rather than a scripted
  simulator, to measure recall against adversarially-optimized triggers
  rather than an idealized deterministic rule.
- Sweeping the Isolation Forest/LOF `contamination` parameter to
  characterize the precision/recall tradeoff explicitly, rather than
  reporting a single operating point.
