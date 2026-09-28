# Threat Model

## What NeuroFence is designed to detect

- **Supply-chain modification**: a model's weight files changed after
  publication, detected via manifest hash re-verification
  (`neurofence.acquisition.manifest.verify_manifest`) when a prior
  baseline manifest is available.
- **Weight poisoning / tampering**: localized, distributed, layer-level,
  and neuron-level modifications to weight tensors, detected via
  statistical anomaly detection (`neurofence.weight_forensics`) and,
  when a trusted reference model is supplied, direct differential
  comparison (`compare_models`).
- **Trigger-conditioned behavior ("backdoors")**: a model that behaves
  normally except when a specific input phrase is present, detected via
  candidate-trigger fuzzing and behavioral comparison
  (`neurofence.fuzzing.trigger_discovery`) and, when internal activations
  are captured, activation-level separation analysis
  (`neurofence.activation.trigger_analysis`).
- **Activation-level anomalies**: internal representations that behave
  unusually for a given layer relative to the model's other layers
  (`neurofence.activation.anomaly`), reusing the same consensus detector
  as weight forensics.
- **Behavioral drift**: output changes (refusal behavior, semantic
  content, length) between a candidate model and a trusted reference
  model across a fixed prompt suite (`neurofence.behavioral`).

## What NeuroFence explicitly does NOT guarantee

- **Absence of a detected anomaly does not prove a model is safe.** It
  means the detectors that ran found no evidence -- a materially weaker
  claim. A model that was never scanned with `--behavioral`, `--trigger`,
  or `--activations` has simply not been checked along those dimensions.
- **Unknown attack techniques may evade every detector implemented
  here.** All detection is signature/statistics-based; a sufficiently
  subtle or novel poisoning technique designed to evade these specific
  methods is not guaranteed to be caught. The synthetic attack lab
  (`neurofence.attack_lab`) demonstrates this concretely: the
  `low_magnitude` perturbation kind (noise comparable to normal weight
  scale) is caught 0% of the time by the current weight-anomaly detector
  in our own measured benchmark -- see `docs/evaluation.md`.
- **Behavioral and trigger-discovery analysis depend entirely on the
  prompt/trigger coverage used for a given scan.** A trigger phrase that
  was never tested cannot be found. `DEFAULT_PROMPTS`
  (`neurofence.behavioral.prompts`) is a small, fixed set across 7
  categories -- real deployments should supply broader, domain-specific
  prompt sets.
- **Statistical weight anomalies can have entirely benign explanations**:
  unusual but legitimate initialization, architecture-specific layers
  (e.g. a bias vector genuinely differing in scale from weight matrices),
  or ordinary fine-tuning. NeuroFence reports *evidence*, never a
  unilateral verdict -- see `neurofence.fusion` for how Anomaly Score and
  Threat Confidence are kept conceptually distinct for exactly this
  reason.
- **Reference-free detection is inherently weaker than differential
  comparison.** Without a `--reference` model, weight forensics can only
  ask "does this layer look unusual relative to this model's *other*
  layers," not "does this layer differ from a known-good baseline."
- **Anomaly Score and Threat Confidence are project-defined heuristic
  scores**, not calibrated probabilities, and not derived from a labeled
  real-world dataset of known-clean and known-poisoned production
  models. They are derived from documented, inspectable formulas (see
  `neurofence.fusion.combine`) and from measured (not invented)
  performance against *synthetic* ground truth (`neurofence.evaluation`).
  Neither number should be treated as "the model is N% likely to be
  malicious."
- **NeuroFence itself does not execute untrusted model code.**
  `trust_remote_code` is hard-disabled throughout (see
  `neurofence.config.schema.ModelConfig`), and legacy pickle-format
  (`.bin`/`.pt`/`.pth`) weight files are hashed but never unpickled --
  parameter counting and weight-level forensics are simply unavailable
  for pickle-only checkpoints (see `neurofence.acquisition.metadata`).
  This is a safety boundary, not a detection gap: NeuroFence would rather
  refuse to analyze a pickle checkpoint's internals than risk running
  arbitrary code from an untrusted source.

## Multi-signal principle

No single detector's output is treated as a final determination. Evidence
fusion (`neurofence.fusion.combine.fuse_evidence`) explicitly rewards
*agreement* across independent detectors: an isolated elevated signal
(nothing else corroborates it) yields materially lower Threat Confidence
than the same severity corroborated by multiple independent detectors,
even though both cases can produce a similar Anomaly Score. This is the
concrete, numeric expression of "never make a security determination from
one detector."
