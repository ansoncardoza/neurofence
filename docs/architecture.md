# Architecture

## Pipeline

```
MODEL DIRECTORY
      |
      v
SECURE ACQUISITION            neurofence.acquisition
  - path-traversal-safe file walk
  - SHA-256/SHA-512 manifest (deterministic)
  - safe metadata extraction (config.json + safetensors headers only;
    pickle weight files hashed, never unpickled)
      |
      v
WEIGHT FORENSICS               neurofence.weight_forensics
  - per-tensor statistics (numerically-safe: overflow -> None, not inf/NaN)
  - spectral analysis (SVD, truncated for large matrices)
  - layer anomaly detection (Isolation Forest + LOF + Mahalanobis consensus,
    or MAD-based statistical fallback for small layer counts)
  - differential analysis (ΔW vs. a --reference model)
      |
      v
BEHAVIORAL BASELINE             neurofence.behavioral
  - fixed prompt catalog (7 categories)
  - ModelRunner protocol -> HuggingFaceCausalLMRunner (greedy decoding,
    trust_remote_code hard-disabled, context-window truncation)
  - comparison: semantic similarity, refusal detection, length ratio
  - KL / Jensen-Shannon divergence for logit-level comparison
      |
      v
ADVERSARIAL FUZZING              neurofence.fuzzing
  - deterministic mutators: random_text, repetition, unicode
    (homoglyphs/zero-width/mixed-script), formatting, prompt_mutation
  - candidate-trigger discovery: baseline vs. mutated-prompt comparison,
    ranked by evidence, "consistent" only on a strict majority
      |
      v
ACTIVATION FORENSICS              neurofence.activation
  - forward-hook capture of compact per-layer summaries (nn.Linear +
    transformers Conv1D)
  - reuses weight_forensics' anomaly consensus detector on activation
    feature vectors
  - PCA + DBSCAN for cross-prompt structure; activation-level trigger
    separation analysis
      |
      v
EVIDENCE FUSION                  neurofence.fusion
  - per-detector SubScore (0-100, or not_evaluated)
  - Anomaly Score (weighted average of what ran)
  - Threat Confidence (rewards agreement across independent detectors)
  - Risk Classification (LOW/MEDIUM/HIGH/CRITICAL via neurofence.config.RiskConfig)
      |
      v
REPORTING                        neurofence.reporting
  - ScanReport: the canonical data model both renderers read from
  - JSON report (machine-readable, generated first)
  - PDF report (ReportLab; all model-/user-controlled text escaped)
```

Supporting, cross-cutting modules:

- `neurofence.attack_lab` -- synthetic clean/poisoned model + scripted
  backdoor generators with ground-truth labels, used only by
  `neurofence.evaluation` and its tests.
- `neurofence.evaluation` -- classification metrics (precision/recall/F1/
  ROC-AUC/PR-AUC, all reporting `None` with a note when undefined) and a
  benchmark harness that measures real detector performance against the
  attack lab's synthetic ground truth.
- `neurofence.config` -- the single source of truth for every threshold
  and risk weight (`neurofence/config/default.yaml`), loaded via
  `load_config()`.

## Package layout

```
neurofence/
  acquisition/       Milestone 1 -- manifest, metadata, path safety, formats
  weight_forensics/  Milestone 2 -- statistics, robust stats, spectral,
                      anomaly, differential, pipeline
  behavioral/         Milestone 3 -- prompts, runner, embeddings, refusal,
                      distribution, comparison, pipeline
  fuzzing/            Milestone 4 -- mutators, generator, trigger_discovery
  activation/         Milestone 5 -- capture, pipeline, anomaly, reduction,
                      clustering, trigger_analysis
  fusion/             Milestone 6 -- scores, combine
  attack_lab/         Milestone 7 -- synthetic_models, dataset, trigger_experiments
  evaluation/         Milestone 7 -- metrics, benchmark
  reporting/           Milestone 8 -- report, json_report, pdf_report
  config/             schema (Pydantic), loader, default.yaml
  cli.py              Typer CLI: scan, report, benchmark
  exceptions.py        shared exception hierarchy
  logging_setup.py    central logging configuration
```

`tests/` mirrors this layout 1:1, plus `tests/test_cli.py` (end-to-end CLI
tests via `typer.testing.CliRunner`) and `tests/test_config.py`.

## Design decisions worth knowing

- **Weight forensics needs no PyTorch.** Tensors are read directly from
  safetensors via NumPy; PyTorch/transformers are only pulled in (the
  `ml` extra) starting at Milestone 3, which is the first stage that
  actually runs a model.
- **Layer anomaly detection is shared code**, not duplicated: both
  `weight_forensics.anomaly` and `activation.anomaly` call the same
  `detect_layer_anomalies` function -- a feature vector is a feature
  vector regardless of whether it summarizes a weight tensor or an
  activation tensor.
- **Every numeric feature that scales with tensor size was found to be a
  bug, not a feature** (see `weight_forensics/pipeline.py`'s
  `build_layer_feature_vector`): raw L1/L2 norms were replaced with
  per-element-normalized versions after the attack-lab benchmark exposed
  100% false positives on differently-shaped layers.
