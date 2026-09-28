"""NeuroFence command-line interface.

Milestone 0-8 scope: `scan` performs secure acquisition (manifest +
metadata) and, unless disabled, weight forensics (per-tensor statistics,
spectral analysis, layer anomaly detection). Passing --reference also runs
differential weight analysis against a trusted baseline model. Passing
--behavioral loads the model with transformers and runs the behavioral
test suite, comparing against --reference's outputs if also given.
Passing --trigger runs candidate-trigger discovery. Passing --activations
captures and analyzes internal activations, and combined with --trigger
also runs activation-level trigger separation analysis. `scan` always
ends with evidence fusion (anomaly score, threat confidence, risk label)
and assembles a full ScanReport: --output writes the JSON report,
--pdf additionally renders a PDF report from the exact same data.

`report` renders a PDF from a JSON report file previously written by
`scan --output`, without re-running any detector.

`benchmark` measures detector performance against synthetic ground-truth
data (neurofence.attack_lab) -- every number is measured at run time, not
invented.
"""

from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.console import Console

from neurofence.acquisition import build_manifest, extract_model_metadata
from neurofence.activation import (
    capture_activations_for_prompts,
    detect_activation_anomalies,
)
from neurofence.activation.trigger_analysis import analyze_trigger_activations
from neurofence.attack_lab import build_labeled_dataset
from neurofence.behavioral import (
    DEFAULT_PROMPTS,
    HuggingFaceCausalLMRunner,
    compare_behavioral_runs,
    load_causal_lm,
    run_behavioral_suite,
)
from neurofence.config import load_config
from neurofence.evaluation import evaluate_trigger_detector, evaluate_weight_anomaly_detector
from neurofence.exceptions import NeuroFenceError
from neurofence.fusion import (
    activation_anomaly_score,
    behavioral_anomaly_score,
    fuse_evidence,
    integrity_score,
    trigger_evidence_score,
    weight_anomaly_score,
)
from neurofence.fuzzing import discover_trigger_candidates
from neurofence.fuzzing.mutators import mutate_prompt_append
from neurofence.logging_setup import configure_logging
from neurofence.reporting import (
    ScanReport,
    build_report,
    render_json_report,
    render_pdf_report,
    write_json_report,
)
from neurofence.weight_forensics import analyze_model_weights, compare_models

app = typer.Typer(add_completion=False, help="NeuroFence: AI model forensic and security platform.")
console = Console()


@app.callback()
def main(
    log_level: str = typer.Option("INFO", help="Logging level (DEBUG, INFO, WARNING, ERROR)."),
) -> None:
    configure_logging(log_level)


@app.command()
def scan(
    model_path: Path = typer.Argument(..., help="Path to a local model directory."),
    reference: Path | None = typer.Option(
        None, "--reference", help="Trusted baseline model directory for differential analysis."
    ),
    weights: bool = typer.Option(
        True, "--weights/--no-weights", help="Run weight forensics (statistics, spectral, anomaly)."
    ),
    behavioral: bool = typer.Option(
        False, "--behavioral", help="Load the model and run the behavioral test suite."
    ),
    trigger: list[str] = typer.Option(
        [], "--trigger", help="Candidate trigger phrase to test (repeatable). Implies --behavioral."
    ),
    activations: bool = typer.Option(
        False, "--activations", help="Capture and analyze internal activations."
    ),
    output: Path | None = typer.Option(
        None, "--output", "-o", help="Write the JSON report to this file."
    ),
    pdf: Path | None = typer.Option(None, "--pdf", help="Also render a PDF report to this file."),
) -> None:
    """Run secure acquisition, and the enabled detectors, against a model directory."""
    try:
        manifest = build_manifest(model_path)
        metadata = extract_model_metadata(model_path)
    except NeuroFenceError as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        raise typer.Exit(code=1) from e

    weight_result = None
    differential = None
    behavioral_results = None
    behavioral_comparisons = None
    trigger_results = None
    activation_anomaly_summaries = None
    activation_summary: dict[str, object] | None = None

    if weights:
        try:
            weight_result = analyze_model_weights(model_path)
        except NeuroFenceError as e:
            console.print(f"[yellow]Weight forensics skipped:[/yellow] {e}")

    if reference is not None:
        try:
            differential = compare_models(reference, model_path)
        except NeuroFenceError as e:
            console.print(f"[bold red]Error:[/bold red] {e}")
            raise typer.Exit(code=1) from e

    if behavioral or trigger or activations:
        try:
            model, tokenizer = load_causal_lm(model_path)
            runner = HuggingFaceCausalLMRunner(model, tokenizer, model_id=str(model_path))

            if behavioral:
                behavioral_results = run_behavioral_suite(runner)

                if reference is not None:
                    ref_model, ref_tokenizer = load_causal_lm(reference)
                    ref_runner = HuggingFaceCausalLMRunner(
                        ref_model, ref_tokenizer, model_id=str(reference)
                    )
                    ref_results = run_behavioral_suite(ref_runner)
                    behavioral_comparisons = compare_behavioral_runs(
                        ref_results, behavioral_results
                    )

            if trigger:
                trigger_results = discover_trigger_candidates(runner, DEFAULT_PROMPTS, trigger)

            if activations:
                base_prompts = [(p.id, p.prompt) for p in DEFAULT_PROMPTS]
                captured = capture_activations_for_prompts(model, tokenizer, base_prompts)
                activation_anomaly_summaries = [
                    detect_activation_anomalies(layer_stats) for layer_stats in captured.values()
                ]
                flagged_total = sum(
                    sum(1 for r in s.results if r.is_outlier) for s in activation_anomaly_summaries
                )
                activation_summary = {
                    "prompts_captured": len(captured),
                    "layers_flagged_total": flagged_total,
                }

                if trigger:
                    trigger_prompts = [
                        (f"{p.id}__{t}", mutate_prompt_append(p.prompt, t))
                        for p in DEFAULT_PROMPTS
                        for t in trigger
                    ]
                    trigger_captured = capture_activations_for_prompts(
                        model, tokenizer, trigger_prompts
                    )
                    layer_names = sorted({name for layers in captured.values() for name in layers})
                    separated = sum(
                        1
                        for layer_name in layer_names
                        if analyze_trigger_activations(
                            captured, trigger_captured, layer_name
                        ).consistent_separation
                    )
                    activation_summary["layers_with_consistent_trigger_separation"] = separated
        except NeuroFenceError as e:
            console.print(f"[yellow]Behavioral/fuzzing analysis skipped:[/yellow] {e}")

    # Evidence fusion: combine whatever detectors actually ran into an
    # Anomaly Score and a separate Threat Confidence. Note: integrity_score
    # is always "not_evaluated" here -- that detector re-verifies a model
    # directory against a previously stored manifest, and `scan` does not
    # yet accept a stored baseline manifest to compare against (only a
    # full --reference *model* for differential weight analysis, a
    # different check). This is a known gap, not a fabricated "clean".
    config = load_config()
    sub_scores = {
        "integrity": integrity_score(None),
        "weight_anomaly": weight_anomaly_score(
            weight_result.anomaly_detection if weight_result else None
        ),
        "behavioral": behavioral_anomaly_score(behavioral_comparisons),
        "activation": activation_anomaly_score(activation_anomaly_summaries),
        "trigger": trigger_evidence_score(trigger_results),
    }
    fusion_result = fuse_evidence(sub_scores, config.risk.weights, config.risk)

    report = build_report(
        model_path=str(model_path),
        manifest=manifest,
        metadata=metadata,
        evidence_fusion=fusion_result,
        weight_forensics=weight_result,
        differential=differential,
        behavioral=behavioral_results,
        behavioral_comparison=behavioral_comparisons,
        trigger_candidates=trigger_results,
        activation_summary=activation_summary,
    )

    if output:
        write_json_report(report, output)
        console.print(f"JSON report written to {output}")
    else:
        # Plain print, not console.print: Rich soft-wraps long lines,
        # which corrupts the JSON (breaks strings mid-value) when stdout
        # is captured or piped into a JSON consumer.
        print(render_json_report(report))

    if pdf:
        render_pdf_report(report, pdf)
        console.print(f"PDF report written to {pdf}")


@app.command(name="report")
def report_command(
    json_path: Path = typer.Argument(..., help="JSON report file written by `scan --output`."),
    pdf: Path = typer.Option(..., "--pdf", help="Path to render the PDF report to."),
) -> None:
    """Render a PDF report from a previously generated JSON report, without re-running detectors."""
    try:
        scan_report = ScanReport.model_validate_json(json_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        console.print(f"[bold red]Error:[/bold red] could not load {json_path}: {e}")
        raise typer.Exit(code=1) from e

    render_pdf_report(scan_report, pdf)
    console.print(f"PDF report written to {pdf}")


@app.command()
def benchmark(
    n_clean: int = typer.Option(15, help="Clean synthetic models to generate."),
    n_poisoned_per_kind: int = typer.Option(
        3, help="Poisoned synthetic models per perturbation kind."
    ),
    seed: int = typer.Option(1337, help="Random seed for reproducible synthetic data."),
    output: Path | None = typer.Option(
        None, "--output", "-o", help="Write JSON results to this file."
    ),
) -> None:
    """Measure detector performance against synthetic ground-truth data.

    Every number is computed by actually running the detector against
    freshly generated synthetic data -- nothing here is a cached or
    invented "typical" result.
    """
    dataset = build_labeled_dataset(
        n_clean=n_clean, n_poisoned_per_kind=n_poisoned_per_kind, seed=seed
    )
    weight_result = evaluate_weight_anomaly_detector(dataset)
    trigger_result = evaluate_trigger_detector()

    console.print(f"[bold]Weight anomaly detector[/bold] ({weight_result.detector})")
    console.print(f"  Dataset: {weight_result.n_samples} synthetic models (seed={seed})")
    console.print(f"  Accuracy: {weight_result.metrics.accuracy:.3f}")
    console.print(f"  Precision: {weight_result.metrics.precision}")
    console.print(f"  Recall: {weight_result.metrics.recall}")
    console.print(f"  F1: {weight_result.metrics.f1}")
    console.print(f"  ROC-AUC: {weight_result.roc_auc} ({weight_result.roc_auc_note or 'ok'})")
    console.print(
        f"  Layer localization rate (correct layer flagged, poisoned samples only): "
        f"{weight_result.layer_localization_rate}"
    )
    console.print(f"  Runtime: {weight_result.runtime_seconds:.3f}s")
    console.print()
    console.print(f"[bold]Trigger detector[/bold] ({trigger_result.detector})")
    console.print(f"  Accuracy: {trigger_result.metrics.accuracy:.3f}")
    console.print(f"  Runtime: {trigger_result.runtime_seconds:.3f}s")

    if output:
        payload = {
            "weight_anomaly": weight_result.model_dump(),
            "trigger": trigger_result.model_dump(),
        }
        output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        console.print(f"\nFull results written to {output}")


if __name__ == "__main__":
    app()
