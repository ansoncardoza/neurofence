"""NeuroFence command-line interface.

Milestone 0-5 scope: `scan` performs secure acquisition (manifest +
metadata) and, unless disabled, weight forensics (per-tensor statistics,
spectral analysis, layer anomaly detection). Passing --reference also runs
differential weight analysis against a trusted baseline model. Passing
--behavioral loads the model with transformers and runs the behavioral
test suite, comparing against --reference's outputs if also given.
Passing --trigger runs candidate-trigger discovery. Passing --activations
captures and analyzes internal activations, and combined with --trigger
also runs activation-level trigger separation analysis. Evidence fusion
and reporting are added in later milestones and will extend this same
command rather than replace it.
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
from neurofence.behavioral import (
    DEFAULT_PROMPTS,
    HuggingFaceCausalLMRunner,
    compare_behavioral_runs,
    load_causal_lm,
    run_behavioral_suite,
)
from neurofence.config import load_config
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
        None, "--output", "-o", help="Write JSON results to this file."
    ),
) -> None:
    """Run secure acquisition, and optionally weight forensics, against a model directory."""
    try:
        manifest = build_manifest(model_path)
        metadata = extract_model_metadata(model_path)
    except NeuroFenceError as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        raise typer.Exit(code=1) from e

    result: dict[str, object] = {
        "manifest": manifest.model_dump(),
        "metadata": metadata.model_dump(),
    }

    # Kept as live objects (not just the serialized dicts in `result`) so
    # evidence fusion at the end can consume them directly.
    weight_result = None
    behavioral_comparisons = None
    trigger_results = None
    activation_anomaly_summaries = None

    if weights:
        try:
            weight_result = analyze_model_weights(model_path)
            result["weight_forensics"] = weight_result.model_dump()
        except NeuroFenceError as e:
            console.print(f"[yellow]Weight forensics skipped:[/yellow] {e}")
            result["weight_forensics"] = {"status": "skipped", "reason": str(e)}

    if reference is not None:
        try:
            differential = compare_models(reference, model_path)
            result["differential"] = differential.model_dump()
        except NeuroFenceError as e:
            console.print(f"[bold red]Error:[/bold red] {e}")
            raise typer.Exit(code=1) from e

    if behavioral or trigger or activations:
        try:
            model, tokenizer = load_causal_lm(model_path)
            runner = HuggingFaceCausalLMRunner(model, tokenizer, model_id=str(model_path))

            if behavioral:
                suite_results = run_behavioral_suite(runner)
                result["behavioral"] = {
                    "test_count": len(suite_results),
                    "results": [r.model_dump() for r in suite_results],
                }

                if reference is not None:
                    ref_model, ref_tokenizer = load_causal_lm(reference)
                    ref_runner = HuggingFaceCausalLMRunner(
                        ref_model, ref_tokenizer, model_id=str(reference)
                    )
                    ref_results = run_behavioral_suite(ref_runner)
                    behavioral_comparisons = compare_behavioral_runs(ref_results, suite_results)
                    result["behavioral_comparison"] = [
                        c.model_dump() for c in behavioral_comparisons
                    ]

            if trigger:
                trigger_results = discover_trigger_candidates(runner, DEFAULT_PROMPTS, trigger)
                result["trigger_candidates"] = [r.model_dump() for r in trigger_results]

            if activations:
                base_prompts = [(p.id, p.prompt) for p in DEFAULT_PROMPTS]
                captured = capture_activations_for_prompts(model, tokenizer, base_prompts)
                activation_anomaly_summaries = [
                    detect_activation_anomalies(layer_stats) for layer_stats in captured.values()
                ]
                result["activation_forensics"] = {
                    "prompts_captured": len(captured),
                    "per_prompt_anomaly_detection": {
                        cid: summary.model_dump()
                        for cid, summary in zip(captured, activation_anomaly_summaries, strict=True)
                    },
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
                    layer_names = {name for layers in captured.values() for name in layers}
                    result["activation_trigger_analysis"] = {
                        layer_name: analyze_trigger_activations(
                            captured, trigger_captured, layer_name
                        ).model_dump()
                        for layer_name in sorted(layer_names)
                    }
        except NeuroFenceError as e:
            console.print(f"[yellow]Behavioral/fuzzing analysis skipped:[/yellow] {e}")
            result["behavioral"] = {"status": "skipped", "reason": str(e)}

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
    result["evidence_fusion"] = fusion_result.model_dump()

    text = json.dumps(result, indent=2)
    if output:
        output.write_text(text, encoding="utf-8")
        console.print(f"Results written to {output}")
    else:
        console.print(text)


if __name__ == "__main__":
    app()
