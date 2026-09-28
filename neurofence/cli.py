"""NeuroFence command-line interface.

Milestone 0-3 scope: `scan` performs secure acquisition (manifest +
metadata) and, unless disabled, weight forensics (per-tensor statistics,
spectral analysis, layer anomaly detection). Passing --reference also runs
differential weight analysis against a trusted baseline model. Passing
--behavioral loads the model with transformers and runs the behavioral
test suite, comparing against --reference's outputs if also given.
Fuzzing and activation analysis are added in later milestones and will
extend this same command rather than replace it.
"""

from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.console import Console

from neurofence.acquisition import build_manifest, extract_model_metadata
from neurofence.behavioral import (
    HuggingFaceCausalLMRunner,
    compare_behavioral_runs,
    load_causal_lm,
    run_behavioral_suite,
)
from neurofence.exceptions import NeuroFenceError
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

    if behavioral:
        try:
            model, tokenizer = load_causal_lm(model_path)
            runner = HuggingFaceCausalLMRunner(model, tokenizer, model_id=str(model_path))
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
                comparisons = compare_behavioral_runs(ref_results, suite_results)
                result["behavioral_comparison"] = [c.model_dump() for c in comparisons]
        except NeuroFenceError as e:
            console.print(f"[yellow]Behavioral analysis skipped:[/yellow] {e}")
            result["behavioral"] = {"status": "skipped", "reason": str(e)}

    text = json.dumps(result, indent=2)
    if output:
        output.write_text(text, encoding="utf-8")
        console.print(f"Results written to {output}")
    else:
        console.print(text)


if __name__ == "__main__":
    app()
