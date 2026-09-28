"""NeuroFence command-line interface.

Milestone 0/1 scope: `scan` currently performs secure acquisition only
(manifest + metadata). Weight/behavioral/activation analysis, fusion, and
reporting are added in later milestones and will extend this command rather
than replace it.
"""

from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.console import Console

from neurofence.acquisition import build_manifest, extract_model_metadata
from neurofence.exceptions import NeuroFenceError
from neurofence.logging_setup import configure_logging

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
    output: Path | None = typer.Option(
        None, "--output", "-o", help="Write JSON results to this file."
    ),
) -> None:
    """Run secure acquisition (manifest + metadata) against a model directory."""
    try:
        manifest = build_manifest(model_path)
        metadata = extract_model_metadata(model_path)
    except NeuroFenceError as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        raise typer.Exit(code=1) from e

    result = {
        "manifest": manifest.model_dump(),
        "metadata": metadata.model_dump(),
    }

    text = json.dumps(result, indent=2)
    if output:
        output.write_text(text, encoding="utf-8")
        console.print(f"Results written to {output}")
    else:
        console.print(text)


if __name__ == "__main__":
    app()
