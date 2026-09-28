"""End-to-end CLI tests using typer's CliRunner -- these invoke the real
`app`, not the underlying functions directly, so they catch wiring bugs
(wrong option names, exceptions escaping past the try/except, argument
mismatches) that unit tests on individual modules cannot.

Behavioral/activation/trigger flags are deliberately not exercised here
(they require loading a real transformers model, which is covered by the
dedicated integration tests in tests/behavioral, tests/activation, and
tests/fuzzing, plus manual end-to-end verification during development).
This file focuses on the acquisition/weight-forensics/reporting/benchmark
paths, which need no ML model loading and so run fast.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from safetensors.numpy import save_file
from typer.testing import CliRunner

from neurofence.cli import app

runner = CliRunner()


@pytest.fixture
def small_model_dir(tmp_path: Path) -> Path:
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    rng = np.random.default_rng(0)
    tensors = {
        f"layer{i}.weight": rng.standard_normal((16, 16)).astype(np.float32) for i in range(10)
    }
    save_file(tensors, str(model_dir / "model.safetensors"))
    (model_dir / "config.json").write_text(
        json.dumps({"architectures": ["Demo"], "num_hidden_layers": 10}), encoding="utf-8"
    )
    return model_dir


def test_help_exits_zero() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "scan" in result.stdout
    assert "benchmark" in result.stdout
    assert "report" in result.stdout


def test_scan_missing_model_path_fails_cleanly(tmp_path: Path) -> None:
    result = runner.invoke(app, ["scan", str(tmp_path / "does_not_exist")])
    assert result.exit_code == 1
    assert "Error" in result.stdout


def test_scan_writes_json_report(small_model_dir: Path, tmp_path: Path) -> None:
    output = tmp_path / "report.json"
    result = runner.invoke(app, ["scan", str(small_model_dir), "--output", str(output)])

    assert result.exit_code == 0, result.stdout
    assert output.exists()

    parsed = json.loads(output.read_text(encoding="utf-8"))
    assert parsed["model_path"] == str(small_model_dir)
    assert parsed["weight_forensics"] is not None
    assert parsed["evidence_fusion"]["status"] == "evaluated"
    assert "findings" in parsed
    assert "limitations" in parsed


def test_scan_no_weights_skips_weight_forensics(small_model_dir: Path, tmp_path: Path) -> None:
    output = tmp_path / "report.json"
    result = runner.invoke(
        app, ["scan", str(small_model_dir), "--no-weights", "--output", str(output)]
    )

    assert result.exit_code == 0, result.stdout
    parsed = json.loads(output.read_text(encoding="utf-8"))
    assert parsed["weight_forensics"] is None
    assert parsed["evidence_fusion"]["status"] == "not_evaluated"


def test_scan_prints_json_when_no_output_given(small_model_dir: Path) -> None:
    result = runner.invoke(app, ["scan", str(small_model_dir)])
    assert result.exit_code == 0, result.stdout
    parsed = json.loads(result.stdout)
    assert parsed["model_path"] == str(small_model_dir)


def test_scan_with_pdf_writes_both_formats(small_model_dir: Path, tmp_path: Path) -> None:
    json_out = tmp_path / "report.json"
    pdf_out = tmp_path / "report.pdf"
    result = runner.invoke(
        app,
        ["scan", str(small_model_dir), "--output", str(json_out), "--pdf", str(pdf_out)],
    )

    assert result.exit_code == 0, result.stdout
    assert json_out.exists()
    assert pdf_out.exists()
    assert pdf_out.read_bytes()[:5] == b"%PDF-"


def test_scan_reference_incompatible_still_produces_report(
    small_model_dir: Path, tmp_path: Path
) -> None:
    other_model_dir = tmp_path / "other_model"
    other_model_dir.mkdir()
    save_file(
        {"totally_different.weight": np.zeros((4, 4), dtype=np.float32)},
        str(other_model_dir / "model.safetensors"),
    )
    (other_model_dir / "config.json").write_text(json.dumps({"architectures": ["Other"]}))

    output = tmp_path / "report.json"
    result = runner.invoke(
        app,
        [
            "scan",
            str(small_model_dir),
            "--reference",
            str(other_model_dir),
            "--output",
            str(output),
        ],
    )

    assert result.exit_code == 0, result.stdout
    parsed = json.loads(output.read_text(encoding="utf-8"))
    assert parsed["differential"]["status"] == "reference_incompatible"


def test_report_command_renders_pdf_from_json(small_model_dir: Path, tmp_path: Path) -> None:
    json_out = tmp_path / "report.json"
    scan_result = runner.invoke(app, ["scan", str(small_model_dir), "--output", str(json_out)])
    assert scan_result.exit_code == 0, scan_result.stdout

    pdf_out = tmp_path / "report.pdf"
    report_result = runner.invoke(app, ["report", str(json_out), "--pdf", str(pdf_out)])

    assert report_result.exit_code == 0, report_result.stdout
    assert pdf_out.exists()
    assert pdf_out.read_bytes()[:5] == b"%PDF-"


def test_report_command_missing_json_file_fails_cleanly(tmp_path: Path) -> None:
    result = runner.invoke(
        app, ["report", str(tmp_path / "nope.json"), "--pdf", str(tmp_path / "out.pdf")]
    )
    assert result.exit_code == 1
    assert "Error" in result.stdout


def test_report_command_malformed_json_fails_cleanly(tmp_path: Path) -> None:
    bad_json = tmp_path / "bad.json"
    bad_json.write_text("{not valid json", encoding="utf-8")
    result = runner.invoke(app, ["report", str(bad_json), "--pdf", str(tmp_path / "out.pdf")])
    assert result.exit_code == 1
    assert "Error" in result.stdout


def test_benchmark_runs_and_reports_numbers() -> None:
    result = runner.invoke(app, ["benchmark", "--n-clean", "5", "--n-poisoned-per-kind", "1"])
    assert result.exit_code == 0, result.stdout
    assert "Weight anomaly detector" in result.stdout
    assert "Trigger detector" in result.stdout
    assert "Accuracy:" in result.stdout


def test_benchmark_writes_json_output(tmp_path: Path) -> None:
    output = tmp_path / "bench.json"
    result = runner.invoke(
        app,
        ["benchmark", "--n-clean", "5", "--n-poisoned-per-kind", "1", "--output", str(output)],
    )
    assert result.exit_code == 0, result.stdout
    assert output.exists()
    parsed = json.loads(output.read_text(encoding="utf-8"))
    assert "weight_anomaly" in parsed
    assert "trigger" in parsed
