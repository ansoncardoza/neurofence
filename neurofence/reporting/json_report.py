"""JSON rendering of a ScanReport -- the machine-readable report format.
Generated first, per the project's "JSON first, then PDF" ordering: the
PDF renderer reads from the same ScanReport object, so the two can never
disagree about content, only presentation.
"""

from __future__ import annotations

from pathlib import Path

from neurofence.reporting.report import ScanReport


def render_json_report(report: ScanReport) -> str:
    return report.model_dump_json(indent=2)


def write_json_report(report: ScanReport, path: str | Path) -> Path:
    target = Path(path)
    target.write_text(render_json_report(report), encoding="utf-8")
    return target
