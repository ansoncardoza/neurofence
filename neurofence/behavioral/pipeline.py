"""Runs a behavioral test suite against a model and compares two runs
(e.g. baseline model vs. candidate, or baseline prompt vs. a mutated one).
"""

from __future__ import annotations

from pydantic import BaseModel

from neurofence.behavioral.comparison import BehavioralComparison, compare_outputs
from neurofence.behavioral.embeddings import SimilarityBackend, TfidfSimilarityBackend
from neurofence.behavioral.prompts import DEFAULT_PROMPTS, PromptCase
from neurofence.behavioral.runner import ModelOutput, ModelRunner


class BehavioralTestResult(BaseModel):
    case_id: str
    category: str
    output: ModelOutput


def run_behavioral_suite(
    runner: ModelRunner,
    prompts: list[PromptCase] | None = None,
) -> list[BehavioralTestResult]:
    cases = prompts if prompts is not None else DEFAULT_PROMPTS
    results = []
    for case in cases:
        output = runner.generate(case.prompt)
        results.append(
            BehavioralTestResult(case_id=case.id, category=case.category.value, output=output)
        )
    return results


class BehavioralComparisonRecord(BaseModel):
    case_id: str
    category: str
    comparison: BehavioralComparison


def compare_behavioral_runs(
    baseline: list[BehavioralTestResult],
    candidate: list[BehavioralTestResult],
    backend: SimilarityBackend | None = None,
) -> list[BehavioralComparisonRecord]:
    """Pair up two runs by case_id and compare each pair.

    Case IDs present in only one run are skipped (not silently ignored --
    callers can diff the case_id sets themselves if that matters; most
    callers run the identical prompt set through both, in which case this
    never happens).
    """
    backend = backend or TfidfSimilarityBackend()
    baseline_by_id = {r.case_id: r for r in baseline}
    records = []

    for cand in candidate:
        base = baseline_by_id.get(cand.case_id)
        if base is None:
            continue
        comparison = compare_outputs(base.output.text, cand.output.text, backend)
        records.append(
            BehavioralComparisonRecord(
                case_id=cand.case_id, category=cand.category, comparison=comparison
            )
        )

    return records
