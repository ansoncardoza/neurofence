from __future__ import annotations

from neurofence.behavioral.embeddings import TfidfSimilarityBackend
from neurofence.behavioral.pipeline import compare_behavioral_runs, run_behavioral_suite
from neurofence.behavioral.prompts import PromptCase, PromptCategory
from neurofence.behavioral.runner import HuggingFaceCausalLMRunner
from tests.behavioral.model_fixtures import build_tiny_model

_SMALL_PROMPTS = [
    PromptCase(id="p1", category=PromptCategory.GENERAL_KNOWLEDGE, prompt="Hello"),
    PromptCase(id="p2", category=PromptCategory.REASONING, prompt="2 plus 2 is"),
]


def _make_runner(seed: int) -> HuggingFaceCausalLMRunner:
    model, tokenizer = build_tiny_model(seed=seed)
    return HuggingFaceCausalLMRunner(model, tokenizer, model_id=f"tiny-{seed}", max_new_tokens=6)


def test_run_behavioral_suite_produces_one_result_per_prompt() -> None:
    runner = _make_runner(seed=10)
    results = run_behavioral_suite(runner, _SMALL_PROMPTS)

    assert len(results) == 2
    assert {r.case_id for r in results} == {"p1", "p2"}
    assert all(isinstance(r.output.text, str) for r in results)


def test_compare_behavioral_runs_same_model_high_similarity() -> None:
    runner = _make_runner(seed=11)
    baseline = run_behavioral_suite(runner, _SMALL_PROMPTS)
    repeat = run_behavioral_suite(runner, _SMALL_PROMPTS)

    records = compare_behavioral_runs(baseline, repeat, TfidfSimilarityBackend())

    assert len(records) == 2
    for record in records:
        # Same model, same prompts, greedy decoding -> outputs should be identical.
        assert record.comparison.exact_match is True
        assert record.comparison.semantic_similarity == 1.0


def test_compare_behavioral_runs_different_models_may_diverge() -> None:
    runner_a = _make_runner(seed=20)
    runner_b = _make_runner(seed=21)

    results_a = run_behavioral_suite(runner_a, _SMALL_PROMPTS)
    results_b = run_behavioral_suite(runner_b, _SMALL_PROMPTS)

    records = compare_behavioral_runs(results_a, results_b, TfidfSimilarityBackend())

    assert len(records) == 2
    for record in records:
        assert 0.0 <= record.comparison.semantic_similarity <= 1.0


def test_compare_behavioral_runs_mismatched_case_ids_skipped() -> None:
    runner = _make_runner(seed=30)
    baseline = run_behavioral_suite(runner, _SMALL_PROMPTS)
    other_prompts = [PromptCase(id="different_id", category=PromptCategory.CODING, prompt="foo")]
    candidate = run_behavioral_suite(runner, other_prompts)

    records = compare_behavioral_runs(baseline, candidate, TfidfSimilarityBackend())

    assert records == []


def test_run_behavioral_suite_default_prompts_do_not_crash() -> None:
    runner = _make_runner(seed=40)
    results = run_behavioral_suite(runner)  # uses DEFAULT_PROMPTS
    assert len(results) > 0
