from __future__ import annotations

from neurofence.fuzzing.generator import ALL_CATEGORIES, generate_fuzz_cases

BASE = [("p1", "What is the capital of France?"), ("p2", "Explain recursion.")]


def test_empty_base_prompts_only_random_text_generated() -> None:
    cases = generate_fuzz_cases([], categories=list(ALL_CATEGORIES), max_prompts=100)
    assert len(cases) > 0
    assert all(c.category == "random_text" for c in cases)


def test_no_categories_selected_no_random_text_produces_no_cases() -> None:
    cases = generate_fuzz_cases([], categories=["repetition"], max_prompts=100)
    assert cases == []


def test_only_selected_categories_appear() -> None:
    cases = generate_fuzz_cases(BASE, categories=["repetition"], max_prompts=100)
    assert cases
    assert all(c.category == "repetition" for c in cases)


def test_max_prompts_budget_respected() -> None:
    cases = generate_fuzz_cases(BASE, categories=list(ALL_CATEGORIES), max_prompts=5)
    assert len(cases) == 5


def test_smaller_budget_is_prefix_of_larger_budget() -> None:
    small = generate_fuzz_cases(BASE, categories=list(ALL_CATEGORIES), max_prompts=5, seed=1)
    large = generate_fuzz_cases(BASE, categories=list(ALL_CATEGORIES), max_prompts=50, seed=1)
    assert [c.prompt for c in small] == [c.prompt for c in large[:5]]


def test_deterministic_given_seed() -> None:
    a = generate_fuzz_cases(BASE, categories=list(ALL_CATEGORIES), seed=7)
    b = generate_fuzz_cases(BASE, categories=list(ALL_CATEGORIES), seed=7)
    assert [c.prompt for c in a] == [c.prompt for c in b]


def test_different_seed_different_output() -> None:
    a = generate_fuzz_cases(BASE, categories=["unicode"], seed=1)
    b = generate_fuzz_cases(BASE, categories=["unicode"], seed=2)
    assert [c.prompt for c in a] != [c.prompt for c in b]


def test_case_ids_unique() -> None:
    cases = generate_fuzz_cases(BASE, categories=list(ALL_CATEGORIES), max_prompts=200)
    ids = [c.id for c in cases]
    assert len(ids) == len(set(ids))


def test_candidate_trigger_category_appends_each_trigger() -> None:
    cases = generate_fuzz_cases(
        BASE,
        categories=["candidate_trigger"],
        candidate_triggers=["MAGIC_TRIGGER_XYZ"],
        max_prompts=100,
    )
    assert len(cases) == len(BASE)
    assert all("MAGIC_TRIGGER_XYZ" in c.prompt for c in cases)
    assert all(c.category == "candidate_trigger" for c in cases)


def test_candidate_trigger_category_empty_triggers_produces_no_cases() -> None:
    cases = generate_fuzz_cases(BASE, categories=["candidate_trigger"], candidate_triggers=[])
    assert cases == []


def test_base_prompt_and_case_id_recorded() -> None:
    cases = generate_fuzz_cases(BASE, categories=["repetition"], max_prompts=100)
    assert cases[0].base_case_id == "p1"
    assert cases[0].base_prompt == BASE[0][1]


def test_random_text_has_no_base_prompt() -> None:
    cases = generate_fuzz_cases([], categories=["random_text"], max_prompts=5)
    assert all(c.base_case_id is None and c.base_prompt is None for c in cases)


def test_unknown_category_silently_ignored() -> None:
    cases = generate_fuzz_cases(BASE, categories=["not_a_real_category"], max_prompts=100)
    assert cases == []


def test_max_prompts_zero_or_negative_produces_no_cases() -> None:
    cases = generate_fuzz_cases(BASE, categories=list(ALL_CATEGORIES), max_prompts=0)
    assert cases == []
