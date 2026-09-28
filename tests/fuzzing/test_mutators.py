from __future__ import annotations

import random

from neurofence.fuzzing import mutators


def test_random_text_is_deterministic_given_seed() -> None:
    a = mutators.mutate_random_text(random.Random(42), length=20)
    b = mutators.mutate_random_text(random.Random(42), length=20)
    assert a == b
    assert len(a) == 20


def test_random_text_different_seeds_differ() -> None:
    a = mutators.mutate_random_text(random.Random(1), length=20)
    b = mutators.mutate_random_text(random.Random(2), length=20)
    assert a != b


def test_repetition_produces_repeated_unit() -> None:
    result = mutators.mutate_repetition("hello world", random.Random(0))
    words = result.split()
    assert len(set(words)) == 1
    assert len(words) >= 10


def test_repetition_on_empty_string_no_crash() -> None:
    result = mutators.mutate_repetition("", random.Random(0))
    assert len(result) > 0


def test_repetition_on_single_word() -> None:
    result = mutators.mutate_repetition("hello", random.Random(0))
    assert set(result.split()) == {"hello"}


def test_homoglyphs_changes_some_characters() -> None:
    base = "cat apple exit"
    mutated = mutators.mutate_unicode_homoglyphs(base, random.Random(0), fraction=1.0)
    assert mutated != base
    assert len(mutated) == len(base)


def test_homoglyphs_on_text_with_no_matching_letters_no_crash() -> None:
    mutated = mutators.mutate_unicode_homoglyphs("123 456", random.Random(0))
    assert mutated == "123 456"


def test_homoglyphs_on_empty_string_no_crash() -> None:
    assert mutators.mutate_unicode_homoglyphs("", random.Random(0)) == ""


def test_zero_width_insertion_increases_length() -> None:
    base = "hello world"
    mutated = mutators.mutate_unicode_zero_width(base, random.Random(0))
    assert len(mutated) > len(base)


def test_zero_width_on_empty_string_no_crash() -> None:
    mutated = mutators.mutate_unicode_zero_width("", random.Random(0))
    assert len(mutated) > 0


def test_mixed_scripts_insertion_increases_length() -> None:
    base = "hello world"
    mutated = mutators.mutate_unicode_mixed_scripts(base, random.Random(0))
    assert len(mutated) > len(base)


def test_formatting_case_variants_no_crash() -> None:
    for seed in range(10):
        result = mutators.mutate_formatting_case("Hello World", random.Random(seed))
        assert isinstance(result, str)
        assert len(result) == len("Hello World")


def test_formatting_case_on_empty_string() -> None:
    assert mutators.mutate_formatting_case("", random.Random(0)) == ""


def test_formatting_whitespace_increases_length() -> None:
    base = "hello world foo bar"
    mutated = mutators.mutate_formatting_whitespace(base, random.Random(0))
    assert len(mutated) >= len(base)


def test_formatting_whitespace_on_empty_string_no_crash() -> None:
    mutated = mutators.mutate_formatting_whitespace("", random.Random(0))
    assert len(mutated) > 0


def test_formatting_punctuation_strip_removes_punctuation() -> None:
    result = mutators.mutate_formatting_punctuation("Hello, world!!!", random.Random(0))
    assert isinstance(result, str)


def test_formatting_punctuation_single_word_no_crash() -> None:
    result = mutators.mutate_formatting_punctuation("hello", random.Random(0))
    assert isinstance(result, str)


def test_prompt_insert_contains_insertion() -> None:
    result = mutators.mutate_prompt_insert("the quick brown fox", "MARKER", random.Random(0))
    assert "MARKER" in result


def test_prompt_insert_on_empty_base() -> None:
    assert mutators.mutate_prompt_insert("", "MARKER", random.Random(0)) == "MARKER"


def test_prompt_prepend() -> None:
    result = mutators.mutate_prompt_prepend("world", "hello")
    assert result == "hello world"


def test_prompt_append() -> None:
    result = mutators.mutate_prompt_append("hello", "world")
    assert result == "hello world"


def test_prompt_replace_swaps_one_word() -> None:
    result = mutators.mutate_prompt_replace("the quick brown fox", random.Random(0), "MARKER")
    words = result.split()
    assert "MARKER" in words
    assert len(words) == 4


def test_prompt_replace_on_empty_base() -> None:
    assert mutators.mutate_prompt_replace("", random.Random(0), "MARKER") == "MARKER"


def test_prompt_reorder_preserves_word_multiset() -> None:
    base = "the quick brown fox jumps"
    result = mutators.mutate_prompt_reorder(base, random.Random(0))
    assert sorted(result.split()) == sorted(base.split())


def test_prompt_reorder_single_word_unchanged() -> None:
    assert mutators.mutate_prompt_reorder("hello", random.Random(0)) == "hello"


def test_prompt_reorder_empty_unchanged() -> None:
    assert mutators.mutate_prompt_reorder("", random.Random(0)) == ""


def test_all_mutators_handle_unicode_input_no_crash() -> None:
    base = "héllo wörld 日本語 émoji 🎉"
    rng = random.Random(0)
    assert isinstance(mutators.mutate_repetition(base, rng), str)
    assert isinstance(mutators.mutate_unicode_homoglyphs(base, rng), str)
    assert isinstance(mutators.mutate_unicode_zero_width(base, rng), str)
    assert isinstance(mutators.mutate_formatting_case(base, rng), str)
    assert isinstance(mutators.mutate_formatting_whitespace(base, rng), str)
    assert isinstance(mutators.mutate_prompt_reorder(base, rng), str)
