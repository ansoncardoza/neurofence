from __future__ import annotations

from neurofence.behavioral.comparison import compare_outputs
from neurofence.behavioral.embeddings import TfidfSimilarityBackend

BACKEND = TfidfSimilarityBackend()


def test_identical_outputs() -> None:
    text = "Paris is the capital of France."
    result = compare_outputs(text, text, BACKEND)
    assert result.exact_match is True
    assert result.semantic_similarity == 1.0
    assert result.semantic_distance == 0.0
    assert result.refusal_changed is False
    assert result.length_ratio == 1.0


def test_refusal_appearing_is_flagged() -> None:
    result = compare_outputs(
        "The capital of France is Paris.",
        "I cannot help with that request.",
        BACKEND,
    )
    assert result.refusal_baseline.detected is False
    assert result.refusal_test.detected is True
    assert result.refusal_changed is True


def test_refusal_disappearing_is_flagged() -> None:
    result = compare_outputs(
        "I'm sorry, I can't help with that.",
        "Sure, here is the answer.",
        BACKEND,
    )
    assert result.refusal_changed is True


def test_both_outputs_empty() -> None:
    result = compare_outputs("", "", BACKEND)
    assert result.exact_match is True
    assert result.length_baseline == 0
    assert result.length_test == 0
    assert result.length_ratio == 1.0
    assert result.length_ratio_note is not None


def test_baseline_empty_test_nonempty_ratio_undefined() -> None:
    result = compare_outputs("", "some output", BACKEND)
    assert result.length_ratio is None
    assert result.length_ratio_note is not None
    assert result.exact_match is False


def test_length_ratio_computed_correctly() -> None:
    result = compare_outputs("abcde", "abcdeabcde", BACKEND)
    assert result.length_baseline == 5
    assert result.length_test == 10
    assert result.length_ratio == 2.0


def test_totally_different_text_low_similarity() -> None:
    result = compare_outputs(
        "The mitochondria is the powerhouse of the cell.",
        "Quarterly revenue exceeded analyst expectations this year.",
        BACKEND,
    )
    assert result.semantic_similarity < 0.5
    assert result.exact_match is False
