from __future__ import annotations

from neurofence.behavioral.refusal import detect_refusal


def test_detects_common_refusal_phrase() -> None:
    result = detect_refusal("I'm sorry, but I can't help with that request.")
    assert result.detected is True
    assert result.matched_text is not None


def test_detects_cannot_contraction() -> None:
    result = detect_refusal("I cannot provide instructions for that.")
    assert result.detected is True


def test_no_false_positive_on_normal_answer() -> None:
    result = detect_refusal("The capital of France is Paris.")
    assert result.detected is False


def test_empty_string() -> None:
    result = detect_refusal("")
    assert result.detected is False


def test_case_insensitive() -> None:
    result = detect_refusal("I CANNOT do that.")
    assert result.detected is True


def test_as_an_ai_pattern() -> None:
    result = detect_refusal("As an AI language model, I cannot browse the internet.")
    assert result.detected is True


def test_policy_pattern() -> None:
    result = detect_refusal("That request goes against our policy.")
    assert result.detected is True


def test_long_text_with_refusal_buried_inside() -> None:
    text = "Here is some context. " * 50 + "I must decline to answer that." + " More text." * 20
    result = detect_refusal(text)
    assert result.detected is True


def test_unicode_text_no_crash() -> None:
    # Non-English refusal phrasing is not covered by these English-only
    # heuristics -- a documented limitation, not a bug.
    result = detect_refusal("こんにちは、私はそれを手伝うことができません。")
    assert result.detected is False
