from __future__ import annotations

from neurofence.behavioral.embeddings import TfidfSimilarityBackend


def test_identical_text_similarity_one() -> None:
    backend = TfidfSimilarityBackend()
    assert backend.similarity("The cat sat on the mat.", "The cat sat on the mat.") == 1.0


def test_both_empty_similarity_one() -> None:
    backend = TfidfSimilarityBackend()
    assert backend.similarity("", "") == 1.0


def test_one_empty_similarity_zero() -> None:
    backend = TfidfSimilarityBackend()
    assert backend.similarity("some text", "") == 0.0
    assert backend.similarity("", "some text") == 0.0


def test_whitespace_only_treated_as_empty() -> None:
    backend = TfidfSimilarityBackend()
    assert backend.similarity("   ", "   ") == 1.0


def test_completely_unrelated_text_low_similarity() -> None:
    backend = TfidfSimilarityBackend()
    sim = backend.similarity(
        "Photosynthesis converts sunlight into chemical energy in plants.",
        "The stock market fell sharply after the interest rate announcement.",
    )
    assert sim < 0.5


def test_paraphrase_with_shared_words_has_positive_similarity() -> None:
    backend = TfidfSimilarityBackend()
    sim = backend.similarity(
        "The quick brown fox jumps over the lazy dog.",
        "The quick brown fox leaps over the lazy dog.",
    )
    assert sim > 0.5


def test_similarity_always_in_bounds() -> None:
    backend = TfidfSimilarityBackend()
    pairs = [
        ("a", "b"),
        ("aaaaaa", "a"),
        ("!!!", "???"),
        ("123", "456"),
        ("x", "x x x x x"),
    ]
    for a, b in pairs:
        sim = backend.similarity(a, b)
        assert 0.0 <= sim <= 1.0


def test_punctuation_only_no_crash() -> None:
    backend = TfidfSimilarityBackend()
    sim = backend.similarity("!!!???...", "...???!!!")
    assert 0.0 <= sim <= 1.0


def test_very_long_text_no_crash() -> None:
    backend = TfidfSimilarityBackend()
    a = "word " * 5000
    b = "word " * 5000 + "different"
    sim = backend.similarity(a, b)
    assert 0.0 <= sim <= 1.0


def test_similarity_is_symmetric() -> None:
    backend = TfidfSimilarityBackend()
    a, b = "the sun rises in the east", "the moon sets in the west"
    assert backend.similarity(a, b) == backend.similarity(b, a)
