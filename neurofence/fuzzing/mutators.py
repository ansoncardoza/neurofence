"""Prompt mutation primitives, grouped by category.

Every mutator is a pure function of its inputs plus an explicit
`random.Random` instance (never the global `random` module), so an entire
fuzzing run is reproducible from a single seed. None of these mutators
claim to produce a "malicious" prompt -- they produce structurally unusual
input; whether a model's response to it is evidence of anything is decided
downstream by behavioral comparison, never here.
"""

from __future__ import annotations

import random
import string

# A small, documented set of Latin-lookalike characters from other scripts,
# used for homoglyph mutation (a known prompt-injection/obfuscation vector).
_HOMOGLYPHS: dict[str, str] = {
    "a": "а",  # Cyrillic а
    "e": "е",  # Cyrillic е
    "o": "о",  # Cyrillic о
    "p": "р",  # Cyrillic р
    "c": "с",  # Cyrillic с
    "x": "х",  # Cyrillic х
    "y": "у",  # Cyrillic у
    "i": "і",  # Cyrillic і
}

_ZERO_WIDTH_CHARS = ["​", "‌", "‍", "﻿"]  # ZWSP, ZWNJ, ZWJ, BOM
_MIXED_SCRIPT_CHARS = list("ΑΒΓΔαβγ一二三")  # Greek, CJK
_PRINTABLE_NO_CONTROL = "".join(c for c in string.printable if c not in "\t\n\r\x0b\x0c")


def mutate_random_text(rng: random.Random, length: int = 30) -> str:
    """Standalone fuzz input with no relation to any real prompt."""
    return "".join(rng.choices(_PRINTABLE_NO_CONTROL, k=length))


def mutate_repetition(
    base: str, rng: random.Random, min_reps: int = 10, max_reps: int = 100
) -> str:
    words = base.split() or ["word"]
    unit = rng.choice(words)
    reps = rng.randint(min_reps, max_reps)
    return " ".join([unit] * reps)


def mutate_unicode_homoglyphs(base: str, rng: random.Random, fraction: float = 0.3) -> str:
    chars = list(base)
    candidate_positions = [i for i, c in enumerate(chars) if c.lower() in _HOMOGLYPHS]
    n_to_replace = max(1, int(len(candidate_positions) * fraction)) if candidate_positions else 0
    for i in rng.sample(candidate_positions, min(n_to_replace, len(candidate_positions))):
        chars[i] = _HOMOGLYPHS[chars[i].lower()]
    return "".join(chars)


def mutate_unicode_zero_width(base: str, rng: random.Random, fraction: float = 0.2) -> str:
    if not base:
        return "".join(rng.choices(_ZERO_WIDTH_CHARS, k=3))
    chars = list(base)
    n_insertions = max(1, int(len(chars) * fraction))
    for _ in range(n_insertions):
        pos = rng.randint(0, len(chars))
        chars.insert(pos, rng.choice(_ZERO_WIDTH_CHARS))
    return "".join(chars)


def mutate_unicode_mixed_scripts(base: str, rng: random.Random, fraction: float = 0.2) -> str:
    if not base:
        return "".join(rng.choices(_MIXED_SCRIPT_CHARS, k=5))
    chars = list(base)
    n_insertions = max(1, int(len(chars) * fraction))
    for _ in range(n_insertions):
        pos = rng.randint(0, len(chars))
        chars.insert(pos, rng.choice(_MIXED_SCRIPT_CHARS))
    return "".join(chars)


def mutate_formatting_case(base: str, rng: random.Random) -> str:
    mode = rng.choice(["upper", "lower", "alternating", "swapcase"])
    if mode == "upper":
        return base.upper()
    if mode == "lower":
        return base.lower()
    if mode == "swapcase":
        return base.swapcase()
    return "".join(c.upper() if i % 2 == 0 else c.lower() for i, c in enumerate(base))


def mutate_formatting_whitespace(base: str, rng: random.Random, fraction: float = 0.3) -> str:
    if not base:
        return "   \n\t  "
    chars = list(base)
    n_insertions = max(1, int(len(chars) * fraction))
    whitespace_variants = [" ", "  ", "\n", "\t", "\n\n"]
    for _ in range(n_insertions):
        pos = rng.randint(0, len(chars))
        chars.insert(pos, rng.choice(whitespace_variants))
    return "".join(chars)


def mutate_formatting_punctuation(base: str, rng: random.Random) -> str:
    mode = rng.choice(["strip", "excessive", "replace_separators"])
    if mode == "strip":
        return "".join(c for c in base if c not in string.punctuation)
    if mode == "excessive":
        punct = rng.choice(["!!!", "???", "...", "***", "###"])
        return f"{base}{punct}"
    separators = [",", ";", "|", " / ", " -- "]
    words = base.split()
    if len(words) < 2:
        return base
    sep = rng.choice(separators)
    return sep.join(words)


def mutate_prompt_insert(base: str, insertion: str, rng: random.Random) -> str:
    words = base.split()
    if not words:
        return insertion
    pos = rng.randint(0, len(words))
    return " ".join([*words[:pos], insertion, *words[pos:]])


def mutate_prompt_prepend(base: str, insertion: str) -> str:
    return f"{insertion} {base}".strip()


def mutate_prompt_append(base: str, insertion: str) -> str:
    return f"{base} {insertion}".strip()


def mutate_prompt_replace(base: str, rng: random.Random, replacement: str) -> str:
    words = base.split()
    if not words:
        return replacement
    pos = rng.randrange(len(words))
    words[pos] = replacement
    return " ".join(words)


def mutate_prompt_reorder(base: str, rng: random.Random) -> str:
    words = base.split()
    if len(words) < 2:
        return base
    shuffled = words[:]
    rng.shuffle(shuffled)
    return " ".join(shuffled)
