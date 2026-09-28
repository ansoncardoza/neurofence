"""Model file format classification.

NeuroFence must never unpickle untrusted weight files: Python's `pickle`
(the format behind legacy PyTorch `.bin`/`.pt` checkpoints) can execute
arbitrary code on load. We classify formats so downstream code knows which
files are safe to introspect (safetensors: safe, memory-mapped, no code
execution) versus which can only be hashed and flagged (pickle-based).
"""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path


class FileFormat(StrEnum):
    SAFETENSORS = "safetensors"
    PICKLE_WEIGHTS = "pickle_weights"  # .bin / .pt / .pth -- unsafe to unpickle
    CONFIG_JSON = "config_json"
    TOKENIZER = "tokenizer"
    TEXT = "text"
    OTHER = "other"


_PICKLE_SUFFIXES = {".bin", ".pt", ".pth", ".ckpt"}
_TOKENIZER_NAMES = {
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "special_tokens_map.json",
    "spiece.model",
}
_TEXT_SUFFIXES = {".txt", ".md"}


def classify(path: Path) -> FileFormat:
    name = path.name
    suffix = path.suffix.lower()

    if suffix == ".safetensors":
        return FileFormat.SAFETENSORS
    if name == "config.json":
        return FileFormat.CONFIG_JSON
    if name in _TOKENIZER_NAMES:
        return FileFormat.TOKENIZER
    if suffix in _PICKLE_SUFFIXES:
        return FileFormat.PICKLE_WEIGHTS
    if suffix in _TEXT_SUFFIXES:
        return FileFormat.TEXT
    return FileFormat.OTHER


def is_safe_to_introspect(fmt: FileFormat) -> bool:
    """Whether NeuroFence may open and parse the file's structured content
    (beyond hashing). Pickle-based weight files are hashed only -- never
    unpickled -- because deserializing them can execute arbitrary code.
    """
    return fmt in (
        FileFormat.SAFETENSORS,
        FileFormat.CONFIG_JSON,
        FileFormat.TOKENIZER,
        FileFormat.TEXT,
    )
