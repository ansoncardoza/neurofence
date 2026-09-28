"""Streaming, deterministic file hashing.

Uses chunked reads so hashing does not require loading an entire (possibly
multi-gigabyte) weight file into memory at once.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

_CHUNK_SIZE = 4 * 1024 * 1024  # 4 MiB


def hash_file(path: Path) -> tuple[str, str]:
    """Return (sha256_hex, sha512_hex) for the file at `path`.

    Both digests are computed in a single streaming pass over the file.
    """
    sha256 = hashlib.sha256()
    sha512 = hashlib.sha512()

    with open(path, "rb") as f:
        while chunk := f.read(_CHUNK_SIZE):
            sha256.update(chunk)
            sha512.update(chunk)

    return sha256.hexdigest(), sha512.hexdigest()
