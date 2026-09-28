"""Path-safety helpers for handling untrusted model directories.

A model repository is untrusted input. File names inside a manifest, a
config.json, or a downloaded snapshot must never be allowed to resolve
outside the intended root -- that is a classic path-traversal vector
(`../../etc/passwd`, absolute paths, symlink escapes).
"""

from __future__ import annotations

from pathlib import Path

from neurofence.exceptions import PathTraversalError


def resolve_within_root(root: Path, candidate: str | Path) -> Path:
    """Resolve `candidate` (relative or absolute) against `root` and verify
    the result stays inside `root` after following symlinks.

    Raises PathTraversalError if it does not.
    """
    root_resolved = root.resolve()
    candidate_path = Path(candidate)

    # Reject absolute candidates outright -- a manifest entry or config field
    # should never claim an absolute filesystem path.
    if candidate_path.is_absolute():
        raise PathTraversalError(f"Absolute path not allowed: {candidate}")

    joined = (root_resolved / candidate_path).resolve()

    try:
        joined.relative_to(root_resolved)
    except ValueError as e:
        raise PathTraversalError(
            f"Path '{candidate}' escapes model root '{root_resolved}' "
            f"(resolved to '{joined}')"
        ) from e

    return joined


def is_safe_relative_path(root: Path, candidate: str | Path) -> bool:
    """Non-raising boolean form of resolve_within_root, for filtering."""
    try:
        resolve_within_root(root, candidate)
        return True
    except PathTraversalError:
        return False
