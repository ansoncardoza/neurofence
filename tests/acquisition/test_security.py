from __future__ import annotations

from pathlib import Path

import pytest

from neurofence.acquisition.security import is_safe_relative_path, resolve_within_root
from neurofence.exceptions import PathTraversalError


def test_resolves_normal_relative_path(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    (root / "sub").mkdir()

    resolved = resolve_within_root(root, "sub/file.txt")

    assert resolved == (root / "sub" / "file.txt").resolve()


def test_rejects_dotdot_traversal(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()

    with pytest.raises(PathTraversalError):
        resolve_within_root(root, "../../etc/passwd")


def test_rejects_absolute_path(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()

    with pytest.raises(PathTraversalError):
        resolve_within_root(root, "/etc/passwd")


def test_rejects_deeply_nested_traversal(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()

    with pytest.raises(PathTraversalError):
        resolve_within_root(root, "a/b/c/../../../../../../outside.txt")


def test_is_safe_relative_path_boolean_form(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()

    assert is_safe_relative_path(root, "ok.txt") is True
    assert is_safe_relative_path(root, "../escape.txt") is False


def test_windows_style_traversal_rejected(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()

    with pytest.raises(PathTraversalError):
        resolve_within_root(root, "..\\..\\windows\\system32")
