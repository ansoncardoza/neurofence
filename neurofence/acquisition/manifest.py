"""Deterministic file manifest for a model directory.

The manifest is the integrity baseline: a sorted, hashed inventory of every
file under a model root. Building it twice on unchanged files must produce
byte-identical JSON (aside from `generated_at`), and changing a single byte
in any file must change that file's hashes.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, Field

from neurofence.acquisition.formats import FileFormat, classify
from neurofence.acquisition.hashing import hash_file
from neurofence.acquisition.security import resolve_within_root
from neurofence.exceptions import AcquisitionError, PathTraversalError
from neurofence.logging_setup import get_logger

logger = get_logger(__name__)


class FileRecord(BaseModel):
    path: str  # POSIX-style, relative to model root
    format: FileFormat
    size: int = Field(ge=0)
    sha256: str
    sha512: str
    is_symlink: bool = False


class ModelManifest(BaseModel):
    model_name: str
    root_path: str
    file_count: int
    total_size_bytes: int
    generated_at: str
    files: list[FileRecord]

    def file_index(self) -> dict[str, FileRecord]:
        return {f.path: f for f in self.files}


class IntegrityDiscrepancy(BaseModel):
    kind: str  # "hash_mismatch" | "missing_file" | "extra_file" | "size_mismatch"
    path: str
    detail: str


class VerificationResult(BaseModel):
    matched: bool
    discrepancies: list[IntegrityDiscrepancy] = Field(default_factory=list)
    files_checked: int


def build_manifest(model_dir: str | Path, model_name: str | None = None) -> ModelManifest:
    """Walk `model_dir`, hash every regular file, and produce a deterministic
    manifest sorted by relative path.

    Raises AcquisitionError if `model_dir` does not exist, is not a
    directory, or contains no files.
    """
    root = Path(model_dir)
    if not root.exists():
        raise AcquisitionError(f"Model directory does not exist: {root}")
    if not root.is_dir():
        raise AcquisitionError(f"Model path is not a directory: {root}")

    records: list[FileRecord] = []
    total_size = 0

    for entry in sorted(root.rglob("*")):
        if not entry.is_file():
            continue

        rel = entry.relative_to(root)
        rel_posix = rel.as_posix()

        try:
            resolved = resolve_within_root(root, rel_posix)
        except PathTraversalError:
            logger.warning("Skipping file outside model root: %s", rel_posix)
            continue

        is_symlink = entry.is_symlink()
        if is_symlink:
            # A symlink target could point outside the model root even if
            # the link name itself is fine. Confirm the resolved real path
            # also stays within root before hashing.
            real_target = entry.resolve()
            try:
                real_target.relative_to(root.resolve())
            except ValueError:
                logger.warning("Skipping symlink escaping model root: %s", rel_posix)
                continue

        sha256, sha512 = hash_file(resolved)
        size = resolved.stat().st_size
        total_size += size

        records.append(
            FileRecord(
                path=rel_posix,
                format=classify(entry),
                size=size,
                sha256=sha256,
                sha512=sha512,
                is_symlink=is_symlink,
            )
        )

    if not records:
        raise AcquisitionError(f"No files found under model directory: {root}")

    records.sort(key=lambda r: r.path)

    return ModelManifest(
        model_name=model_name or root.name,
        root_path=str(root.resolve()),
        file_count=len(records),
        total_size_bytes=total_size,
        generated_at=datetime.now(UTC).isoformat(),
        files=records,
    )


def verify_manifest(manifest: ModelManifest, model_dir: str | Path) -> VerificationResult:
    """Recompute hashes for `model_dir` and compare against `manifest`.

    A mismatch is reported as an IntegrityDiscrepancy -- it is evidence for
    evidence fusion, not an automatic classification of poisoning.
    """
    current = build_manifest(model_dir, model_name=manifest.model_name)
    current_index = current.file_index()
    expected_index = manifest.file_index()

    discrepancies: list[IntegrityDiscrepancy] = []

    for path, expected in expected_index.items():
        actual = current_index.get(path)
        if actual is None:
            discrepancies.append(
                IntegrityDiscrepancy(
                    kind="missing_file", path=path, detail="File in manifest not found on disk"
                )
            )
            continue
        if actual.size != expected.size:
            discrepancies.append(
                IntegrityDiscrepancy(
                    kind="size_mismatch",
                    path=path,
                    detail=f"expected size {expected.size}, found {actual.size}",
                )
            )
        if actual.sha256 != expected.sha256 or actual.sha512 != expected.sha512:
            discrepancies.append(
                IntegrityDiscrepancy(
                    kind="hash_mismatch",
                    path=path,
                    detail=f"expected sha256 {expected.sha256}, found {actual.sha256}",
                )
            )

    for path in current_index:
        if path not in expected_index:
            discrepancies.append(
                IntegrityDiscrepancy(
                    kind="extra_file", path=path, detail="File on disk not present in manifest"
                )
            )

    return VerificationResult(
        matched=len(discrepancies) == 0,
        discrepancies=discrepancies,
        files_checked=len(expected_index),
    )
