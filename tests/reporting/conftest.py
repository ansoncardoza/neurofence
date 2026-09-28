from __future__ import annotations

import pytest

from neurofence.acquisition.manifest import FileRecord, ModelManifest
from neurofence.acquisition.metadata import ModelMetadata
from neurofence.fusion.combine import fuse_evidence
from neurofence.fusion.scores import SubScore
from neurofence.reporting.report import ScanReport, build_report


@pytest.fixture
def manifest() -> ModelManifest:
    return ModelManifest(
        model_name="test-model",
        root_path="/fake/path",
        file_count=1,
        total_size_bytes=1024,
        generated_at="2026-01-01T00:00:00+00:00",
        files=[
            FileRecord(
                path="model.safetensors",
                format="safetensors",
                size=1024,
                sha256="a" * 64,
                sha512="b" * 128,
            )
        ],
    )


@pytest.fixture
def metadata() -> ModelMetadata:
    return ModelMetadata(
        architecture="TestArch",
        model_type="test",
        parameter_count=1000,
        tensor_count=5,
        layer_count=2,
        weight_format="safetensors",
        shard_count=1,
        config={"architectures": ["TestArch"]},
        warnings=[],
    )


def make_sub_scores(**overrides: SubScore | None) -> dict[str, SubScore]:
    base = {
        name: SubScore(name=name, status="not_evaluated")
        for name in ("integrity", "weight_anomaly", "behavioral", "activation", "trigger")
    }
    for name, value in overrides.items():
        if value is not None:
            base[name] = value
    return base


@pytest.fixture
def not_evaluated_fusion():
    from neurofence.config.schema import RiskWeights

    return fuse_evidence(make_sub_scores(), RiskWeights())


def minimal_report(manifest: ModelManifest, metadata: ModelMetadata, fusion) -> ScanReport:
    return build_report(
        model_path="/fake/path", manifest=manifest, metadata=metadata, evidence_fusion=fusion
    )
