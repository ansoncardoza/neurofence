"""Typed configuration schema for NeuroFence.

Every threshold and weight that affects a security determination lives here,
not scattered through the codebase, so a reviewer can audit the full set of
tunable assumptions in one place.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator


class ModelConfig(BaseModel):
    device: Literal["auto", "cpu", "cuda"] = "auto"
    trust_remote_code: bool = Field(
        default=False,
        description="Must stay False: executing model-provided code is an RCE risk.",
    )

    @field_validator("trust_remote_code")
    @classmethod
    def _forbid_remote_code(cls, v: bool) -> bool:
        if v:
            raise ValueError(
                "trust_remote_code=True is disallowed by NeuroFence policy: "
                "loading untrusted models must never execute arbitrary code."
            )
        return v


class AnalysisConfig(BaseModel):
    weight_analysis: bool = True
    behavioral_analysis: bool = True
    activation_analysis: bool = True
    max_layers_sampled: int | None = Field(
        default=None,
        description="Cap on layers analyzed for spectral/anomaly detection; None = all.",
    )


class FuzzingConfig(BaseModel):
    enabled: bool = False
    max_prompts: int = Field(default=500, gt=0, le=100_000)
    categories: list[str] = Field(
        default_factory=lambda: [
            "random_text",
            "repetition",
            "unicode",
            "formatting",
            "prompt_mutation",
        ]
    )
    candidate_triggers: list[str] = Field(default_factory=list)


class ActivationConfig(BaseModel):
    enabled: bool = False
    selected_layers: Literal["auto"] | list[str] = "auto"
    max_samples_per_layer: int = Field(default=256, gt=0)


class RiskWeights(BaseModel):
    """Relative weights in the ThreatEvidence fusion formula.

    Not required to sum to 1.0 -- evidence fusion normalizes by the sum of
    weights actually contributing (i.e. whose detector ran), so a disabled
    detector does not silently zero out the final score.
    """

    integrity: float = Field(default=0.15, ge=0)
    weight_anomaly: float = Field(default=0.25, ge=0)
    behavioral: float = Field(default=0.25, ge=0)
    activation: float = Field(default=0.20, ge=0)
    trigger: float = Field(default=0.15, ge=0)

    @model_validator(mode="after")
    def _nonzero_total(self) -> RiskWeights:
        total = (
            self.integrity
            + self.weight_anomaly
            + self.behavioral
            + self.activation
            + self.trigger
        )
        if total <= 0:
            raise ValueError("At least one risk weight must be positive.")
        return self


class RiskBand(BaseModel):
    label: Literal["LOW", "MEDIUM", "HIGH", "CRITICAL"]
    min_score: int = Field(ge=0, le=100)
    max_score: int = Field(ge=0, le=100)

    @model_validator(mode="after")
    def _valid_range(self) -> RiskBand:
        if self.min_score > self.max_score:
            raise ValueError(f"Risk band {self.label}: min_score > max_score")
        return self


class RiskConfig(BaseModel):
    weights: RiskWeights = Field(default_factory=RiskWeights)
    bands: list[RiskBand] = Field(
        default_factory=lambda: [
            RiskBand(label="LOW", min_score=0, max_score=25),
            RiskBand(label="MEDIUM", min_score=26, max_score=50),
            RiskBand(label="HIGH", min_score=51, max_score=75),
            RiskBand(label="CRITICAL", min_score=76, max_score=100),
        ]
    )

    @model_validator(mode="after")
    def _bands_cover_0_100_without_gaps(self) -> RiskConfig:
        ordered = sorted(self.bands, key=lambda b: b.min_score)
        if ordered[0].min_score != 0 or ordered[-1].max_score != 100:
            raise ValueError("Risk bands must cover the full 0-100 range.")
        # Deliberate pairwise zip: ordered[1:] is one shorter by construction.
        for prev, nxt in zip(ordered, ordered[1:], strict=False):
            if nxt.min_score != prev.max_score + 1:
                raise ValueError(
                    f"Risk bands have a gap/overlap between {prev.label} and {nxt.label}."
                )
        return self

    def classify(self, score: float) -> str:
        clamped = max(0, min(100, round(score)))
        for band in self.bands:
            if band.min_score <= clamped <= band.max_score:
                return band.label
        raise AssertionError("unreachable: bands validated to cover 0-100")


class Config(BaseModel):
    model: ModelConfig = Field(default_factory=ModelConfig)
    analysis: AnalysisConfig = Field(default_factory=AnalysisConfig)
    fuzzing: FuzzingConfig = Field(default_factory=FuzzingConfig)
    activation: ActivationConfig = Field(default_factory=ActivationConfig)
    risk: RiskConfig = Field(default_factory=RiskConfig)
    random_seed: int = 1337
