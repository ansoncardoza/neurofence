from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from neurofence.config import Config, load_config
from neurofence.config.schema import ModelConfig, RiskConfig
from neurofence.exceptions import ConfigError


def test_default_config_loads() -> None:
    config = load_config()
    assert isinstance(config, Config)
    assert config.model.trust_remote_code is False


def test_missing_file_raises_config_error(tmp_path: Path) -> None:
    with pytest.raises(ConfigError):
        load_config(tmp_path / "nope.yaml")


def test_invalid_yaml_raises_config_error(tmp_path: Path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text("model: [unclosed", encoding="utf-8")
    with pytest.raises(ConfigError):
        load_config(bad)


def test_non_mapping_yaml_raises_config_error(tmp_path: Path) -> None:
    bad = tmp_path / "list.yaml"
    bad.write_text("- 1\n- 2\n", encoding="utf-8")
    with pytest.raises(ConfigError):
        load_config(bad)


def test_trust_remote_code_true_is_rejected() -> None:
    with pytest.raises(ValidationError):
        ModelConfig(trust_remote_code=True)


def test_risk_bands_must_cover_full_range() -> None:
    with pytest.raises(ValidationError):
        RiskConfig.model_validate(
            {
                "bands": [
                    {"label": "LOW", "min_score": 0, "max_score": 50},
                    {"label": "HIGH", "min_score": 60, "max_score": 100},
                ]
            }
        )


def test_risk_bands_reject_overlap() -> None:
    with pytest.raises(ValidationError):
        RiskConfig.model_validate(
            {
                "bands": [
                    {"label": "LOW", "min_score": 0, "max_score": 60},
                    {"label": "HIGH", "min_score": 50, "max_score": 100},
                ]
            }
        )


def test_classify_score_bands() -> None:
    risk = RiskConfig()
    assert risk.classify(0) == "LOW"
    assert risk.classify(25) == "LOW"
    assert risk.classify(26) == "MEDIUM"
    assert risk.classify(75) == "HIGH"
    assert risk.classify(76) == "CRITICAL"
    assert risk.classify(100) == "CRITICAL"


def test_classify_clamps_out_of_range_scores() -> None:
    risk = RiskConfig()
    assert risk.classify(-10) == "LOW"
    assert risk.classify(1000) == "CRITICAL"


def test_all_zero_weights_rejected() -> None:
    from neurofence.config.schema import RiskWeights

    with pytest.raises(ValidationError):
        RiskWeights(integrity=0, weight_anomaly=0, behavioral=0, activation=0, trigger=0)


def test_custom_config_overrides_defaults(tmp_path: Path) -> None:
    custom = tmp_path / "custom.yaml"
    custom.write_text(
        "fuzzing:\n  enabled: true\n  max_prompts: 10\nrandom_seed: 42\n",
        encoding="utf-8",
    )
    config = load_config(custom)
    assert config.fuzzing.enabled is True
    assert config.fuzzing.max_prompts == 10
    assert config.random_seed == 42
    # unspecified sections still get their defaults
    assert config.model.device == "auto"
