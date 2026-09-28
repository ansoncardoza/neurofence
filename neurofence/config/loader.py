"""Loads and validates NeuroFence configuration from YAML."""

from __future__ import annotations

from pathlib import Path

import yaml

from neurofence.config.schema import Config
from neurofence.exceptions import ConfigError

DEFAULT_CONFIG_PATH = Path(__file__).parent / "default.yaml"


def load_config(path: str | Path | None = None) -> Config:
    """Load configuration from `path`, falling back to bundled defaults.

    Raises ConfigError on missing file, invalid YAML, or schema validation
    failure -- config errors must be loud and never silently ignored, since
    a bad threshold could quietly disable a detector.
    """
    target = Path(path) if path is not None else DEFAULT_CONFIG_PATH

    if not target.exists():
        raise ConfigError(f"Config file not found: {target}")

    try:
        raw = yaml.safe_load(target.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as e:
        raise ConfigError(f"Invalid YAML in {target}: {e}") from e

    if not isinstance(raw, dict):
        raise ConfigError(f"Config file {target} must define a mapping at the top level.")

    try:
        return Config.model_validate(raw)
    except Exception as e:  # pydantic ValidationError, re-raised as domain error
        raise ConfigError(f"Invalid configuration in {target}: {e}") from e
