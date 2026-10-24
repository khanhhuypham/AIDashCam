"""Read ``config.yaml`` from disk into a validated :class:`DashcamConfig`."""

from __future__ import annotations

from pathlib import Path

import yaml

from dashcam_ml.config.schema import BASE_DIR_CONTEXT_KEY, DashcamConfig

DEFAULT_CONFIG_PATH: Path = Path("config.yaml")


def load_config(*, config_path: Path = DEFAULT_CONFIG_PATH) -> DashcamConfig:
    resolved_path: Path = config_path.resolve()
    if not resolved_path.is_file():
        raise FileNotFoundError(f"config file not found: {resolved_path}")
    with resolved_path.open(mode="r", encoding="utf-8") as handle:
        raw: object = yaml.safe_load(stream=handle)
    if not isinstance(raw, dict):
        raise ValueError(f"{resolved_path} must contain a YAML mapping at the top level")
    return DashcamConfig.model_validate(obj=raw, context={BASE_DIR_CONTEXT_KEY: str(resolved_path.parent)})
