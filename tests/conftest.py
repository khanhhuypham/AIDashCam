from __future__ import annotations

from pathlib import Path

import pytest

from dashcam_ml.config.loader import load_config
from dashcam_ml.config.schema import DashcamConfig

PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def config() -> DashcamConfig:
    return load_config(config_path=PROJECT_ROOT / "config.yaml")
