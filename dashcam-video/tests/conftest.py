from __future__ import annotations

from pathlib import Path

import pytest

from dashcam_video.config.loader import load_config
from dashcam_video.config.schema import RootConfig

PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def config() -> RootConfig:
    return load_config(config_path=PROJECT_ROOT / "config.yaml")
