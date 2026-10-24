"""Config layer: schema of ``config.yaml`` and its loader."""

from dashcam_ml.config.loader import DEFAULT_CONFIG_PATH, load_config
from dashcam_ml.config.schema import DashcamConfig

__all__ = ["DEFAULT_CONFIG_PATH", "DashcamConfig", "load_config"]
