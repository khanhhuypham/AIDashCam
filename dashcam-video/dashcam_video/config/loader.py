"""Read ``config.yaml`` from disk into a validated :class:`RootConfig`."""

from __future__ import annotations

from pathlib import Path

import yaml

from dashcam_video.config.schema import BASE_DIR_CONTEXT_KEY, RootConfig

DEFAULT_CONFIG_PATH: Path = Path("config.yaml")


def load_config(*, config_path: Path = DEFAULT_CONFIG_PATH) -> RootConfig:
    """Read, validate, and load the application configuration from a YAML file.

    Args:
        config_path: Path to the configuration YAML file (defaults to ``config.yaml``).

    Returns:
        RootConfig: A fully validated configuration data model instance.

    Raises:
        FileNotFoundError: If the configuration file does not exist at the specified path.
        ValueError: If the top-level structure of the YAML file is not a dictionary mapping.
    """
    # 1. Resolve the configuration file path to an absolute path
    resolved_path: Path = config_path.resolve()

    # 2. Check if the configuration file actually exists on disk
    if not resolved_path.is_file():
        raise FileNotFoundError(f"Configuration file not found at the resolved path: {resolved_path}")

    # 3. Open the file and parse its content into raw Python data structures via PyYAML
    with resolved_path.open(mode="r", encoding="utf-8") as handle:
        raw_data: object = yaml.safe_load(stream=handle)

    # 4. Ensure the top-level structure of the YAML file is a key-value mapping (dictionary)
    if not isinstance(raw_data, dict):
        raise ValueError(f"Configuration file at {resolved_path} must contain a dictionary mapping at the top level.")

    # 5. Validate the raw data using Pydantic and pass the parent directory context for relative paths
    config_context = {BASE_DIR_CONTEXT_KEY: str(resolved_path.parent)}

    return RootConfig.model_validate(obj=raw_data, context=config_context)
