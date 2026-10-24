"""Export the runtime thresholds as ``DashcamConfig.json`` for the Swift app.

Keys stay ``snake_case``; decode in Swift with
``JSONDecoder().keyDecodingStrategy = .convertFromSnakeCase``. The structure must not
change: the published app reads this file.
"""

from __future__ import annotations

from pathlib import Path

from dashcam_video.config.schema import DashcamConfig
from dashcam_video.utils.files import read_json, write_json
from dashcam_video.utils.logging_utils import get_logger

LOGGER = get_logger(name=__name__)
IOS_CONFIG_SCHEMA_VERSION: int = 1


def read_manifest_models(*, manifest_path: Path) -> list[object]:
    """The ``models`` list of dashcam-train's ``manifest.json``; empty when not exported yet."""
    if not manifest_path.is_file():
        LOGGER.warning("models manifest not found at %s: 'models' will be empty", manifest_path)
        return []
    data: object = read_json(path=manifest_path)
    if not isinstance(data, dict):
        raise ValueError(f"{manifest_path} must contain a JSON object")
    models: object = data.get("models", [])
    if not isinstance(models, list):
        raise ValueError(f"'models' in {manifest_path} must be a list")
    return models


def build_ios_config(*, config: DashcamConfig) -> dict[str, object]:
    return {
        "schema_version": IOS_CONFIG_SCHEMA_VERSION,
        "camera": config.camera.model_dump(mode="json"),
        "classes": [spec.model_dump(mode="json") for spec in config.classes],
        "inference": config.inference.model_dump(mode="json", include={"conf", "iou", "max_det"}),
        "tracker": config.tracker.model_dump(mode="json"),
        "fcw": config.fcw.model_dump(mode="json"),
        "ldw": config.ldw.model_dump(mode="json"),
        "lane_segmenter": config.twinlitenet.model_dump(
            mode="json", include={"input_width", "input_height", "mask_threshold"}
        ),
        "models": read_manifest_models(manifest_path=config.ios_config_export.models_manifest),
    }


def export_ios_config(*, config: DashcamConfig) -> Path:
    output_path: Path = config.ios_config_export.output_path
    write_json(path=output_path, payload=build_ios_config(config=config))
    LOGGER.info("iOS config written to %s (add it to the Xcode target bundle)", output_path)
    return output_path
