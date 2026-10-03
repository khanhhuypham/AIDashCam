"""App iOS (sau pipeline) - export the runtime thresholds as JSON for the Swift app.

Keys stay ``snake_case``; decode in Swift with
``JSONDecoder().keyDecodingStrategy = .convertFromSnakeCase``.
"""

from __future__ import annotations

from pathlib import Path

from dashcam_ml.config.schema import DashcamConfig
from dashcam_ml.export.manifest import read_manifest
from dashcam_ml.utils.files import write_json
from dashcam_ml.utils.logging_utils import get_logger

LOGGER = get_logger(name=__name__)
IOS_CONFIG_SCHEMA_VERSION: int = 1


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
        "models": [entry.to_dict() for entry in read_manifest(output_dir=config.coreml_export.output_dir)],
    }


def export_ios_config(*, config: DashcamConfig) -> Path:
    output_path: Path = config.ios_config_export.output_path
    write_json(path=output_path, payload=build_ios_config(config=config))
    LOGGER.info("iOS config written to %s (add it to the Xcode target bundle)", output_path)
    return output_path
