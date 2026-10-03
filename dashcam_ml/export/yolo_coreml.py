"""CV14 - export YOLO to Core ML ``.mlpackage`` (one file per profile in config).

Uses the official Ultralytics exporter. With ``nms: true`` the package is a Core ML
pipeline (model + NMS) that Vision turns directly into ``VNRecognizedObjectObservation``.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from dashcam_ml.config.schema import CoremlExportConfig, CoremlProfile, ImageSize
from dashcam_ml.domain.enums import ModelKind
from dashcam_ml.export.manifest import ModelManifestEntry, upsert_manifest
from dashcam_ml.models.yolo_detector import to_ultralytics_imgsz, ultralytics_precision_kwargs
from dashcam_ml.utils.files import remove_path
from dashcam_ml.utils.logging_utils import get_logger
from dashcam_ml.utils.runtime import require_coreml_conversion_platform

LOGGER = get_logger(name=__name__)
MLPACKAGE_SUFFIX: str = ".mlpackage"
COREML_FORMAT: str = "coreml"
NMS_INPUT_NAMES: list[str] = ["image", "iouThreshold", "confidenceThreshold"]
NMS_OUTPUT_NAMES: list[str] = ["confidence", "coordinates"]
RAW_INPUT_NAMES: list[str] = ["image"]


def image_size_hw(*, imgsz: ImageSize) -> tuple[int, int]:
    return (imgsz, imgsz) if isinstance(imgsz, int) else (imgsz[0], imgsz[1])


def export_yolo_coreml(*, config: CoremlExportConfig) -> Path:
    require_coreml_conversion_platform()
    from ultralytics import YOLO

    config.output_dir.mkdir(parents=True, exist_ok=True)
    entries: list[ModelManifestEntry] = []
    for profile in config.profiles:
        LOGGER.info("exporting profile %s (imgsz=%s, %s)", profile.name, profile.imgsz, profile.precision.value)
        model: YOLO = YOLO(model=str(config.weights))
        exported_path: Path = Path(
            str(
                model.export(
                    format=COREML_FORMAT,
                    imgsz=to_ultralytics_imgsz(imgsz=profile.imgsz),
                    nms=config.nms,
                    batch=1,
                    **ultralytics_precision_kwargs(precision=profile.precision),
                )
            )
        )
        destination: Path = config.output_dir / f"{profile.name}{MLPACKAGE_SUFFIX}"
        remove_path(path=destination)
        shutil.move(src=str(exported_path), dst=str(destination))
        entries.append(
            _manifest_entry(
                profile=profile,
                destination=destination,
                weights=config.weights,
                class_names=[str(model.names[index]) for index in sorted(model.names)],
                nms=config.nms,
            )
        )
        LOGGER.info("saved %s", destination)
    manifest_path: Path = upsert_manifest(output_dir=config.output_dir, entries=entries)
    LOGGER.info("manifest updated: %s", manifest_path)
    return manifest_path


def _manifest_entry(
    *, profile: CoremlProfile, destination: Path, weights: Path, class_names: list[str], nms: bool
) -> ModelManifestEntry:
    height, width = image_size_hw(imgsz=profile.imgsz)
    return ModelManifestEntry(
        name=profile.name,
        kind=ModelKind.DETECTOR,
        file_name=destination.name,
        input_width=width,
        input_height=height,
        precision=profile.precision,
        source_weights=weights.name,
        target_devices=profile.target_devices,
        input_names=NMS_INPUT_NAMES if nms else RAW_INPUT_NAMES,
        output_names=NMS_OUTPUT_NAMES if nms else [],
        class_names=class_names,
        includes_nms=nms,
    )
