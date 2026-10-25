"""Export YOLO to TFLite (LiteRT) ``.tflite`` for the Android app (one file per profile in config).

Uses the official Ultralytics exporter (``format="litert"``). LiteRT cannot embed NMS, so the
model returns raw anchors and the Android app runs NMS itself (see ``output_layout`` in the
manifest). An fp32 model runs in fp16 on the Android GPU delegate; int8 profiles are
calibrated on the ``val`` images of ``calibration_data``.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from dashcam_ml.config.schema import TfliteExportConfig, TfliteProfile
from dashcam_ml.domain.enums import ModelKind, WeightPrecision
from dashcam_ml.export.manifest import ModelManifestEntry, android_manifest_header, upsert_manifest
from dashcam_ml.export.yolo_coreml import image_size_hw
from dashcam_ml.models.ultralytics_utils import to_ultralytics_imgsz, ultralytics_precision_kwargs
from dashcam_ml.utils.files import remove_path
from dashcam_ml.utils.logging_utils import get_logger
from dashcam_ml.utils.runtime import require_litert_platform

LOGGER = get_logger(name=__name__)
TFLITE_SUFFIX: str = ".tflite"
LITERT_FORMAT: str = "litert"
BOX_CHANNELS: int = 4
TensorInfo = tuple[str, list[int]]
"""``(name, shape)`` of one model input or output, read back from the ``.tflite`` file."""


def output_layout(*, input_shape: list[int], output_shape: list[int], class_count: int) -> str:
    return (
        f"input float32 {input_shape}: RGB scaled to 0..1, letterboxed (pad 114). "
        f"output float32 {output_shape}: {BOX_CHANNELS + class_count} values per anchor = cx, cy, w, h "
        f"normalised 0..1, then {class_count} class scores in class_names order; NMS not included"
    )


def export_yolo_tflite(*, config: TfliteExportConfig) -> Path:
    require_litert_platform(action="export-tflite")
    from ultralytics import YOLO

    config.output_dir.mkdir(parents=True, exist_ok=True)
    entries: list[ModelManifestEntry] = []
    for profile in config.profiles:
        LOGGER.info("exporting profile %s (imgsz=%s, %s)", profile.name, profile.imgsz, profile.precision.value)
        model: YOLO = YOLO(model=str(config.weights))
        calibration: dict[str, object] = (
            {"data": str(config.calibration_data), "fraction": config.calibration_fraction}
            if profile.precision is WeightPrecision.INT8
            else {}
        )
        exported_path: Path = Path(
            str(
                model.export(
                    format=LITERT_FORMAT,
                    imgsz=to_ultralytics_imgsz(imgsz=profile.imgsz),
                    batch=1,
                    **ultralytics_precision_kwargs(precision=profile.precision),
                    **calibration,
                )
            )
        )
        destination: Path = config.output_dir / f"{profile.name}{TFLITE_SUFFIX}"
        remove_path(path=destination)
        shutil.move(src=str(exported_path), dst=str(destination))
        inputs, outputs = _tensor_details(model_path=destination)
        class_names: list[str] = [str(model.names[index]) for index in sorted(model.names)]
        entries.append(
            _manifest_entry(
                profile=profile,
                destination=destination,
                weights=config.weights,
                class_names=class_names,
                inputs=inputs,
                outputs=outputs,
            )
        )
        LOGGER.info("saved %s", destination)
    manifest_path: Path = upsert_manifest(
        output_dir=config.output_dir,
        entries=entries,
        header=android_manifest_header(minimum_android_sdk=config.minimum_android_sdk),
    )
    LOGGER.info("manifest updated: %s", manifest_path)
    return manifest_path


def _tensor_details(*, model_path: Path) -> tuple[list[TensorInfo], list[TensorInfo]]:
    from ai_edge_litert.interpreter import Interpreter

    interpreter = Interpreter(model_path=str(model_path))
    return (
        [(str(detail["name"]), [int(side) for side in detail["shape"]]) for detail in interpreter.get_input_details()],
        [(str(detail["name"]), [int(side) for side in detail["shape"]]) for detail in interpreter.get_output_details()],
    )


def _manifest_entry(
    *,
    profile: TfliteProfile,
    destination: Path,
    weights: Path,
    class_names: list[str],
    inputs: list[TensorInfo],
    outputs: list[TensorInfo],
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
        input_names=[name for name, _ in inputs],
        output_names=[name for name, _ in outputs],
        class_names=class_names,
        includes_nms=False,
        output_layout=output_layout(
            input_shape=inputs[0][1], output_shape=outputs[0][1], class_count=len(class_names)
        ),
    )
