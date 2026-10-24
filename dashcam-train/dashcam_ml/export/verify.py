"""CV14 - check that exported Core ML models match PyTorch on the same images.

Must run on macOS (Core ML can only execute on Apple platforms). Both backends get
the *same* letterboxed image, so any difference comes from conversion/precision.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np
import numpy.typing as npt

from dashcam_ml.config.schema import CoremlVerifyConfig, TwinLiteNetConfig
from dashcam_ml.domain.entities import BoundingBox, Detection
from dashcam_ml.domain.enums import ModelKind
from dashcam_ml.export.comparison import (
    DetectionComparison,
    MaskComparison,
    compare_detections,
    compare_probability_maps,
)
from dashcam_ml.export.manifest import ModelManifestEntry, read_manifest
from dashcam_ml.export.twinlitenet_coreml import DRIVABLE_OUTPUT_NAME, INPUT_NAME, LANE_OUTPUT_NAME
from dashcam_ml.utils.files import iter_images, write_json
from dashcam_ml.utils.logging_utils import get_logger
from dashcam_ml.utils.runtime import require_macos

if TYPE_CHECKING:
    from PIL.Image import Image as PilImage

LOGGER = get_logger(name=__name__)
BgrImage = npt.NDArray[np.uint8]
LETTERBOX_PAD_VALUE: int = 114


@dataclass(frozen=True, slots=True)
class DetectorVerification:
    model_name: str
    images: int
    reference_boxes: int
    coreml_boxes: int
    matched_boxes: int
    match_rate: float
    mean_iou: float
    mean_confidence_abs_diff: float


@dataclass(frozen=True, slots=True)
class SegmenterVerification:
    model_name: str
    images: int
    drivable_mean_abs_diff: float
    drivable_mask_iou: float
    lane_mean_abs_diff: float
    lane_mask_iou: float


def letterbox(*, image: BgrImage, height: int, width: int) -> BgrImage:
    """Resize keeping aspect ratio, pad to ``height x width`` (same as Ultralytics)."""
    source_height, source_width = image.shape[:2]
    ratio: float = min(height / source_height, width / source_width)
    new_width: int = int(round(source_width * ratio))
    new_height: int = int(round(source_height * ratio))
    resized: BgrImage = cv2.resize(src=image, dsize=(new_width, new_height), interpolation=cv2.INTER_LINEAR)
    pad_left: int = (width - new_width) // 2
    pad_top: int = (height - new_height) // 2
    return cv2.copyMakeBorder(
        src=resized,
        top=pad_top,
        bottom=height - new_height - pad_top,
        left=pad_left,
        right=width - new_width - pad_left,
        borderType=cv2.BORDER_CONSTANT,
        value=(LETTERBOX_PAD_VALUE, LETTERBOX_PAD_VALUE, LETTERBOX_PAD_VALUE),
    )


def verify_coreml_models(
    *, verify_config: CoremlVerifyConfig, models_dir: Path, twinlitenet_config: TwinLiteNetConfig
) -> Path:
    require_macos(action="verify-coreml")
    entries: list[ModelManifestEntry] = read_manifest(output_dir=models_dir)
    if not entries:
        raise FileNotFoundError(f"no manifest.json in {models_dir}; run export-coreml first")
    images: list[Path] = list(iter_images(directory=verify_config.images_dir))[: verify_config.max_images]
    if not images:
        raise FileNotFoundError(f"no images under {verify_config.images_dir}")

    detector_reports: list[DetectorVerification] = []
    segmenter_reports: list[SegmenterVerification] = []
    for entry in entries:
        model_path: Path = models_dir / entry.file_name
        match entry.kind:
            case ModelKind.DETECTOR:
                detector_reports.append(
                    _verify_detector(entry=entry, model_path=model_path, images=images, config=verify_config)
                )
            case ModelKind.LANE_SEGMENTER:
                segmenter_reports.append(
                    _verify_segmenter(entry=entry, model_path=model_path, images=images, config=twinlitenet_config)
                )

    for report in detector_reports:
        LOGGER.info(
            "%s: match %.1f%%, IoU %.3f, |Δconf| %.4f",
            report.model_name,
            report.match_rate * 100.0,
            report.mean_iou,
            report.mean_confidence_abs_diff,
        )
    for report in segmenter_reports:
        LOGGER.info(
            "%s: drivable IoU %.3f, lane IoU %.3f", report.model_name, report.drivable_mask_iou, report.lane_mask_iou
        )
    write_json(
        path=verify_config.output_json,
        payload={
            "detectors": [asdict(report) for report in detector_reports],
            "segmenters": [asdict(report) for report in segmenter_reports],
        },
    )
    return verify_config.output_json


def _verify_detector(
    *, entry: ModelManifestEntry, model_path: Path, images: list[Path], config: CoremlVerifyConfig
) -> DetectorVerification:
    import coremltools as ct
    from ultralytics import YOLO

    reference_model: YOLO = YOLO(model=str(config.pytorch_weights))
    coreml_pipeline = ct.models.MLModel(model=str(model_path)) if entry.includes_nms else None
    coreml_yolo: YOLO | None = None if entry.includes_nms else YOLO(model=str(model_path), task="detect")
    comparisons: list[DetectionComparison] = []
    for image_path in images:
        frame: BgrImage = cv2.imread(filename=str(image_path))
        boxed: BgrImage = letterbox(image=frame, height=entry.input_height, width=entry.input_width)
        reference: list[Detection] = _predict_ultralytics(
            model=reference_model, image=boxed, entry=entry, config=config
        )
        candidate: list[Detection] = (
            _predict_coreml_pipeline(model=coreml_pipeline, image=boxed, entry=entry, config=config)
            if coreml_pipeline is not None
            else _predict_ultralytics(model=coreml_yolo, image=boxed, entry=entry, config=config)
        )
        comparisons.append(
            compare_detections(
                reference=reference, candidate=candidate, match_iou_threshold=config.match_iou_threshold
            )
        )
    reference_total: int = sum(item.reference_count for item in comparisons)
    matched_total: int = sum(item.matched_count for item in comparisons)
    matched_items: list[DetectionComparison] = [item for item in comparisons if item.matched_count > 0]
    return DetectorVerification(
        model_name=entry.name,
        images=len(images),
        reference_boxes=reference_total,
        coreml_boxes=sum(item.candidate_count for item in comparisons),
        matched_boxes=matched_total,
        match_rate=matched_total / reference_total if reference_total > 0 else 1.0,
        mean_iou=float(np.mean(a=[item.mean_iou for item in matched_items])) if matched_items else 0.0,
        mean_confidence_abs_diff=(
            float(np.mean(a=[item.mean_confidence_abs_diff for item in matched_items])) if matched_items else 0.0
        ),
    )


def _predict_ultralytics(
    *, model: object, image: BgrImage, entry: ModelManifestEntry, config: CoremlVerifyConfig
) -> list[Detection]:
    results = model.predict(  # type: ignore[attr-defined]
        source=image,
        imgsz=[entry.input_height, entry.input_width],
        conf=config.conf,
        iou=config.iou,
        device="cpu",
        verbose=False,
    )
    result = results[0]
    names: dict[int, str] = result.names
    return [
        Detection(
            box=BoundingBox(x1=float(x1), y1=float(y1), x2=float(x2), y2=float(y2)),
            class_name=names[int(class_id)],
            confidence=float(confidence),
        )
        for (x1, y1, x2, y2), class_id, confidence in zip(
            result.boxes.xyxyn.cpu().numpy(), result.boxes.cls.cpu().numpy(), result.boxes.conf.cpu().numpy(), strict=True
        )
    ]


def _predict_coreml_pipeline(
    *, model: object, image: BgrImage, entry: ModelManifestEntry, config: CoremlVerifyConfig
) -> list[Detection]:
    """Core ML NMS pipeline: ``coordinates`` = normalised (cx, cy, w, h); ``confidence`` = (N, classes)."""
    spec_inputs: set[str] = {item.name for item in model.get_spec().description.input}  # type: ignore[attr-defined]
    inputs: dict[str, object] = {"image": _to_pil(image=image)}
    if "iouThreshold" in spec_inputs:
        inputs["iouThreshold"] = config.iou
    if "confidenceThreshold" in spec_inputs:
        inputs["confidenceThreshold"] = config.conf
    outputs: dict[str, npt.NDArray[np.float32]] = model.predict(data=inputs)  # type: ignore[attr-defined]
    coordinates: npt.NDArray[np.float32] = np.asarray(outputs["coordinates"]).reshape(-1, 4)
    confidences: npt.NDArray[np.float32] = np.asarray(outputs["confidence"]).reshape(coordinates.shape[0], -1)
    detections: list[Detection] = []
    for (center_x, center_y, width, height), scores in zip(coordinates, confidences, strict=True):
        class_id: int = int(np.argmax(a=scores))
        confidence: float = float(scores[class_id])
        if confidence < config.conf:
            continue
        detections.append(
            Detection(
                box=BoundingBox(
                    x1=float(center_x - width / 2.0),
                    y1=float(center_y - height / 2.0),
                    x2=float(center_x + width / 2.0),
                    y2=float(center_y + height / 2.0),
                ),
                class_name=entry.class_names[class_id],
                confidence=confidence,
            )
        )
    return detections


def _verify_segmenter(
    *, entry: ModelManifestEntry, model_path: Path, images: list[Path], config: TwinLiteNetConfig
) -> SegmenterVerification:
    import coremltools as ct
    import torch

    from dashcam_ml.models.twinlitenet import TwinLiteNetProbabilityHead, load_twinlitenet, preprocess_frame

    reference_model: TwinLiteNetProbabilityHead = TwinLiteNetProbabilityHead(
        backbone=load_twinlitenet(config=config)
    ).eval()
    coreml_model = ct.models.MLModel(model=str(model_path))
    drivable: list[MaskComparison] = []
    lane: list[MaskComparison] = []
    for image_path in images:
        frame: BgrImage = cv2.imread(filename=str(image_path))
        with torch.inference_mode():
            reference_drivable, reference_lane = reference_model(preprocess_frame(frame_bgr=frame, config=config))
        resized: BgrImage = cv2.resize(
            src=frame, dsize=(config.input_width, config.input_height), interpolation=cv2.INTER_LINEAR
        )
        outputs: dict[str, npt.NDArray[np.float32]] = coreml_model.predict(data={INPUT_NAME: _to_pil(image=resized)})
        drivable.append(
            compare_probability_maps(
                reference=reference_drivable[0].numpy(),
                candidate=np.asarray(outputs[DRIVABLE_OUTPUT_NAME]).reshape(config.input_height, config.input_width),
                threshold=config.mask_threshold,
            )
        )
        lane.append(
            compare_probability_maps(
                reference=reference_lane[0].numpy(),
                candidate=np.asarray(outputs[LANE_OUTPUT_NAME]).reshape(config.input_height, config.input_width),
                threshold=config.mask_threshold,
            )
        )
    return SegmenterVerification(
        model_name=entry.name,
        images=len(images),
        drivable_mean_abs_diff=float(np.mean(a=[item.mean_abs_prob_diff for item in drivable])),
        drivable_mask_iou=float(np.mean(a=[item.mask_iou for item in drivable])),
        lane_mean_abs_diff=float(np.mean(a=[item.mean_abs_prob_diff for item in lane])),
        lane_mask_iou=float(np.mean(a=[item.mask_iou for item in lane])),
    )


def _to_pil(*, image: BgrImage) -> PilImage:
    from PIL import Image

    return Image.fromarray(obj=cv2.cvtColor(src=image, code=cv2.COLOR_BGR2RGB))
