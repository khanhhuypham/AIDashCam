"""Adapter from Ultralytics YOLO to the framework-free :class:`Detection` type."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from dashcam_ml.config.schema import ImageSize, InferenceConfig
from dashcam_ml.core.class_catalog import ClassCatalog
from dashcam_ml.domain.entities import BoundingBox, Detection
from dashcam_ml.domain.enums import WeightPrecision
from dashcam_ml.utils.runtime import resolve_device

if TYPE_CHECKING:
    from ultralytics import YOLO

BgrImage = npt.NDArray[np.uint8]
QUANTIZE_BITS: dict[WeightPrecision, int | None] = {
    WeightPrecision.FP32: None,
    WeightPrecision.FP16: 16,
    WeightPrecision.INT8: 8,
}


def ultralytics_precision_kwargs(*, precision: WeightPrecision) -> dict[str, object]:
    """Ultralytics >= 8.4 uses ``quantize``; older releases use ``half`` / ``int8`` flags."""
    from ultralytics.cfg import DEFAULT_CFG_DICT

    if "quantize" in DEFAULT_CFG_DICT:
        return {"quantize": QUANTIZE_BITS[precision]}
    return {"half": precision is WeightPrecision.FP16, "int8": precision is WeightPrecision.INT8}


def to_ultralytics_imgsz(*, imgsz: ImageSize) -> int | list[int]:
    return imgsz if isinstance(imgsz, int) else [imgsz[0], imgsz[1]]


class YoloDetector:
    def __init__(self, *, config: InferenceConfig, catalog: ClassCatalog, device: str) -> None:
        from ultralytics import YOLO

        self._config: InferenceConfig = config
        self._catalog: ClassCatalog = catalog
        self._device: str | None = resolve_device(device=device)
        self._precision_kwargs: dict[str, object] = ultralytics_precision_kwargs(precision=config.precision)
        self._model: YOLO = YOLO(model=str(config.weights))
        self._class_names: dict[int, str] = {int(index): str(name) for index, name in self._model.names.items()}

    @property
    def class_names(self) -> dict[int, str]:
        return self._class_names

    def detect(self, *, frame_bgr: BgrImage) -> list[Detection]:
        """Run the model on one frame; keep only classes listed in ``config.classes``."""
        results = self._model.predict(
            source=frame_bgr,
            imgsz=to_ultralytics_imgsz(imgsz=self._config.imgsz),
            conf=self._config.conf,
            iou=self._config.iou,
            max_det=self._config.max_det,
            device=self._device,
            verbose=False,
            **self._precision_kwargs,
        )
        boxes = results[0].boxes
        coordinates: npt.NDArray[np.float32] = boxes.xyxyn.cpu().numpy()
        class_ids: npt.NDArray[np.float32] = boxes.cls.cpu().numpy()
        confidences: npt.NDArray[np.float32] = boxes.conf.cpu().numpy()

        detections: list[Detection] = []
        for (x1, y1, x2, y2), class_id, confidence in zip(coordinates, class_ids, confidences, strict=True):
            class_name: str = self._class_names[int(class_id)]
            if not self._catalog.contains(class_name=class_name):
                continue
            detections.append(
                Detection(
                    box=BoundingBox(x1=float(x1), y1=float(y1), x2=float(x2), y2=float(y2)),
                    class_name=class_name,
                    confidence=float(confidence),
                )
            )
        return detections
