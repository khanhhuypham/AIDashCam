"""Small helpers shared by ``evaluate`` and ``export-coreml`` for calling Ultralytics.

Running the detector on video lives in the dashcam-video project.
"""

from __future__ import annotations

from dashcam_ml.config.schema import ImageSize
from dashcam_ml.domain.enums import WeightPrecision

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
