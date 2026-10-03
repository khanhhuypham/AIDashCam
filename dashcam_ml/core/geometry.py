"""Monocular camera geometry: distance from box size and ego-lane membership."""

from __future__ import annotations

import math
from dataclasses import dataclass

from dashcam_ml.config.schema import ClassSpec, InPathRoi
from dashcam_ml.domain.entities import BoundingBox
from dashcam_ml.domain.enums import ReferenceDimension


@dataclass(frozen=True, slots=True)
class CameraModel:
    horizontal_fov_deg: float
    aspect_ratio: float
    """Frame width divided by frame height."""

    @property
    def tan_half_hfov(self) -> float:
        return math.tan(math.radians(self.horizontal_fov_deg) / 2.0)

    @property
    def tan_half_vfov(self) -> float:
        return self.tan_half_hfov / self.aspect_ratio


def reference_box_size(*, box: BoundingBox, spec: ClassSpec) -> float:
    """Normalised box side that corresponds to ``spec.reference_size_m``."""
    return box.width if spec.reference_dimension is ReferenceDimension.WIDTH else box.height


def estimate_distance_m(*, box: BoundingBox, spec: ClassSpec, camera: CameraModel) -> float | None:
    """Pinhole model: ``size_norm = real_size / (distance * 2 * tan(fov / 2))``."""
    size_norm: float = reference_box_size(box=box, spec=spec)
    if size_norm <= 0.0:
        return None
    tan_half_fov: float = (
        camera.tan_half_hfov if spec.reference_dimension is ReferenceDimension.WIDTH else camera.tan_half_vfov
    )
    return spec.reference_size_m / (size_norm * 2.0 * tan_half_fov)


def is_in_path(*, box: BoundingBox, roi: InPathRoi) -> bool:
    """True when the bottom-centre of the box lies inside the ego-lane trapezoid."""
    point_x: float = box.center_x
    point_y: float = box.bottom_y
    if point_y < roi.top_y or point_y > roi.bottom_y:
        return False
    progress: float = (point_y - roi.top_y) / (roi.bottom_y - roi.top_y)
    half_width: float = roi.top_half_width + progress * (roi.bottom_half_width - roi.top_half_width)
    return abs(point_x - roi.center_x) <= half_width
