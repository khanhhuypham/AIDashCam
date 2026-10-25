"""Monocular camera geometry: distance from box size (pinhole model).

Ego-lane membership is ``InPathRoi.contains`` in ``config/schema.py``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from dashcam_video.config.schema import ClassSpec
from dashcam_video.domain.entities import BoundingBox
from dashcam_video.domain.enums import ReferenceDimension


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

    def estimate_distance_m(self, *, box: BoundingBox, spec: ClassSpec) -> float | None:
        """Pinhole model: ``size_norm = real_size / (distance * 2 * tan(fov / 2))``."""
        size_norm: float = box.size_along(dimension=spec.reference_dimension)
        if size_norm <= 0.0:
            return None
        tan_half_fov: float = (
            self.tan_half_hfov if spec.reference_dimension is ReferenceDimension.WIDTH else self.tan_half_vfov
        )
        return spec.reference_size_m / (size_norm * 2.0 * tan_half_fov)
