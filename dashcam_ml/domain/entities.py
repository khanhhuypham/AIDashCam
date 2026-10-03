"""Framework-free value objects shared by every layer.

All boxes use normalised image coordinates (0..1, origin top-left), the same
convention as Apple Vision after flipping Y, so the logic ports to Swift 1:1.
"""

from __future__ import annotations

from dataclasses import dataclass

from dashcam_ml.domain.enums import AlertLevel, LaneDepartureSide

JsonDict = dict[str, object]


@dataclass(frozen=True, slots=True)
class BoundingBox:
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def width(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        return max(0.0, self.y2 - self.y1)

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def center_x(self) -> float:
        return (self.x1 + self.x2) / 2.0

    @property
    def bottom_y(self) -> float:
        return self.y2

    def iou(self, *, other: BoundingBox) -> float:
        inter_w: float = max(0.0, min(self.x2, other.x2) - max(self.x1, other.x1))
        inter_h: float = max(0.0, min(self.y2, other.y2) - max(self.y1, other.y1))
        intersection: float = inter_w * inter_h
        union: float = self.area + other.area - intersection
        return intersection / union if union > 0.0 else 0.0

    def to_pixels(self, *, frame_width: int, frame_height: int) -> tuple[int, int, int, int]:
        return (
            int(round(self.x1 * frame_width)),
            int(round(self.y1 * frame_height)),
            int(round(self.x2 * frame_width)),
            int(round(self.y2 * frame_height)),
        )

    def to_dict(self) -> JsonDict:
        return {"x1": round(self.x1, 6), "y1": round(self.y1, 6), "x2": round(self.x2, 6), "y2": round(self.y2, 6)}


@dataclass(frozen=True, slots=True)
class Detection:
    box: BoundingBox
    class_name: str
    confidence: float

    def to_dict(self) -> JsonDict:
        return {"box": self.box.to_dict(), "class_name": self.class_name, "confidence": round(self.confidence, 4)}


@dataclass(frozen=True, slots=True)
class FcwResult:
    lead_track_id: int | None
    distance_m: float | None
    ttc_s: float | None
    headway_s: float | None
    raw_level: AlertLevel
    active_level: AlertLevel
    alert_triggered: bool

    def to_dict(self) -> JsonDict:
        return {
            "lead_track_id": self.lead_track_id,
            "distance_m": _round_optional(value=self.distance_m, digits=3),
            "ttc_s": _round_optional(value=self.ttc_s, digits=3),
            "headway_s": _round_optional(value=self.headway_s, digits=3),
            "raw_level": self.raw_level.name,
            "active_level": self.active_level.name,
            "alert_triggered": self.alert_triggered,
        }


@dataclass(frozen=True, slots=True)
class LanePosition:
    lane_center_x: float
    lane_width: float
    offset_ratio: float
    valid_rows_ratio: float

    def to_dict(self) -> JsonDict:
        return {
            "lane_center_x": round(self.lane_center_x, 5),
            "lane_width": round(self.lane_width, 5),
            "offset_ratio": round(self.offset_ratio, 5),
            "valid_rows_ratio": round(self.valid_rows_ratio, 5),
        }


@dataclass(frozen=True, slots=True)
class LdwResult:
    lane_position: LanePosition | None
    raw_side: LaneDepartureSide
    active_side: LaneDepartureSide
    alert_triggered: bool

    def to_dict(self) -> JsonDict:
        return {
            "lane_position": self.lane_position.to_dict() if self.lane_position is not None else None,
            "raw_side": self.raw_side.value,
            "active_side": self.active_side.value,
            "alert_triggered": self.alert_triggered,
        }


def _round_optional(*, value: float | None, digits: int) -> float | None:
    return None if value is None else round(value, digits)
