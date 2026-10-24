"""Framework-free value objects used by the Core ML verification.

All boxes use normalised image coordinates (0..1, origin top-left), the same
convention as Apple Vision after flipping Y.
"""

from __future__ import annotations

from dataclasses import dataclass

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
