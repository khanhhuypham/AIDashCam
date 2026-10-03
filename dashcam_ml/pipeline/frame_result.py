"""Per-AI-frame record: inputs + outputs of the logic, serialised for Swift parity tests."""

from __future__ import annotations

from dataclasses import dataclass

from dashcam_ml.core.tracking import Track
from dashcam_ml.domain.entities import Detection, FcwResult, JsonDict, LdwResult


@dataclass(frozen=True, slots=True)
class FrameResult:
    frame_index: int
    timestamp_s: float
    ego_speed_kmh: float | None
    detections: list[Detection]
    tracks: list[Track]
    fcw: FcwResult | None
    ldw: LdwResult | None

    def to_dict(self) -> JsonDict:
        return {
            "frame_index": self.frame_index,
            "timestamp_s": round(self.timestamp_s, 4),
            "ego_speed_kmh": self.ego_speed_kmh,
            "detections": [detection.to_dict() for detection in self.detections],
            "tracks": [track.to_dict() for track in self.tracks],
            "fcw": self.fcw.to_dict() if self.fcw is not None else None,
            "ldw": self.ldw.to_dict() if self.ldw is not None else None,
        }
