"""Greedy IoU multi-object tracker.

Deliberately simple and deterministic (ties broken by index) so the Swift port in
the iOS app produces exactly the same track IDs on the same input.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass, field

from dashcam_ml.config.schema import TrackerConfig
from dashcam_ml.domain.entities import BoundingBox, Detection, JsonDict


@dataclass(slots=True)
class Track:
    track_id: int
    class_name: str
    box: BoundingBox
    confidence: float
    hits: int
    missed_frames: int
    first_seen_s: float
    last_seen_s: float
    history: deque[tuple[float, BoundingBox]] = field(default_factory=deque)

    def to_dict(self) -> JsonDict:
        return {
            "track_id": self.track_id,
            "class_name": self.class_name,
            "box": self.box.to_dict(),
            "confidence": round(self.confidence, 4),
            "hits": self.hits,
        }


class IouTracker:
    def __init__(self, *, config: TrackerConfig) -> None:
        self._config: TrackerConfig = config
        self._tracks: list[Track] = []
        self._next_id: int = 1

    def reset(self) -> None:
        self._tracks = []
        self._next_id = 1

    def update(self, *, detections: Sequence[Detection], timestamp_s: float) -> list[Track]:
        """Associate detections with tracks; return confirmed tracks seen in this frame."""
        candidate_pairs: list[tuple[float, int, int]] = []
        for track_index, track in enumerate(self._tracks):
            for detection_index, detection in enumerate(detections):
                overlap: float = track.box.iou(other=detection.box)
                if overlap >= self._config.iou_threshold:
                    candidate_pairs.append((overlap, track_index, detection_index))
        candidate_pairs.sort(key=lambda pair: (-pair[0], pair[1], pair[2]))

        matched_tracks: set[int] = set()
        matched_detections: set[int] = set()
        for _, track_index, detection_index in candidate_pairs:
            if track_index in matched_tracks or detection_index in matched_detections:
                continue
            self._apply_match(
                track=self._tracks[track_index], detection=detections[detection_index], timestamp_s=timestamp_s
            )
            matched_tracks.add(track_index)
            matched_detections.add(detection_index)

        for track_index, track in enumerate(self._tracks):
            if track_index not in matched_tracks:
                track.missed_frames += 1
        self._tracks = [track for track in self._tracks if track.missed_frames <= self._config.max_missed_frames]

        for detection_index, detection in enumerate(detections):
            if detection_index not in matched_detections:
                self._tracks.append(self._create_track(detection=detection, timestamp_s=timestamp_s))

        return [
            track for track in self._tracks if track.missed_frames == 0 and track.hits >= self._config.min_hits
        ]

    def _apply_match(self, *, track: Track, detection: Detection, timestamp_s: float) -> None:
        track.box = detection.box
        track.class_name = detection.class_name
        track.confidence = detection.confidence
        track.hits += 1
        track.missed_frames = 0
        track.last_seen_s = timestamp_s
        self._append_history(track=track, timestamp_s=timestamp_s)

    def _create_track(self, *, detection: Detection, timestamp_s: float) -> Track:
        track: Track = Track(
            track_id=self._next_id,
            class_name=detection.class_name,
            box=detection.box,
            confidence=detection.confidence,
            hits=1,
            missed_frames=0,
            first_seen_s=timestamp_s,
            last_seen_s=timestamp_s,
        )
        self._next_id += 1
        self._append_history(track=track, timestamp_s=timestamp_s)
        return track

    def _append_history(self, *, track: Track, timestamp_s: float) -> None:
        track.history.append((timestamp_s, track.box))
        oldest_allowed_s: float = timestamp_s - self._config.history_seconds
        while track.history and track.history[0][0] < oldest_allowed_s:
            track.history.popleft()
