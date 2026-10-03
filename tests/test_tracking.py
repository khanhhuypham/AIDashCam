from __future__ import annotations

from dashcam_ml.config.schema import TrackerConfig
from dashcam_ml.core.tracking import IouTracker, Track
from dashcam_ml.domain.entities import BoundingBox, Detection

TRACKER_CONFIG: TrackerConfig = TrackerConfig(iou_threshold=0.3, max_missed_frames=2, min_hits=2, history_seconds=1.0)


def _car(*, x1: float) -> Detection:
    return Detection(box=BoundingBox(x1=x1, y1=0.5, x2=x1 + 0.2, y2=0.7), class_name="car", confidence=0.9)


def test_track_is_confirmed_after_min_hits_and_keeps_id() -> None:
    tracker: IouTracker = IouTracker(config=TRACKER_CONFIG)
    assert tracker.update(detections=[_car(x1=0.40)], timestamp_s=0.0) == []
    confirmed: list[Track] = tracker.update(detections=[_car(x1=0.41)], timestamp_s=0.1)
    assert [track.track_id for track in confirmed] == [1]
    confirmed = tracker.update(detections=[_car(x1=0.42)], timestamp_s=0.2)
    assert [track.track_id for track in confirmed] == [1]
    assert len(confirmed[0].history) == 3


def test_track_is_dropped_after_max_missed_frames() -> None:
    tracker: IouTracker = IouTracker(config=TRACKER_CONFIG)
    tracker.update(detections=[_car(x1=0.4)], timestamp_s=0.0)
    for step in range(1, 4):
        tracker.update(detections=[], timestamp_s=0.1 * step)
    confirmed: list[Track] = tracker.update(detections=[_car(x1=0.4)], timestamp_s=0.5)
    assert confirmed == []
    confirmed = tracker.update(detections=[_car(x1=0.4)], timestamp_s=0.6)
    assert [track.track_id for track in confirmed] == [2]


def test_history_is_trimmed_to_window() -> None:
    tracker: IouTracker = IouTracker(config=TRACKER_CONFIG)
    confirmed: list[Track] = []
    for step in range(20):
        confirmed = tracker.update(detections=[_car(x1=0.4)], timestamp_s=0.1 * step)
    assert confirmed[0].history[0][0] >= 1.9 - 1.0 - 1e-9
