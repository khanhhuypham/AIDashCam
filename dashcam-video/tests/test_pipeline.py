from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt

from dashcam_video.config.schema import RootConfig
from dashcam_video.core.geometry import CameraModel
from dashcam_video.domain.entities import BoundingBox, Detection
from dashcam_video.domain.enums import AlertLevel
from dashcam_video.pipeline.dashcam_pipeline import DashcamPipeline
from dashcam_video.pipeline.frame_result import FrameResult

CAMERA: CameraModel = CameraModel(horizontal_fov_deg=63.0, aspect_ratio=16.0 / 9.0)
FRAME_DT_S: float = 0.1
BLANK_FRAME: npt.NDArray[np.uint8] = np.zeros((9, 16, 3), dtype=np.uint8)


class ScriptedDetector:
    """Returns pre-recorded detections, one list per call, like a model would per frame."""

    def __init__(self, *, frames: Sequence[list[Detection]]) -> None:
        self._frames: list[list[Detection]] = list(frames)
        self._next: int = 0

    def detect(self, *, frame_bgr: npt.NDArray[np.uint8]) -> list[Detection]:
        detections: list[Detection] = self._frames[self._next]
        self._next += 1
        return detections


def _approaching_car(*, steps: int, closing_speed_mps: float) -> list[list[Detection]]:
    frames: list[list[Detection]] = []
    distance_m: float = 25.0
    for _ in range(steps):
        width: float = 1.8 / (distance_m * 2.0 * CAMERA.tan_half_hfov)
        box: BoundingBox = BoundingBox(x1=0.5 - width / 2.0, y1=0.75 - width * 0.8, x2=0.5 + width / 2.0, y2=0.75)
        frames.append([Detection(box=box, class_name="car", confidence=0.9)])
        distance_m = max(2.0, distance_m - closing_speed_mps * FRAME_DT_S)
    return frames


def _run(*, config: RootConfig, frames: Sequence[list[Detection]], ego_speed_kmh: float) -> list[FrameResult]:
    pipeline: DashcamPipeline = DashcamPipeline(
        config=config, camera=CAMERA, detector=ScriptedDetector(frames=frames), lane_segmenter=None
    )
    results: list[FrameResult] = []
    for index in range(len(frames)):
        result, lanes = pipeline.step(
            frame=BLANK_FRAME, frame_index=index, timestamp_s=index * FRAME_DT_S, ego_speed_kmh=ego_speed_kmh
        )
        assert lanes is None
        results.append(result)
    return results


def test_pipeline_runs_without_a_model_and_keeps_one_track_id(config: RootConfig) -> None:
    results: list[FrameResult] = _run(
        config=config, frames=_approaching_car(steps=40, closing_speed_mps=10.0), ego_speed_kmh=50.0
    )
    track_ids: set[int] = {track.track_id for result in results for track in result.tracks}
    assert track_ids == {1}
    assert any(result.fcw is not None and result.fcw.active_level is AlertLevel.CRITICAL for result in results)


def test_pipeline_output_is_deterministic(config: RootConfig) -> None:
    frames: list[list[Detection]] = _approaching_car(steps=30, closing_speed_mps=8.0)
    first: list[dict[str, object]] = [
        result.to_dict() for result in _run(config=config, frames=frames, ego_speed_kmh=40.0)
    ]
    second: list[dict[str, object]] = [
        result.to_dict() for result in _run(config=config, frames=frames, ego_speed_kmh=40.0)
    ]
    assert first == second
