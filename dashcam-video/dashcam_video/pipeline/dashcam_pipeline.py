"""One AI step on one frame: detect -> track -> FCW, and optionally segment -> LDW.

This is the part the iOS and Android apps re-implement. The detector and the lane
segmenter are passed in (anything with ``detect`` / ``segment``), so tests can feed
fixed detections without loading a model.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np
import numpy.typing as npt

from dashcam_video.config.schema import RootConfig
from dashcam_video.core.class_catalog import ClassCatalog
from dashcam_video.core.fcw import ForwardCollisionWarner
from dashcam_video.core.geometry import CameraModel
from dashcam_video.core.ldw import LaneDepartureWarner, LaneSegmentation
from dashcam_video.core.tracking import IouTracker, Track
from dashcam_video.domain.entities import Detection, FcwResult, LdwResult
from dashcam_video.pipeline.frame_result import FrameResult
from dashcam_video.utils.runtime import resolve_torch_device

BgrImage = npt.NDArray[np.uint8]


class Detector(Protocol):
    def detect(self, *, frame_bgr: BgrImage) -> list[Detection]: ...


class LaneSegmenter(Protocol):
    def segment(self, *, frame_bgr: BgrImage) -> LaneSegmentation: ...


class DashcamPipeline:
    def __init__(
        self,
        *,
        config: RootConfig,
        camera: CameraModel,
        detector: Detector,
        lane_segmenter: LaneSegmenter | None,
    ) -> None:
        catalog: ClassCatalog = ClassCatalog(specs=config.classes)
        self._config: RootConfig = config
        self._detector: Detector = detector
        self._lane_segmenter: LaneSegmenter | None = lane_segmenter
        self._tracker: IouTracker = IouTracker(config=config.tracker)
        self._fcw: ForwardCollisionWarner | None = (
            ForwardCollisionWarner(config=config.fcw, catalog=catalog, camera=camera) if config.fcw.enabled else None
        )
        self._ldw: LaneDepartureWarner | None = LaneDepartureWarner(config=config.ldw) if config.ldw.enabled else None

    @classmethod
    def from_config(cls, *, config: RootConfig, camera: CameraModel) -> DashcamPipeline:
        """Pipeline with the real models: YOLO always, TwinLiteNet only when LDW is enabled.

        The model modules are imported here, not at the top, so tests that pass a fake
        detector to ``__init__`` never load torch.
        """
        from dashcam_video.perception.yolo_detector import YoloDetector

        detector: YoloDetector = YoloDetector(
            config=config.inference,
            catalog=ClassCatalog(specs=config.classes),
            device=config.project.device
        )
        lane_segmenter: LaneSegmenter | None = None
        if config.ldw.enabled:
            from dashcam_video.perception.lane_segmenter import TwinLiteNetSegmenter

            lane_segmenter = TwinLiteNetSegmenter(
                config=config.twinlitenet,
                device=resolve_torch_device(device=config.project.device)
            )
        return cls(
            config=config,
            camera=camera,
            detector=detector,
            lane_segmenter=lane_segmenter
        )

    def step(
        self, *, frame: BgrImage, frame_index: int, timestamp_s: float, ego_speed_kmh: float | None
    ) -> tuple[FrameResult, LaneSegmentation | None]:
        detections: list[Detection] = self._detector.detect(frame_bgr=frame)
        tracks: list[Track] = self._tracker.update(detections=detections, timestamp_s=timestamp_s)
        fcw_result: FcwResult | None = (
            self._fcw.update(tracks=tracks, timestamp_s=timestamp_s, ego_speed_kmh=ego_speed_kmh)
            if self._fcw is not None
            else None
        )
        lanes: LaneSegmentation | None = None
        ldw_result: LdwResult | None = None
        if self._lane_segmenter is not None and self._ldw is not None:
            lanes = self._lane_segmenter.segment(frame_bgr=frame)
            ldw_result = self._ldw.update(
                lane_position=self._ldw.estimate_lane_position(lane_mask=lanes.lane_mask),
                timestamp_s=timestamp_s,
                ego_speed_kmh=ego_speed_kmh,
            )
        result: FrameResult = FrameResult(
            frame_index=frame_index,
            timestamp_s=timestamp_s,
            ego_speed_kmh=ego_speed_kmh,
            detections=detections,
            tracks=[track.snapshot() for track in tracks],
            fcw=fcw_result,
            ldw=ldw_result,
        )
        return result, lanes
