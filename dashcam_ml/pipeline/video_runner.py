"""CV04 / CV13 - run detection + tracking + FCW (+ LDW) on a dashcam video.

The AI runs at ``video.ai_fps`` (like the iPhone, which cannot process every frame)
while the output video keeps the source frame rate and re-draws the last result.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np
import numpy.typing as npt

from dashcam_ml.config.schema import DashcamConfig
from dashcam_ml.core.class_catalog import ClassCatalog
from dashcam_ml.core.ego_speed import EgoSpeedProvider, build_speed_provider
from dashcam_ml.core.fcw import ForwardCollisionWarner
from dashcam_ml.core.geometry import CameraModel
from dashcam_ml.core.ldw import LaneDepartureWarner, LaneSegmentation, estimate_lane_position
from dashcam_ml.core.tracking import IouTracker, Track
from dashcam_ml.domain.entities import Detection, FcwResult, LdwResult
from dashcam_ml.export.ios_config import build_ios_config
from dashcam_ml.models.yolo_detector import YoloDetector
from dashcam_ml.pipeline.frame_result import FrameResult
from dashcam_ml.pipeline.renderer import OverlayRenderer
from dashcam_ml.utils.files import write_json
from dashcam_ml.utils.logging_utils import get_logger
from dashcam_ml.utils.runtime import resolve_torch_device

if TYPE_CHECKING:
    from dashcam_ml.models.twinlitenet import TwinLiteNetSegmenter

LOGGER = get_logger(name=__name__)
BgrImage = npt.NDArray[np.uint8]
DEFAULT_FPS: float = 30.0
QUIT_KEY: str = "q"


@dataclass(frozen=True, slots=True)
class VideoRunSummary:
    output_video: Path | None
    events_json: Path | None
    processed_frames: int
    ai_frames: int
    fcw_alerts: int
    ldw_alerts: int


class DashcamPipeline:
    """One AI step = detect -> track -> FCW, and optionally segment -> LDW."""

    def __init__(self, *, config: DashcamConfig, camera: CameraModel) -> None:
        catalog: ClassCatalog = ClassCatalog(specs=config.classes)
        self._config: DashcamConfig = config
        self._detector: YoloDetector = YoloDetector(config=config.inference, catalog=catalog, device=config.project.device)
        self._tracker: IouTracker = IouTracker(config=config.tracker)
        self._fcw: ForwardCollisionWarner | None = (
            ForwardCollisionWarner(config=config.fcw, catalog=catalog, camera=camera) if config.fcw.enabled else None
        )
        self._ldw: LaneDepartureWarner | None = LaneDepartureWarner(config=config.ldw) if config.ldw.enabled else None
        self._segmenter: TwinLiteNetSegmenter | None = None
        if config.ldw.enabled:
            from dashcam_ml.models.twinlitenet import TwinLiteNetSegmenter

            self._segmenter = TwinLiteNetSegmenter(
                config=config.twinlitenet, device=resolve_torch_device(device=config.project.device)
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
        if self._segmenter is not None and self._ldw is not None:
            lanes = self._segmenter.segment(frame_bgr=frame)
            ldw_result = self._ldw.update(
                lane_position=estimate_lane_position(lane_mask=lanes.lane_mask, config=self._config.ldw),
                timestamp_s=timestamp_s,
                ego_speed_kmh=ego_speed_kmh,
            )
        result: FrameResult = FrameResult(
            frame_index=frame_index,
            timestamp_s=timestamp_s,
            ego_speed_kmh=ego_speed_kmh,
            detections=detections,
            tracks=[_snapshot_track(track=track) for track in tracks],
            fcw=fcw_result,
            ldw=ldw_result,
        )
        return result, lanes


def run_video(*, config: DashcamConfig) -> VideoRunSummary:
    video_config = config.video
    capture: cv2.VideoCapture = cv2.VideoCapture(filename=str(video_config.source))
    if not capture.isOpened():
        raise FileNotFoundError(f"cannot open video {video_config.source}")
    fps: float = capture.get(propId=cv2.CAP_PROP_FPS) or DEFAULT_FPS
    width: int = int(capture.get(propId=cv2.CAP_PROP_FRAME_WIDTH))
    height: int = int(capture.get(propId=cv2.CAP_PROP_FRAME_HEIGHT))
    camera: CameraModel = CameraModel(horizontal_fov_deg=config.camera.horizontal_fov_deg, aspect_ratio=width / height)

    pipeline: DashcamPipeline = DashcamPipeline(config=config, camera=camera)
    speed_provider: EgoSpeedProvider = build_speed_provider(config=config.ego_speed)
    renderer: OverlayRenderer = OverlayRenderer(roi=config.fcw.in_path_roi, show_roi=config.fcw.enabled)

    video_config.output_dir.mkdir(parents=True, exist_ok=True)
    output_video: Path | None = video_config.output_dir / f"{video_config.source.stem}_debug.mp4" if video_config.save_video else None
    writer: cv2.VideoWriter | None = (
        cv2.VideoWriter(
            filename=str(output_video),
            fourcc=cv2.VideoWriter_fourcc(*video_config.output_codec),
            fps=fps,
            frameSize=(width, height),
        )
        if output_video is not None
        else None
    )

    ai_interval_s: float = 1.0 / video_config.ai_fps
    next_ai_s: float = 0.0
    frame_index: int = 0
    records: list[FrameResult] = []
    last_result: FrameResult | None = None
    last_lanes: LaneSegmentation | None = None
    LOGGER.info("processing %s (%dx%d @ %.1f fps, AI @ %.1f fps)", video_config.source.name, width, height, fps, video_config.ai_fps)
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            timestamp_s: float = frame_index / fps
            if video_config.max_seconds is not None and timestamp_s > video_config.max_seconds:
                break
            if timestamp_s + 1e-9 >= next_ai_s:
                last_result, last_lanes = pipeline.step(
                    frame=frame,
                    frame_index=frame_index,
                    timestamp_s=timestamp_s,
                    ego_speed_kmh=speed_provider.speed_kmh(timestamp_s=timestamp_s),
                )
                records.append(last_result)
                _log_alerts(result=last_result)
                next_ai_s += ai_interval_s
            if writer is not None or video_config.display:
                canvas: BgrImage = renderer.draw(frame=frame, result=last_result, lanes=last_lanes)
                if writer is not None:
                    writer.write(image=canvas)
                if video_config.display:
                    cv2.imshow(winname="AIDashCam", mat=canvas)
                    if cv2.waitKey(delay=1) & 0xFF == ord(QUIT_KEY):
                        break
            frame_index += 1
    finally:
        capture.release()
        if writer is not None:
            writer.release()
        if video_config.display:
            cv2.destroyAllWindows()

    events_json: Path | None = None
    if video_config.save_events_json:
        events_json = video_config.output_dir / f"{video_config.source.stem}_events.json"
        write_json(
            path=events_json,
            payload={
                "meta": {
                    "source": video_config.source.name,
                    "frame_width": width,
                    "frame_height": height,
                    "fps": fps,
                    "ai_fps": video_config.ai_fps,
                    "weights": config.inference.weights.name,
                    "ios_config": build_ios_config(config=config),
                },
                "frames": [record.to_dict() for record in records],
            },
        )
    summary: VideoRunSummary = VideoRunSummary(
        output_video=output_video,
        events_json=events_json,
        processed_frames=frame_index,
        ai_frames=len(records),
        fcw_alerts=sum(1 for record in records if record.fcw is not None and record.fcw.alert_triggered),
        ldw_alerts=sum(1 for record in records if record.ldw is not None and record.ldw.alert_triggered),
    )
    LOGGER.info(
        "done: %d frames, %d AI frames, %d FCW alerts, %d LDW alerts",
        summary.processed_frames,
        summary.ai_frames,
        summary.fcw_alerts,
        summary.ldw_alerts,
    )
    return summary


def _snapshot_track(*, track: Track) -> Track:
    """Copy without history so later tracker updates do not mutate stored results."""
    return Track(
        track_id=track.track_id,
        class_name=track.class_name,
        box=track.box,
        confidence=track.confidence,
        hits=track.hits,
        missed_frames=track.missed_frames,
        first_seen_s=track.first_seen_s,
        last_seen_s=track.last_seen_s,
    )


def _log_alerts(*, result: FrameResult) -> None:
    if result.fcw is not None and result.fcw.alert_triggered:
        LOGGER.warning(
            "t=%.2fs FCW %s (track #%s, TTC=%s)",
            result.timestamp_s,
            result.fcw.active_level.name,
            result.fcw.lead_track_id,
            None if result.fcw.ttc_s is None else round(result.fcw.ttc_s, 2),
        )
    if result.ldw is not None and result.ldw.alert_triggered:
        LOGGER.warning("t=%.2fs LDW %s", result.timestamp_s, result.ldw.active_side.value)
