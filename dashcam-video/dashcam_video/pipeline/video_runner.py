"""CV04 / CV13 - run detection + tracking + FCW (+ LDW) on a dashcam video.

The AI runs at ``video.ai_fps`` (like a phone, which cannot process every frame)
while the output video keeps the source frame rate and re-draws the last result.
This module only handles video in/out; the per-frame logic is in ``dashcam_pipeline``
and the events file in ``events_writer``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import numpy.typing as npt

from dashcam_video.config.schema import RootConfig
from dashcam_video.core.ego_speed import EgoSpeedProvider, build_speed_provider
from dashcam_video.core.geometry import CameraModel
from dashcam_video.core.ldw import LaneSegmentation
from dashcam_video.pipeline.dashcam_pipeline import DashcamPipeline
from dashcam_video.pipeline.events_writer import write_events_json
from dashcam_video.pipeline.frame_result import FrameResult
from dashcam_video.pipeline.renderer import OverlayRenderer
from dashcam_video.utils.logging_utils import get_logger

LOGGER = get_logger(name=__name__)
BgrImage = npt.NDArray[np.uint8]
DEFAULT_FPS: float = 30.0
QUIT_KEY: str = "q"
DEBUG_VIDEO_SUFFIX: str = "_debug.mp4"


@dataclass(frozen=True, slots=True)
class VideoRunSummary:
    output_video: Path | None
    events_json: Path | None
    processed_frames: int
    ai_frames: int
    fcw_alerts: int
    ldw_alerts: int


def run_video(*, config: RootConfig) -> VideoRunSummary:
    video_config = config.video
    capture: cv2.VideoCapture = cv2.VideoCapture(filename=str(video_config.source))
    if not capture.isOpened():
        raise FileNotFoundError(f"cannot open video {video_config.source}")
    fps: float = capture.get(propId=cv2.CAP_PROP_FPS) or DEFAULT_FPS
    width: int = int(capture.get(propId=cv2.CAP_PROP_FRAME_WIDTH))
    height: int = int(capture.get(propId=cv2.CAP_PROP_FRAME_HEIGHT))
    camera: CameraModel = CameraModel(horizontal_fov_deg=config.camera.horizontal_fov_deg, aspect_ratio=width / height)

    pipeline: DashcamPipeline = DashcamPipeline.from_config(config=config, camera=camera)
    speed_provider: EgoSpeedProvider = build_speed_provider(config=config.ego_speed)
    renderer: OverlayRenderer = OverlayRenderer(roi=config.fcw.in_path_roi, show_roi=config.fcw.enabled)

    video_config.output_dir.mkdir(parents=True, exist_ok=True)
    output_video: Path | None = (
        video_config.output_dir / f"{video_config.source.stem}{DEBUG_VIDEO_SUFFIX}" if video_config.save_video else None
    )
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
    LOGGER.info(
        "processing %s (%dx%d @ %.1f fps, AI @ %.1f fps)",
        video_config.source.name,
        width,
        height,
        fps,
        video_config.ai_fps,
    )
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

    events_json: Path | None = (
        write_events_json(config=config, records=records, frame_width=width, frame_height=height, fps=fps)
        if video_config.save_events_json
        else None
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
