"""Write ``<video>_events.json``: the logic's input and output on every AI frame.

It is the parity test data for the Swift (iOS) and Kotlin (Android) ports, so its
structure must stay stable.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from dashcam_video.config.schema import RootConfig
from dashcam_video.domain.enums import Platform
from dashcam_video.export.app_config import build_app_config
from dashcam_video.pipeline.frame_result import FrameResult
from dashcam_video.utils.files import write_json

EVENTS_SUFFIX: str = "_events.json"


def write_events_json(
    *, config: RootConfig, records: Sequence[FrameResult], frame_width: int, frame_height: int, fps: float
) -> Path:
    video_config = config.video
    events_json: Path = video_config.output_dir / f"{video_config.source.stem}{EVENTS_SUFFIX}"
    write_json(
        path=events_json,
        payload={
            "meta": {
                "source": video_config.source.name,
                "frame_width": frame_width,
                "frame_height": frame_height,
                "fps": fps,
                "ai_fps": video_config.ai_fps,
                "weights": config.inference.weights.name,
                # key kept as "ios_config" for existing Swift parity tests; thresholds are the same on Android
                "ios_config": build_app_config(
                    config=config, target=config.app_config_export.target(platform=Platform.IOS)
                ),
            },
            "frames": [record.to_dict() for record in records],
        },
    )
    return events_json
