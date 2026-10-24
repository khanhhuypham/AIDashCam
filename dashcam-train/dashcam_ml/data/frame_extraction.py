"""CV08 - extract sharp, non-duplicate frames from self-recorded Vietnamese videos."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import numpy.typing as npt

from dashcam_ml.config.schema import FrameExtractionConfig
from dashcam_ml.utils.logging_utils import get_logger

LOGGER = get_logger(name=__name__)
GrayImage = npt.NDArray[np.uint8]
BgrImage = npt.NDArray[np.uint8]
DUPLICATE_CHECK_SIZE: tuple[int, int] = (160, 90)
DEFAULT_FPS: float = 30.0


@dataclass(frozen=True, slots=True)
class ExtractionStats:
    video: str
    saved: int
    skipped_blurry: int
    skipped_duplicate: int


def extract_frames(*, config: FrameExtractionConfig) -> list[ExtractionStats]:
    extensions: set[str] = {extension.lower() for extension in config.video_extensions}
    videos: list[Path] = sorted(
        path for path in config.videos_dir.rglob(pattern="*") if path.suffix.lower() in extensions
    )
    if not videos:
        raise FileNotFoundError(f"no videos with extensions {sorted(extensions)} under {config.videos_dir}")
    config.output_dir.mkdir(parents=True, exist_ok=True)
    all_stats: list[ExtractionStats] = []
    for video_path in videos:
        stats: ExtractionStats = _extract_from_video(video_path=video_path, config=config)
        LOGGER.info(
            "%s: saved %d, blurry %d, duplicate %d", stats.video, stats.saved, stats.skipped_blurry, stats.skipped_duplicate
        )
        all_stats.append(stats)
    return all_stats


def blur_score(*, gray: GrayImage) -> float:
    """Variance of the Laplacian: low value means a blurry image."""
    return float(cv2.Laplacian(src=gray, ddepth=cv2.CV_64F).var())


def _extract_from_video(*, video_path: Path, config: FrameExtractionConfig) -> ExtractionStats:
    capture: cv2.VideoCapture = cv2.VideoCapture(filename=str(video_path))
    if not capture.isOpened():
        raise OSError(f"cannot open video {video_path}")
    fps: float = capture.get(propId=cv2.CAP_PROP_FPS) or DEFAULT_FPS
    step: int = max(1, int(round(fps * config.every_n_seconds)))
    saved: int = 0
    blurry: int = 0
    duplicate: int = 0
    previous_thumbnail: GrayImage | None = None
    frame_index: int = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if frame_index % step == 0:
                gray: GrayImage = cv2.cvtColor(src=frame, code=cv2.COLOR_BGR2GRAY)
                thumbnail: GrayImage = cv2.resize(src=gray, dsize=DUPLICATE_CHECK_SIZE, interpolation=cv2.INTER_AREA)
                if blur_score(gray=gray) < config.blur_threshold:
                    blurry += 1
                elif previous_thumbnail is not None and _mean_abs_diff(
                    first=thumbnail, second=previous_thumbnail
                ) < config.duplicate_threshold:
                    duplicate += 1
                else:
                    output_path: Path = (
                        config.output_dir / f"{video_path.stem}{config.name_separator}f{frame_index:06d}.jpg"
                    )
                    cv2.imwrite(
                        filename=str(output_path), img=frame, params=[cv2.IMWRITE_JPEG_QUALITY, config.jpeg_quality]
                    )
                    previous_thumbnail = thumbnail
                    saved += 1
            frame_index += 1
    finally:
        capture.release()
    return ExtractionStats(video=video_path.name, saved=saved, skipped_blurry=blurry, skipped_duplicate=duplicate)


def _mean_abs_diff(*, first: GrayImage, second: GrayImage) -> float:
    return float(np.mean(a=cv2.absdiff(src1=first, src2=second)))
