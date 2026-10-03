"""Debug overlay drawn on each video frame (boxes, ROI, lane masks, HUD, alerts)."""

from __future__ import annotations

from enum import Enum

import cv2
import numpy as np
import numpy.typing as npt

from dashcam_ml.config.schema import InPathRoi
from dashcam_ml.domain.enums import AlertLevel, LaneDepartureSide
from dashcam_ml.core.ldw import LaneSegmentation
from dashcam_ml.pipeline.frame_result import FrameResult

BgrImage = npt.NDArray[np.uint8]
BgrColor = tuple[int, int, int]
FONT: int = cv2.FONT_HERSHEY_SIMPLEX
MASK_ALPHA: float = 0.4


class Palette(Enum):
    TRACK = (255, 200, 0)
    LEAD = (0, 215, 255)
    ROI = (200, 200, 200)
    DRIVABLE = (0, 160, 0)
    LANE = (0, 0, 255)
    TEXT = (255, 255, 255)
    CAUTION = (0, 165, 255)
    CRITICAL = (0, 0, 230)
    HUD_BACKGROUND = (30, 30, 30)

    @property
    def bgr(self) -> BgrColor:
        return self.value


ALERT_COLORS: dict[AlertLevel, Palette] = {AlertLevel.CAUTION: Palette.CAUTION, AlertLevel.CRITICAL: Palette.CRITICAL}


class OverlayRenderer:
    def __init__(self, *, roi: InPathRoi, show_roi: bool) -> None:
        self._roi: InPathRoi = roi
        self._show_roi: bool = show_roi

    def draw(self, *, frame: BgrImage, result: FrameResult | None, lanes: LaneSegmentation | None) -> BgrImage:
        canvas: BgrImage = frame.copy()
        if lanes is not None:
            self._draw_masks(canvas=canvas, lanes=lanes)
        if self._show_roi:
            self._draw_roi(canvas=canvas)
        if result is not None:
            self._draw_tracks(canvas=canvas, result=result)
            self._draw_hud(canvas=canvas, result=result)
        return canvas

    def _draw_masks(self, *, canvas: BgrImage, lanes: LaneSegmentation) -> None:
        height, width = canvas.shape[:2]
        for mask, color in ((lanes.drivable_mask, Palette.DRIVABLE), (lanes.lane_mask, Palette.LANE)):
            resized: npt.NDArray[np.bool_] = cv2.resize(
                src=mask.astype(np.uint8), dsize=(width, height), interpolation=cv2.INTER_NEAREST
            ).astype(bool)
            color_layer: BgrImage = np.zeros_like(canvas)
            color_layer[:] = color.bgr
            blended: BgrImage = cv2.addWeighted(
                src1=canvas, alpha=1.0 - MASK_ALPHA, src2=color_layer, beta=MASK_ALPHA, gamma=0.0
            )
            canvas[resized] = blended[resized]

    def _draw_roi(self, *, canvas: BgrImage) -> None:
        height, width = canvas.shape[:2]
        roi: InPathRoi = self._roi
        points: npt.NDArray[np.int32] = np.array(
            [
                [(roi.center_x - roi.top_half_width) * width, roi.top_y * height],
                [(roi.center_x + roi.top_half_width) * width, roi.top_y * height],
                [(roi.center_x + roi.bottom_half_width) * width, roi.bottom_y * height],
                [(roi.center_x - roi.bottom_half_width) * width, roi.bottom_y * height],
            ],
            dtype=np.int32,
        )
        cv2.polylines(img=canvas, pts=[points], isClosed=True, color=Palette.ROI.bgr, thickness=1)

    def _draw_tracks(self, *, canvas: BgrImage, result: FrameResult) -> None:
        height, width = canvas.shape[:2]
        lead_id: int | None = result.fcw.lead_track_id if result.fcw is not None else None
        for track in result.tracks:
            x1, y1, x2, y2 = track.box.to_pixels(frame_width=width, frame_height=height)
            color: Palette = Palette.LEAD if track.track_id == lead_id else Palette.TRACK
            cv2.rectangle(img=canvas, pt1=(x1, y1), pt2=(x2, y2), color=color.bgr, thickness=2)
            cv2.putText(
                img=canvas,
                text=f"#{track.track_id} {track.class_name} {track.confidence:.2f}",
                org=(x1, max(12, y1 - 5)),
                fontFace=FONT,
                fontScale=0.45,
                color=color.bgr,
                thickness=1,
                lineType=cv2.LINE_AA,
            )

    def _draw_hud(self, *, canvas: BgrImage, result: FrameResult) -> None:
        lines: list[str] = [f"t={result.timestamp_s:6.2f}s  speed={_format_optional(value=result.ego_speed_kmh, unit='km/h')}"]
        banner: tuple[str, Palette] | None = None
        if result.fcw is not None:
            fcw = result.fcw
            lines.append(
                f"FCW lead={fcw.lead_track_id}  d={_format_optional(value=fcw.distance_m, unit='m')}  "
                f"TTC={_format_optional(value=fcw.ttc_s, unit='s')}  headway={_format_optional(value=fcw.headway_s, unit='s')}"
            )
            if fcw.active_level is not AlertLevel.NONE:
                banner = (f"FCW {fcw.active_level.name}", ALERT_COLORS[fcw.active_level])
        if result.ldw is not None:
            ldw = result.ldw
            offset: float | None = ldw.lane_position.offset_ratio if ldw.lane_position is not None else None
            lines.append(f"LDW offset={_format_optional(value=offset, unit='')}  side={ldw.active_side.value}")
            if banner is None and ldw.active_side is not LaneDepartureSide.NONE:
                banner = (f"LANE DEPARTURE {ldw.active_side.value.upper()}", Palette.CAUTION)

        cv2.rectangle(img=canvas, pt1=(0, 0), pt2=(canvas.shape[1], 22 * len(lines) + 8), color=Palette.HUD_BACKGROUND.bgr, thickness=-1)
        for index, line in enumerate(lines):
            cv2.putText(
                img=canvas,
                text=line,
                org=(8, 20 + 22 * index),
                fontFace=FONT,
                fontScale=0.55,
                color=Palette.TEXT.bgr,
                thickness=1,
                lineType=cv2.LINE_AA,
            )
        if banner is not None:
            text, color = banner
            height, width = canvas.shape[:2]
            cv2.rectangle(img=canvas, pt1=(0, height - 60), pt2=(width, height), color=color.bgr, thickness=-1)
            cv2.putText(
                img=canvas,
                text=text,
                org=(20, height - 18),
                fontFace=FONT,
                fontScale=1.1,
                color=Palette.TEXT.bgr,
                thickness=2,
                lineType=cv2.LINE_AA,
            )


def _format_optional(*, value: float | None, unit: str) -> str:
    return "--" if value is None else f"{value:.2f}{unit}"
