"""Lane Departure Warning from a binary lane-line mask (TwinLiteNet output).

For each sampled row near the bottom of the frame, the inner edges of the lane
lines left and right of the vehicle centre give the lane centre and width. The
median over rows is compared with the vehicle centre.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from dashcam_ml.config.schema import LdwConfig
from dashcam_ml.core.debounce import AlertDebouncer, DebounceOutput
from dashcam_ml.domain.entities import LanePosition, LdwResult
from dashcam_ml.domain.enums import LaneDepartureSide

BoolMask = npt.NDArray[np.bool_]


@dataclass(frozen=True, slots=True)
class LaneSegmentation:
    """Binary masks at model resolution (``input_height x input_width``)."""

    drivable_mask: BoolMask
    lane_mask: BoolMask


def estimate_lane_position(*, lane_mask: BoolMask, config: LdwConfig) -> LanePosition | None:
    mask_height, mask_width = lane_mask.shape
    center_px: float = config.vehicle_center_x * mask_width
    first_row: int = int(config.roi_top_y * mask_height)
    last_row: int = min(mask_height - 1, int(config.roi_bottom_y * mask_height))
    sampled_rows: range = range(first_row, last_row + 1, config.row_step)
    if len(sampled_rows) == 0:
        return None

    centers: list[float] = []
    widths: list[float] = []
    for row in sampled_rows:
        columns: npt.NDArray[np.intp] = np.flatnonzero(lane_mask[row])
        left_columns: npt.NDArray[np.intp] = columns[columns < center_px]
        right_columns: npt.NDArray[np.intp] = columns[columns >= center_px]
        if left_columns.size == 0 or right_columns.size == 0:
            continue
        left_edge: float = float(left_columns.max())
        right_edge: float = float(right_columns.min())
        width_ratio: float = (right_edge - left_edge) / mask_width
        if config.min_lane_width_ratio <= width_ratio <= config.max_lane_width_ratio:
            centers.append((left_edge + right_edge) / 2.0)
            widths.append(right_edge - left_edge)

    valid_rows_ratio: float = len(centers) / len(sampled_rows)
    if valid_rows_ratio < config.min_valid_rows_ratio:
        return None
    lane_center_px: float = float(np.median(a=np.asarray(centers)))
    lane_width_px: float = float(np.median(a=np.asarray(widths)))
    return LanePosition(
        lane_center_x=lane_center_px / mask_width,
        lane_width=lane_width_px / mask_width,
        offset_ratio=(center_px - lane_center_px) / (lane_width_px / 2.0),
        valid_rows_ratio=valid_rows_ratio,
    )


class LaneDepartureWarner:
    def __init__(self, *, config: LdwConfig) -> None:
        self._config: LdwConfig = config
        self._left_debouncer: AlertDebouncer[LaneDepartureSide] = self._build_debouncer(side=LaneDepartureSide.LEFT)
        self._right_debouncer: AlertDebouncer[LaneDepartureSide] = self._build_debouncer(side=LaneDepartureSide.RIGHT)

    def reset(self) -> None:
        self._left_debouncer.reset()
        self._right_debouncer.reset()

    def update(
        self,
        *,
        lane_position: LanePosition | None,
        timestamp_s: float,
        ego_speed_kmh: float | None,
        turn_signal_active: bool = False,
    ) -> LdwResult:
        """``turn_signal_active`` is confirmed by the driver in the app (iOS cannot read the blinker)."""
        raw_side: LaneDepartureSide = self._raw_side(
            lane_position=lane_position, ego_speed_kmh=ego_speed_kmh, turn_signal_active=turn_signal_active
        )
        left: DebounceOutput[LaneDepartureSide] = self._left_debouncer.update(
            raw_states=[raw_side] if raw_side is LaneDepartureSide.LEFT else [], timestamp_s=timestamp_s
        )
        right: DebounceOutput[LaneDepartureSide] = self._right_debouncer.update(
            raw_states=[raw_side] if raw_side is LaneDepartureSide.RIGHT else [], timestamp_s=timestamp_s
        )
        active_side: LaneDepartureSide = LaneDepartureSide.NONE
        if left.active_state is LaneDepartureSide.LEFT:
            active_side = LaneDepartureSide.LEFT
        elif right.active_state is LaneDepartureSide.RIGHT:
            active_side = LaneDepartureSide.RIGHT
        return LdwResult(
            lane_position=lane_position,
            raw_side=raw_side,
            active_side=active_side,
            alert_triggered=left.alert_triggered or right.alert_triggered,
        )

    def _raw_side(
        self, *, lane_position: LanePosition | None, ego_speed_kmh: float | None, turn_signal_active: bool
    ) -> LaneDepartureSide:
        if lane_position is None or turn_signal_active:
            return LaneDepartureSide.NONE
        if ego_speed_kmh is not None and ego_speed_kmh < self._config.min_ego_speed_kmh:
            return LaneDepartureSide.NONE
        if lane_position.offset_ratio >= self._config.departure_threshold:
            return LaneDepartureSide.RIGHT
        if lane_position.offset_ratio <= -self._config.departure_threshold:
            return LaneDepartureSide.LEFT
        return LaneDepartureSide.NONE

    def _build_debouncer(self, *, side: LaneDepartureSide) -> AlertDebouncer[LaneDepartureSide]:
        return AlertDebouncer(
            idle_state=LaneDepartureSide.NONE,
            ordered_states=(side,),
            min_consecutive_frames=self._config.min_consecutive_frames,
            cooldown_s=self._config.cooldown_s,
        )
