from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pytest

from dashcam_ml.config.schema import DashcamConfig
from dashcam_ml.core.ldw import LaneDepartureWarner, estimate_lane_position
from dashcam_ml.domain.entities import LanePosition, LdwResult
from dashcam_ml.domain.enums import LaneDepartureSide

MASK_HEIGHT: int = 360
MASK_WIDTH: int = 640


def _lane_mask(*, left_x: int, right_x: int) -> npt.NDArray[np.bool_]:
    mask: npt.NDArray[np.bool_] = np.zeros(shape=(MASK_HEIGHT, MASK_WIDTH), dtype=bool)
    mask[:, left_x - 3 : left_x] = True
    mask[:, right_x : right_x + 3] = True
    return mask


def test_centered_lane_has_zero_offset(config: DashcamConfig) -> None:
    position: LanePosition | None = estimate_lane_position(lane_mask=_lane_mask(left_x=160, right_x=480), config=config.ldw)
    assert position is not None
    assert position.offset_ratio == pytest.approx(expected=0.0, abs=0.02)
    assert position.lane_width == pytest.approx(expected=0.5, abs=0.02)


def test_empty_mask_gives_no_position(config: DashcamConfig) -> None:
    empty: npt.NDArray[np.bool_] = np.zeros(shape=(MASK_HEIGHT, MASK_WIDTH), dtype=bool)
    assert estimate_lane_position(lane_mask=empty, config=config.ldw) is None


def test_drift_right_triggers_after_consecutive_frames(config: DashcamConfig) -> None:
    warner: LaneDepartureWarner = LaneDepartureWarner(config=config.ldw)
    # lane shifted left -> vehicle centre sits near the right line
    position: LanePosition | None = estimate_lane_position(lane_mask=_lane_mask(left_x=40, right_x=360), config=config.ldw)
    assert position is not None and position.offset_ratio > config.ldw.departure_threshold
    results: list[LdwResult] = [
        warner.update(lane_position=position, timestamp_s=0.1 * step, ego_speed_kmh=60.0) for step in range(10)
    ]
    first_active: int = next(index for index, result in enumerate(results) if result.active_side is LaneDepartureSide.RIGHT)
    assert first_active == config.ldw.min_consecutive_frames - 1
    assert sum(result.alert_triggered for result in results) == 1


def test_turn_signal_suppresses_warning(config: DashcamConfig) -> None:
    warner: LaneDepartureWarner = LaneDepartureWarner(config=config.ldw)
    position: LanePosition | None = estimate_lane_position(lane_mask=_lane_mask(left_x=40, right_x=360), config=config.ldw)
    results: list[LdwResult] = [
        warner.update(lane_position=position, timestamp_s=0.1 * step, ego_speed_kmh=60.0, turn_signal_active=True)
        for step in range(10)
    ]
    assert all(result.active_side is LaneDepartureSide.NONE for result in results)
