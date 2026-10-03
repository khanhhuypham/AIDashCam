from __future__ import annotations

import pytest

from dashcam_ml.config.schema import DashcamConfig
from dashcam_ml.core.class_catalog import ClassCatalog
from dashcam_ml.core.fcw import ForwardCollisionWarner, estimate_ttc_s
from dashcam_ml.core.geometry import CameraModel, estimate_distance_m, is_in_path
from dashcam_ml.core.tracking import IouTracker, Track
from dashcam_ml.domain.entities import BoundingBox, Detection, FcwResult
from dashcam_ml.domain.enums import AlertLevel

CAMERA: CameraModel = CameraModel(horizontal_fov_deg=63.0, aspect_ratio=16.0 / 9.0)
FRAME_DT_S: float = 0.1


def _lead_car_box(*, distance_m: float) -> BoundingBox:
    """Centred car of width 1.8 m at the given distance."""
    width: float = 1.8 / (distance_m * 2.0 * CAMERA.tan_half_hfov)
    return BoundingBox(x1=0.5 - width / 2.0, y1=0.75 - width * 0.8, x2=0.5 + width / 2.0, y2=0.75)


def test_ttc_from_linear_growth() -> None:
    # size doubles every 1 s from s=1 at t=0 -> at t=1, TTC = s / (ds/dt) = 2 / 1 = 2
    samples: list[tuple[float, float]] = [(0.1 * step, 1.0 + 0.1 * step) for step in range(11)]
    ttc: float | None = estimate_ttc_s(samples=samples)
    assert ttc == pytest.approx(expected=2.0, rel=1e-6)


def test_ttc_is_none_when_not_approaching() -> None:
    assert estimate_ttc_s(samples=[(0.0, 1.0), (0.1, 1.0), (0.2, 0.99)]) is None


def test_distance_estimate_round_trip(config: DashcamConfig) -> None:
    spec = ClassCatalog(specs=config.classes).lookup(class_name="car")
    assert spec is not None
    distance: float | None = estimate_distance_m(box=_lead_car_box(distance_m=20.0), spec=spec, camera=CAMERA)
    assert distance == pytest.approx(expected=20.0, rel=1e-6)


def test_in_path_roi(config: DashcamConfig) -> None:
    roi = config.fcw.in_path_roi
    assert is_in_path(box=BoundingBox(x1=0.45, y1=0.6, x2=0.55, y2=0.8), roi=roi)
    assert not is_in_path(box=BoundingBox(x1=0.0, y1=0.6, x2=0.1, y2=0.8), roi=roi)


def _simulate(*, config: DashcamConfig, closing_speed_mps: float, ego_speed_kmh: float) -> list[FcwResult]:
    tracker: IouTracker = IouTracker(config=config.tracker)
    warner: ForwardCollisionWarner = ForwardCollisionWarner(
        config=config.fcw, catalog=ClassCatalog(specs=config.classes), camera=CAMERA
    )
    results: list[FcwResult] = []
    distance_m: float = 25.0
    for step in range(40):
        timestamp_s: float = step * FRAME_DT_S
        detection: Detection = Detection(box=_lead_car_box(distance_m=distance_m), class_name="car", confidence=0.9)
        tracks: list[Track] = tracker.update(detections=[detection], timestamp_s=timestamp_s)
        results.append(warner.update(tracks=tracks, timestamp_s=timestamp_s, ego_speed_kmh=ego_speed_kmh))
        distance_m = max(2.0, distance_m - closing_speed_mps * FRAME_DT_S)
    return results


def test_fast_approach_triggers_critical_once_per_cooldown(config: DashcamConfig) -> None:
    results: list[FcwResult] = _simulate(config=config, closing_speed_mps=10.0, ego_speed_kmh=50.0)
    assert any(result.active_level is AlertLevel.CRITICAL for result in results)
    triggered: list[FcwResult] = [result for result in results if result.alert_triggered]
    assert 1 <= len(triggered) <= 2  # at most CAUTION then an escalation to CRITICAL


def test_constant_distance_does_not_alert(config: DashcamConfig) -> None:
    results: list[FcwResult] = _simulate(config=config, closing_speed_mps=0.0, ego_speed_kmh=20.0)
    assert all(result.active_level is AlertLevel.NONE for result in results)


def test_low_ego_speed_suppresses_alerts(config: DashcamConfig) -> None:
    results: list[FcwResult] = _simulate(config=config, closing_speed_mps=10.0, ego_speed_kmh=5.0)
    assert not any(result.alert_triggered for result in results)
