"""Forward Collision Warning: lead-vehicle selection, TTC and alert levels.

TTC is estimated from the growth rate of the lead box (``TTC = s / (ds/dt)``), which
does not need camera calibration. Distance (pinhole model) is only used for the
headway check and for filtering far-away objects.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from dashcam_ml.config.schema import ClassSpec, FcwConfig
from dashcam_ml.core.class_catalog import ClassCatalog
from dashcam_ml.core.debounce import AlertDebouncer, DebounceOutput
from dashcam_ml.core.geometry import CameraModel, estimate_distance_m, is_in_path, reference_box_size
from dashcam_ml.core.tracking import Track
from dashcam_ml.domain.entities import FcwResult
from dashcam_ml.domain.enums import AlertLevel

KMH_TO_MPS: float = 1000.0 / 3600.0
_MIN_VARIANCE: float = 1e-9


@dataclass(frozen=True, slots=True)
class _LeadCandidate:
    track: Track
    spec: ClassSpec
    distance_m: float | None


def estimate_ttc_s(*, samples: Sequence[tuple[float, float]]) -> float | None:
    """Time-to-collision from ``(timestamp_s, box_size)`` samples via least squares.

    Returns ``None`` when the object is not approaching (size not growing).
    """
    count: int = len(samples)
    if count < 2:
        return None
    mean_t: float = sum(sample[0] for sample in samples) / count
    mean_size: float = sum(sample[1] for sample in samples) / count
    variance: float = sum((sample[0] - mean_t) ** 2 for sample in samples)
    if variance <= _MIN_VARIANCE:
        return None
    covariance: float = sum((sample[0] - mean_t) * (sample[1] - mean_size) for sample in samples)
    slope: float = covariance / variance
    if slope <= 0.0:
        return None
    latest_size: float = mean_size + slope * (samples[-1][0] - mean_t)
    if latest_size <= 0.0:
        return None
    return latest_size / slope


class ForwardCollisionWarner:
    def __init__(self, *, config: FcwConfig, catalog: ClassCatalog, camera: CameraModel) -> None:
        self._config: FcwConfig = config
        self._catalog: ClassCatalog = catalog
        self._camera: CameraModel = camera
        self._debouncer: AlertDebouncer[AlertLevel] = AlertDebouncer(
            idle_state=AlertLevel.NONE,
            ordered_states=(AlertLevel.CAUTION, AlertLevel.CRITICAL),
            min_consecutive_frames=config.min_consecutive_frames,
            cooldown_s=config.cooldown_s,
        )
        self._smoothed_ttc_s: float | None = None
        self._smoothed_track_id: int | None = None

    def reset(self) -> None:
        self._debouncer.reset()
        self._smoothed_ttc_s = None
        self._smoothed_track_id = None

    def update(self, *, tracks: Sequence[Track], timestamp_s: float, ego_speed_kmh: float | None) -> FcwResult:
        lead: _LeadCandidate | None = self._select_lead(tracks=tracks)
        ttc_s: float | None = None
        headway_s: float | None = None
        if lead is not None:
            ttc_s = self._smooth_ttc(track_id=lead.track.track_id, ttc_s=self._lead_ttc_s(lead=lead, now_s=timestamp_s))
            if lead.distance_m is not None and ego_speed_kmh is not None and ego_speed_kmh > 0.0:
                headway_s = lead.distance_m / (ego_speed_kmh * KMH_TO_MPS)
        else:
            self._smoothed_ttc_s = None
            self._smoothed_track_id = None

        raw_level: AlertLevel = self._raw_level(ttc_s=ttc_s, headway_s=headway_s, ego_speed_kmh=ego_speed_kmh)
        satisfied: list[AlertLevel] = [level for level in (AlertLevel.CAUTION, AlertLevel.CRITICAL) if raw_level >= level]
        debounced: DebounceOutput[AlertLevel] = self._debouncer.update(raw_states=satisfied, timestamp_s=timestamp_s)
        return FcwResult(
            lead_track_id=lead.track.track_id if lead is not None else None,
            distance_m=lead.distance_m if lead is not None else None,
            ttc_s=ttc_s,
            headway_s=headway_s,
            raw_level=raw_level,
            active_level=debounced.active_state,
            alert_triggered=debounced.alert_triggered,
        )

    def _select_lead(self, *, tracks: Sequence[Track]) -> _LeadCandidate | None:
        """Nearest in-path track; falls back to the lowest box when distance is unknown."""
        best: _LeadCandidate | None = None
        best_key: tuple[float, float] | None = None
        for track in tracks:
            spec: ClassSpec | None = self._catalog.lookup(class_name=track.class_name)
            if spec is None or not is_in_path(box=track.box, roi=self._config.in_path_roi):
                continue
            distance_m: float | None = estimate_distance_m(box=track.box, spec=spec, camera=self._camera)
            if distance_m is not None and distance_m > self._config.max_distance_m:
                continue
            key: tuple[float, float] = (
                distance_m if distance_m is not None else float("inf"),
                -track.box.bottom_y,
            )
            if best_key is None or key < best_key:
                best = _LeadCandidate(track=track, spec=spec, distance_m=distance_m)
                best_key = key
        return best

    def _lead_ttc_s(self, *, lead: _LeadCandidate, now_s: float) -> float | None:
        window_start_s: float = now_s - self._config.ttc_window_s
        samples: list[tuple[float, float]] = [
            (sample_s, reference_box_size(box=box, spec=lead.spec))
            for sample_s, box in lead.track.history
            if sample_s >= window_start_s
        ]
        if len(samples) < self._config.ttc_min_samples:
            return None
        return estimate_ttc_s(samples=samples)

    def _smooth_ttc(self, *, track_id: int, ttc_s: float | None) -> float | None:
        if ttc_s is None or track_id != self._smoothed_track_id or self._smoothed_ttc_s is None:
            self._smoothed_ttc_s = ttc_s
        else:
            alpha: float = self._config.ttc_smoothing_alpha
            self._smoothed_ttc_s = alpha * ttc_s + (1.0 - alpha) * self._smoothed_ttc_s
        self._smoothed_track_id = track_id
        return self._smoothed_ttc_s

    def _raw_level(self, *, ttc_s: float | None, headway_s: float | None, ego_speed_kmh: float | None) -> AlertLevel:
        if ego_speed_kmh is not None and ego_speed_kmh < self._config.min_ego_speed_kmh:
            return AlertLevel.NONE
        if ttc_s is not None and ttc_s <= self._config.ttc_critical_s:
            return AlertLevel.CRITICAL
        if ttc_s is not None and ttc_s <= self._config.ttc_caution_s:
            return AlertLevel.CAUTION
        if (
            headway_s is not None
            and self._config.headway_caution_s is not None
            and headway_s <= self._config.headway_caution_s
        ):
            return AlertLevel.CAUTION
        return AlertLevel.NONE
