"""Ego-vehicle speed providers (GPS on iPhone; simulated when replaying video)."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Protocol

import numpy as np
import numpy.typing as npt

from dashcam_ml.config.schema import EgoSpeedConfig
from dashcam_ml.domain.enums import EgoSpeedSource


class EgoSpeedProvider(Protocol):
    def speed_kmh(self, *, timestamp_s: float) -> float | None: ...


class NoSpeedProvider:
    def speed_kmh(self, *, timestamp_s: float) -> float | None:
        return None


class ConstantSpeedProvider:
    def __init__(self, *, speed_kmh: float) -> None:
        self._speed_kmh: float = speed_kmh

    def speed_kmh(self, *, timestamp_s: float) -> float | None:
        return self._speed_kmh


class CsvSpeedProvider:
    """Linear interpolation over a CSV with header ``timestamp_s,speed_kmh``."""

    def __init__(self, *, csv_path: Path) -> None:
        timestamps: list[float] = []
        speeds: list[float] = []
        with csv_path.open(mode="r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(f=handle):
                timestamps.append(float(row["timestamp_s"]))
                speeds.append(float(row["speed_kmh"]))
        if not timestamps:
            raise ValueError(f"speed CSV is empty: {csv_path}")
        order: npt.NDArray[np.intp] = np.argsort(a=np.asarray(timestamps))
        self._timestamps: npt.NDArray[np.float64] = np.asarray(timestamps, dtype=np.float64)[order]
        self._speeds: npt.NDArray[np.float64] = np.asarray(speeds, dtype=np.float64)[order]

    def speed_kmh(self, *, timestamp_s: float) -> float | None:
        return float(np.interp(x=timestamp_s, xp=self._timestamps, fp=self._speeds))


def build_speed_provider(*, config: EgoSpeedConfig) -> EgoSpeedProvider:
    match config.source:
        case EgoSpeedSource.CONSTANT:
            assert config.constant_kmh is not None
            return ConstantSpeedProvider(speed_kmh=config.constant_kmh)
        case EgoSpeedSource.CSV:
            assert config.csv_path is not None
            return CsvSpeedProvider(csv_path=config.csv_path)
        case EgoSpeedSource.NONE:
            return NoSpeedProvider()
