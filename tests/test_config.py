from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from dashcam_ml.config.schema import DashcamConfig, FcwConfig
from dashcam_ml.domain.enums import ClassCategory, CopyMode, EgoSpeedSource, ReferenceDimension
from tests.conftest import PROJECT_ROOT


def test_config_loads_with_enums(config: DashcamConfig) -> None:
    assert config.bdd100k.copy_mode is CopyMode.HARDLINK
    assert config.ego_speed.source is EgoSpeedSource.CONSTANT
    car = next(spec for spec in config.classes if spec.name == "car")
    assert car.category is ClassCategory.VEHICLE
    assert car.reference_dimension is ReferenceDimension.WIDTH


def test_relative_paths_resolve_against_config_dir(config: DashcamConfig) -> None:
    assert config.project.output_dir == (PROJECT_ROOT / "outputs").resolve()
    assert config.inference.weights.is_absolute()


def test_invalid_threshold_order_is_rejected(config: DashcamConfig) -> None:
    raw: dict[str, object] = config.fcw.model_dump()
    raw["ttc_critical_s"] = 5.0
    with pytest.raises(expected_exception=ValidationError):
        FcwConfig.model_validate(obj=raw)


def test_unknown_enum_value_is_rejected(config: DashcamConfig) -> None:
    raw: dict[str, object] = config.model_dump()
    raw["bdd100k"]["copy_mode"] = "symlink"  # type: ignore[index]
    with pytest.raises(expected_exception=ValidationError):
        DashcamConfig.model_validate(obj=raw, context={"base_dir": str(Path.cwd())})
