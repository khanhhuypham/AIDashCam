from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from dashcam_video.config.schema import DashcamConfig, FcwConfig
from dashcam_video.domain.enums import ClassCategory, EgoSpeedSource, ReferenceDimension
from dashcam_video.export.ios_config import build_ios_config
from tests.conftest import PROJECT_ROOT


def test_config_loads_with_enums(config: DashcamConfig) -> None:
    assert config.ego_speed.source is EgoSpeedSource.CONSTANT
    car = next(spec for spec in config.classes if spec.name == "car")
    assert car.category is ClassCategory.VEHICLE
    assert car.reference_dimension is ReferenceDimension.WIDTH


def test_relative_paths_resolve_against_config_dir(config: DashcamConfig) -> None:
    assert config.video.output_dir == (PROJECT_ROOT / "outputs" / "video").resolve()
    assert config.inference.weights.is_absolute()


def test_invalid_threshold_order_is_rejected(config: DashcamConfig) -> None:
    raw: dict[str, object] = config.fcw.model_dump()
    raw["ttc_critical_s"] = 5.0
    with pytest.raises(expected_exception=ValidationError):
        FcwConfig.model_validate(obj=raw)


def test_unknown_enum_value_is_rejected(config: DashcamConfig) -> None:
    raw: dict[str, object] = config.model_dump()
    raw["ego_speed"]["source"] = "gps"  # type: ignore[index]
    with pytest.raises(expected_exception=ValidationError):
        DashcamConfig.model_validate(obj=raw, context={"base_dir": str(Path.cwd())})


def test_ios_config_keeps_the_app_schema(config: DashcamConfig, tmp_path: Path) -> None:
    raw: dict[str, object] = config.model_dump()
    raw["ios_config_export"]["models_manifest"] = str(tmp_path / "missing.json")  # type: ignore[index]
    payload: dict[str, object] = build_ios_config(
        config=DashcamConfig.model_validate(obj=raw, context={"base_dir": str(PROJECT_ROOT)})
    )
    assert list(payload) == [
        "schema_version",
        "camera",
        "classes",
        "inference",
        "tracker",
        "fcw",
        "ldw",
        "lane_segmenter",
        "models",
    ]
    assert payload["models"] == []
