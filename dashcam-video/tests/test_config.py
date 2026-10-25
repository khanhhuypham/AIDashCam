from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from dashcam_video.config.schema import AppConfigExportConfig, AppConfigTarget, FcwConfig, RootConfig
from dashcam_video.domain.enums import ClassCategory, EgoSpeedSource, Platform, ReferenceDimension
from dashcam_video.export.app_config import build_app_config
from tests.conftest import PROJECT_ROOT

APP_CONFIG_KEYS: list[str] = [
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


def test_config_loads_with_enums(config: RootConfig) -> None:
    assert config.ego_speed.source is EgoSpeedSource.CONSTANT
    car = next(spec for spec in config.classes if spec.name == "car")
    assert car.category is ClassCategory.VEHICLE
    assert car.reference_dimension is ReferenceDimension.WIDTH


def test_relative_paths_resolve_against_config_dir(config: RootConfig) -> None:
    assert config.video.output_dir == (PROJECT_ROOT / "outputs" / "video").resolve()
    assert config.inference.weights.is_absolute()


def test_invalid_threshold_order_is_rejected(config: RootConfig) -> None:
    raw: dict[str, object] = config.fcw.model_dump()
    raw["ttc_critical_s"] = 5.0
    with pytest.raises(expected_exception=ValidationError):
        FcwConfig.model_validate(obj=raw)


def test_unknown_enum_value_is_rejected(config: RootConfig) -> None:
    raw: dict[str, object] = config.model_dump()
    raw["ego_speed"]["source"] = "gps"  # type: ignore[index]
    with pytest.raises(expected_exception=ValidationError):
        RootConfig.model_validate(obj=raw, context={"base_dir": str(Path.cwd())})


@pytest.mark.parametrize("platform", list(Platform))
def test_app_config_keeps_the_app_schema(config: RootConfig, tmp_path: Path, platform: Platform) -> None:
    target: AppConfigTarget = AppConfigTarget(
        platform=platform, models_manifest=tmp_path / "missing.json", output_path=tmp_path / "DashcamConfig.json"
    )
    payload: dict[str, object] = build_app_config(config=config, target=target)
    assert list(payload) == APP_CONFIG_KEYS
    assert payload["models"] == []


def test_ios_and_android_configs_differ_only_in_models(config: RootConfig) -> None:
    ios: dict[str, object] = build_app_config(
        config=config, target=config.app_config_export.target(platform=Platform.IOS)
    )
    android: dict[str, object] = build_app_config(
        config=config, target=config.app_config_export.target(platform=Platform.ANDROID)
    )
    assert {key: value for key, value in ios.items() if key != "models"} == {
        key: value for key, value in android.items() if key != "models"
    }


def test_each_platform_is_listed_once(config: RootConfig) -> None:
    target: AppConfigTarget = config.app_config_export.target(platform=Platform.IOS)
    with pytest.raises(expected_exception=ValidationError):
        AppConfigExportConfig.model_validate(obj={"targets": [target.model_dump(), target.model_dump()]})
