from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from dashcam_ml.config.schema import DashcamConfig, TfliteProfile
from dashcam_ml.domain.enums import ModelKind, WeightPrecision
from dashcam_ml.export.manifest import (
    IOS_MANIFEST_HEADER,
    ModelManifestEntry,
    android_manifest_header,
    read_manifest,
    upsert_manifest,
)
from dashcam_ml.export.yolo_tflite import output_layout

IOS_ENTRY_KEYS: list[str] = [
    "name",
    "kind",
    "file_name",
    "input_width",
    "input_height",
    "precision",
    "source_weights",
    "target_devices",
    "input_names",
    "output_names",
    "class_names",
    "includes_nms",
]


def _entry(*, name: str, layout: str | None) -> ModelManifestEntry:
    return ModelManifestEntry(
        name=name,
        kind=ModelKind.DETECTOR,
        file_name=f"{name}.bin",
        input_width=416,
        input_height=416,
        precision=WeightPrecision.FP16,
        source_weights="best.pt",
        target_devices="",
        input_names=["image"],
        output_names=["confidence", "coordinates"],
        class_names=["car"],
        includes_nms=layout is None,
        output_layout=layout,
    )


def test_ios_manifest_keeps_the_keys_the_app_knows(tmp_path: Path) -> None:
    path: Path = upsert_manifest(
        output_dir=tmp_path, entries=[_entry(name="yolo11n_416_fp16", layout=None)], header=IOS_MANIFEST_HEADER
    )
    data: dict[str, object] = json.loads(path.read_text(encoding="utf-8"))
    assert list(data) == ["schema_version", "minimum_ios_version", "models"]
    assert list(data["models"][0]) == IOS_ENTRY_KEYS  # type: ignore[index]


def test_android_manifest_has_sdk_header_and_layout(tmp_path: Path) -> None:
    path: Path = upsert_manifest(
        output_dir=tmp_path,
        entries=[_entry(name="yolo11n_416_fp32", layout="raw anchors")],
        header=android_manifest_header(minimum_android_sdk=26),
    )
    data: dict[str, object] = json.loads(path.read_text(encoding="utf-8"))
    assert data["minimum_android_sdk"] == 26
    assert "minimum_ios_version" not in data
    assert read_manifest(output_dir=tmp_path)[0].output_layout == "raw anchors"


def test_tflite_profile_rejects_fp16() -> None:
    with pytest.raises(expected_exception=ValidationError):
        TfliteProfile.model_validate(obj={"name": "x", "imgsz": 416, "precision": "fp16"})


def test_tflite_profiles_in_config_are_fp32_or_int8(config: DashcamConfig) -> None:
    assert config.tflite_export.profiles
    assert {profile.precision for profile in config.tflite_export.profiles} <= {
        WeightPrecision.FP32,
        WeightPrecision.INT8,
    }


def test_output_layout_describes_shapes_and_missing_nms() -> None:
    layout: str = output_layout(input_shape=[1, 416, 416, 3], output_shape=[1, 14, 3549], class_count=10)
    assert "[1, 416, 416, 3]" in layout
    assert "[1, 14, 3549]" in layout
    assert "NMS not included" in layout
