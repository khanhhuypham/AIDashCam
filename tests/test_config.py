from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from dashcam_ml.config.schema import DashcamConfig, SplitRatios
from dashcam_ml.domain.enums import CopyMode
from tests.conftest import PROJECT_ROOT


def test_config_loads_with_enums(config: DashcamConfig) -> None:
    assert config.bdd100k.copy_mode is CopyMode.HARDLINK
    pedestrian = next(spec for spec in config.classes if spec.name == "pedestrian")
    assert "person" in pedestrian.aliases


def test_relative_paths_resolve_against_config_dir(config: DashcamConfig) -> None:
    assert config.project.output_dir == (PROJECT_ROOT / "outputs").resolve()
    assert config.coreml_export.weights.is_absolute()


def test_split_ratios_must_sum_to_one() -> None:
    with pytest.raises(expected_exception=ValidationError):
        SplitRatios.model_validate(obj={"train": 0.7, "val": 0.2, "test": 0.2})


def test_unknown_enum_value_is_rejected(config: DashcamConfig) -> None:
    raw: dict[str, object] = config.model_dump()
    raw["bdd100k"]["copy_mode"] = "symlink"  # type: ignore[index]
    with pytest.raises(expected_exception=ValidationError):
        DashcamConfig.model_validate(obj=raw, context={"base_dir": str(Path.cwd())})
