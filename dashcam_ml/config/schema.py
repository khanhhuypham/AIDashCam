"""Typed, immutable schema of ``config.yaml``.

Each section of the YAML file maps to one class below. Unknown keys are rejected
(``extra="forbid"``) so typos are caught early. Relative paths are resolved against
the folder that contains the config file.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

from pydantic import AfterValidator, BaseModel, ConfigDict, Field, ValidationInfo, field_validator, model_validator

from dashcam_ml.domain.enums import CopyMode, DatasetSplit, LogLevel, WeightPrecision

BASE_DIR_CONTEXT_KEY: str = "base_dir"
YOLO_STRIDE: int = 32


def _resolve_project_path(value: Path, info: ValidationInfo) -> Path:
    if value.is_absolute():
        return value
    context: dict[str, object] = info.context or {}
    base_dir: object = context.get(BASE_DIR_CONTEXT_KEY)
    root: Path = Path(str(base_dir)) if base_dir is not None else Path.cwd()
    return (root / value).resolve()


ProjectPath = Annotated[Path, AfterValidator(_resolve_project_path)]
ImageSize = int | tuple[int, int]
"""Model input size: a square side, or ``(height, width)``. Must be multiples of 32."""


def _validate_image_size(*, value: ImageSize) -> ImageSize:
    sides: tuple[int, ...] = (value,) if isinstance(value, int) else value
    for side in sides:
        if side <= 0 or side % YOLO_STRIDE != 0:
            raise ValueError(f"image size {value!r} must be positive multiples of {YOLO_STRIDE}")
    return value


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


# ----------------------------------------------------------------------------- general


class ProjectConfig(StrictModel):
    name: str
    output_dir: ProjectPath
    seed: int = 42
    device: str = "auto"
    log_level: LogLevel = LogLevel.INFO


class ClassSpec(StrictModel):
    """Project class name and its other names (COCO ``person`` = ``pedestrian``)."""

    name: str
    aliases: tuple[str, ...] = ()


# ----------------------------------------------------------------------------- data


class Bdd100kConfig(StrictModel):
    dataset_root: ProjectPath
    """Dataset Ninja (Supervisely) export: ``<root>/<split>/img/*.jpg`` + ``<root>/<split>/ann/*.jpg.json``."""
    output_dir: ProjectPath
    class_names: tuple[str, ...]
    category_aliases: dict[str, str] = Field(default_factory=dict)
    copy_mode: CopyMode = CopyMode.HARDLINK
    max_images_per_split: int | None = Field(default=None, gt=0)
    condition_splits: dict[str, dict[str, str]] = Field(default_factory=dict)


class FrameExtractionConfig(StrictModel):
    videos_dir: ProjectPath
    output_dir: ProjectPath
    video_extensions: tuple[str, ...] = (".mp4", ".mov")
    every_n_seconds: float = Field(gt=0.0)
    blur_threshold: float = Field(ge=0.0)
    duplicate_threshold: float = Field(ge=0.0)
    jpeg_quality: int = Field(ge=1, le=100)
    name_separator: str = "__"


class SplitRatios(StrictModel):
    train: float = Field(ge=0.0, le=1.0)
    val: float = Field(ge=0.0, le=1.0)
    test: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _check_sum(self) -> SplitRatios:
        total: float = self.train + self.val + self.test
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"split ratios must sum to 1.0, got {total}")
        return self

    def as_dict(self) -> dict[DatasetSplit, float]:
        return {DatasetSplit.TRAIN: self.train, DatasetSplit.VAL: self.val, DatasetSplit.TEST: self.test}


class SplitConfig(StrictModel):
    images_dir: ProjectPath
    labels_dir: ProjectPath
    output_dir: ProjectPath
    class_names: tuple[str, ...]
    name_separator: str = "__"
    copy_mode: CopyMode = CopyMode.HARDLINK
    ratios: SplitRatios


# ----------------------------------------------------------------------------- training


class TrainConfig(StrictModel):
    model: ProjectPath
    data_yaml: ProjectPath
    epochs: int = Field(gt=0)
    imgsz: int = Field(gt=0)
    batch: int
    workers: int = Field(ge=0)
    patience: int = Field(ge=0)
    optimizer: str = "auto"
    lr0: float = Field(gt=0.0)
    cos_lr: bool = False
    amp: bool = True
    cache: bool = False
    resume: bool = False
    run_name: str

    @field_validator("imgsz")
    @classmethod
    def _check_imgsz(cls, value: int) -> int:
        _validate_image_size(value=value)
        return value


class EvaluateConfig(StrictModel):
    weights: tuple[ProjectPath, ...]
    data_yaml: ProjectPath
    split: str = DatasetSplit.TEST.value
    imgsz_list: tuple[ImageSize, ...]
    conf: float = Field(ge=0.0, le=1.0)
    iou: float = Field(ge=0.0, le=1.0)
    batch: int = Field(gt=0)
    focus_classes: tuple[str, ...] = ()
    copy_mode: CopyMode = CopyMode.HARDLINK

    @field_validator("imgsz_list")
    @classmethod
    def _check_sizes(cls, value: tuple[ImageSize, ...]) -> tuple[ImageSize, ...]:
        for size in value:
            _validate_image_size(value=size)
        return value


# ----------------------------------------------------------------------------- phương án B


class TwinLiteNetConfig(StrictModel):
    repo_dir: ProjectPath
    module: str
    class_name: str
    weights: ProjectPath
    state_dict_prefix: str = "module."
    input_width: int = Field(gt=0)
    input_height: int = Field(gt=0)
    mask_threshold: float = Field(gt=0.0, lt=1.0)


# ----------------------------------------------------------------------------- Core ML


class CoremlProfile(StrictModel):
    name: str = Field(pattern=r"^[A-Za-z0-9_\-]+$")
    imgsz: ImageSize
    precision: WeightPrecision = WeightPrecision.FP16
    target_devices: str = ""

    @field_validator("imgsz")
    @classmethod
    def _check_imgsz(cls, value: ImageSize) -> ImageSize:
        return _validate_image_size(value=value)


class CoremlExportConfig(StrictModel):
    weights: ProjectPath
    output_dir: ProjectPath
    nms: bool = True
    profiles: tuple[CoremlProfile, ...] = Field(min_length=1)


class TwinLiteNetExportConfig(StrictModel):
    output_dir: ProjectPath
    output_name: str = Field(pattern=r"^[A-Za-z0-9_\-]+$")
    precision: WeightPrecision = WeightPrecision.FP16


class CoremlVerifyConfig(StrictModel):
    pytorch_weights: ProjectPath
    images_dir: ProjectPath
    max_images: int = Field(gt=0)
    conf: float = Field(ge=0.0, le=1.0)
    iou: float = Field(ge=0.0, le=1.0)
    match_iou_threshold: float = Field(gt=0.0, le=1.0)
    output_json: ProjectPath


# ----------------------------------------------------------------------------- root


class DashcamConfig(StrictModel):
    project: ProjectConfig
    classes: tuple[ClassSpec, ...] = Field(min_length=1)
    bdd100k: Bdd100kConfig
    frame_extraction: FrameExtractionConfig
    split: SplitConfig
    train: TrainConfig
    evaluate: EvaluateConfig
    twinlitenet: TwinLiteNetConfig
    coreml_export: CoremlExportConfig
    twinlitenet_export: TwinLiteNetExportConfig
    coreml_verify: CoremlVerifyConfig
