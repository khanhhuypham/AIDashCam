"""Every fixed set of choices used across the project, defined as real enums.

Config values in ``config.yaml`` are parsed into these enums by pydantic, so a typo
(e.g. ``copy_mode: hardlnk``) fails at load time instead of deep inside a command.
"""

from __future__ import annotations

from enum import IntEnum, StrEnum


class LogLevel(StrEnum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"


class ClassCategory(StrEnum):
    """Road-user group used by the warning logic."""

    VEHICLE = "vehicle"
    TWO_WHEELER = "two_wheeler"
    PEDESTRIAN = "pedestrian"


class ReferenceDimension(StrEnum):
    """Which side of the bounding box is compared with the real-world size."""

    WIDTH = "width"
    HEIGHT = "height"


class EgoSpeedSource(StrEnum):
    CONSTANT = "constant"
    CSV = "csv"
    NONE = "none"


class CopyMode(StrEnum):
    """How dataset files are materialised into a new folder."""

    COPY = "copy"
    HARDLINK = "hardlink"


class SuperviselyGeometry(StrEnum):
    """``geometryType`` of an object in a Supervisely annotation file."""

    RECTANGLE = "rectangle"
    POLYGON = "polygon"
    LINE = "line"
    BITMAP = "bitmap"
    POINT = "point"
    GRAPH = "graph"
    CUBOID = "cuboid"
    ALPHA_MASK = "alpha_mask"


class DatasetSplit(StrEnum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"


class AlertLevel(IntEnum):
    """Ordered so that ``AlertLevel.CRITICAL > AlertLevel.CAUTION`` holds."""

    NONE = 0
    CAUTION = 1
    CRITICAL = 2


class LaneDepartureSide(StrEnum):
    NONE = "none"
    LEFT = "left"
    RIGHT = "right"


class ModelKind(StrEnum):
    """Role of an exported Core ML model inside the iOS app."""

    DETECTOR = "detector"
    LANE_SEGMENTER = "lane_segmenter"


class WeightPrecision(StrEnum):
    FP32 = "fp32"
    FP16 = "fp16"
    INT8 = "int8"


class Command(StrEnum):
    """CLI sub-commands (``python -m dashcam_ml <command>``)."""

    PREPARE_BDD100K = "prepare-bdd100k"
    EXTRACT_FRAMES = "extract-frames"
    SPLIT_DATASET = "split-dataset"
    TRAIN = "train"
    EVALUATE = "evaluate"
    RUN_VIDEO = "run-video"
    EXPORT_COREML = "export-coreml"
    EXPORT_TWINLITENET = "export-twinlitenet"
    VERIFY_COREML = "verify-coreml"
    EXPORT_IOS_CONFIG = "export-ios-config"
