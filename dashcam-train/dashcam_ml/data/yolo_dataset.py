"""Helpers for the Ultralytics YOLO dataset layout (``images/<split>`` + ``labels/<split>``)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from dashcam_ml.utils.files import read_yaml, write_yaml

IMAGES_DIR_NAME: str = "images"
LABELS_DIR_NAME: str = "labels"
LABEL_EXTENSION: str = ".txt"


@dataclass(frozen=True, slots=True)
class YoloLabel:
    class_id: int
    center_x: float
    center_y: float
    width: float
    height: float

    def to_line(self) -> str:
        return f"{self.class_id} {self.center_x:.6f} {self.center_y:.6f} {self.width:.6f} {self.height:.6f}"

    @staticmethod
    def from_line(*, line: str) -> YoloLabel:
        parts: list[str] = line.split()
        if len(parts) != 5:
            raise ValueError(f"expected 5 values in YOLO label line, got: {line!r}")
        return YoloLabel(
            class_id=int(parts[0]),
            center_x=float(parts[1]),
            center_y=float(parts[2]),
            width=float(parts[3]),
            height=float(parts[4]),
        )


@dataclass(frozen=True, slots=True)
class YoloDatasetInfo:
    root: Path
    splits: dict[str, Path]
    names: dict[int, str]


def label_path_for_image(*, image_path: Path) -> Path:
    """Ultralytics convention: replace the last ``images`` folder with ``labels``."""
    parts: list[str] = list(image_path.with_suffix(LABEL_EXTENSION).parts)
    for index in range(len(parts) - 1, -1, -1):
        if parts[index] == IMAGES_DIR_NAME:
            parts[index] = LABELS_DIR_NAME
            return Path(*parts)
    raise ValueError(f"image path has no '{IMAGES_DIR_NAME}' folder: {image_path}")


def read_labels(*, label_path: Path) -> list[YoloLabel]:
    if not label_path.is_file():
        return []
    lines: list[str] = label_path.read_text(encoding="utf-8").splitlines()
    return [YoloLabel.from_line(line=line) for line in lines if line.strip()]


def write_labels(*, label_path: Path, labels: Sequence[YoloLabel]) -> None:
    label_path.parent.mkdir(parents=True, exist_ok=True)
    label_path.write_text(data="".join(f"{label.to_line()}\n" for label in labels), encoding="utf-8")


def normalize_names(*, names: object) -> dict[int, str]:
    if isinstance(names, Mapping):
        return {int(key): str(value) for key, value in names.items()}
    if isinstance(names, Sequence) and not isinstance(names, str):
        return {index: str(value) for index, value in enumerate(names)}
    raise ValueError(f"'names' in dataset YAML must be a list or mapping, got {type(names).__name__}")


def write_dataset_yaml(*, yaml_path: Path, root: Path, splits: Mapping[str, str], names: Mapping[int, str]) -> None:
    """Always writes an absolute ``path`` so Ultralytics never guesses the dataset root."""
    payload: dict[str, object] = {"path": root.resolve().as_posix(), **dict(splits), "names": dict(names)}
    write_yaml(path=yaml_path, payload=payload)


def read_dataset_yaml(*, yaml_path: Path) -> YoloDatasetInfo:
    data: dict[str, object] = read_yaml(path=yaml_path)
    raw_root: Path = Path(str(data.get("path", yaml_path.parent)))
    root: Path = raw_root if raw_root.is_absolute() else (yaml_path.parent / raw_root).resolve()
    splits: dict[str, Path] = {}
    for key, value in data.items():
        if key in {"path", "names", "nc", "download"} or not isinstance(value, str):
            continue
        split_path: Path = Path(value)
        splits[key] = split_path if split_path.is_absolute() else root / split_path
    return YoloDatasetInfo(root=root, splits=splits, names=normalize_names(names=data["names"]))
