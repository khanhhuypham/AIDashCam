"""CV02 - convert BDD100K detection labels to a YOLO dataset.

Input is the Dataset Ninja export (Supervisely format): ``<root>/<split>/img/*.jpg`` plus one
``<root>/<split>/ann/<image>.json`` per image. Each annotation is read into a :class:`BddFrame`,
then boxes are mapped to class ids and written as YOLO labels with condition splits and ``dataset.yaml``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from tqdm import tqdm

from dashcam_ml.config.schema import Bdd100kConfig
from dashcam_ml.data.yolo_dataset import (
    IMAGES_DIR_NAME,
    LABEL_EXTENSION,
    LABELS_DIR_NAME,
    YoloLabel,
    write_dataset_yaml,
    write_labels,
)
from dashcam_ml.domain.enums import DatasetSplit, SuperviselyGeometry
from dashcam_ml.utils.files import materialize_file
from dashcam_ml.utils.logging_utils import get_logger

LOGGER = get_logger(name=__name__)
DATASET_YAML_NAME: str = "dataset.yaml"
CONVERTED_SPLITS: tuple[DatasetSplit, ...] = (DatasetSplit.TRAIN, DatasetSplit.VAL)
SUPERVISELY_IMAGES_DIR_NAME: str = "img"
SUPERVISELY_ANNOTATIONS_DIR_NAME: str = "ann"
SUPERVISELY_ANNOTATION_SUFFIX: str = ".json"


@dataclass(frozen=True, slots=True)
class BddBox:
    """Pixel box in the source image, ``(x1, y1)`` top-left and ``(x2, y2)`` bottom-right."""

    category: str
    x1: float
    y1: float
    x2: float
    y2: float


@dataclass(frozen=True, slots=True)
class BddFrame:
    """One labelled image read from a Supervisely annotation file."""

    name: str
    source_image: Path
    image_width: int
    image_height: int
    boxes: tuple[BddBox, ...]
    attributes: dict[str, str] = field(default_factory=dict)
    """Scene attributes used for condition splits, e.g. ``{"timeofday": "night", "weather": "rainy"}``."""


@dataclass(frozen=True, slots=True)
class SplitConversionStatistics:
    split: DatasetSplit
    images: int
    boxes: int
    skipped_missing_images: int
    skipped_unknown_categories: int


def convert_bdd100k(*, config: Bdd100kConfig) -> Path:
    condition_lists: dict[str, list[str]] = {name: [] for name in config.condition_splits}
    for split in CONVERTED_SPLITS:
        frames: list[BddFrame] = read_supervisely_frames(
            split_dir=config.dataset_root / split.value,
            max_frames=config.max_images_per_split
        )
        stats: SplitConversionStatistics
        split_condition_lists: dict[str, list[str]]
        stats, split_condition_lists = _convert_split(config=config, split=split, frames=frames)
        for condition_name, image_paths in split_condition_lists.items():
            condition_lists[condition_name].extend(image_paths)
        LOGGER.info(
            "%s: %d images, %d boxes, %d missing images, %d unknown-category boxes",
            stats.split.value,
            stats.images,
            stats.boxes,
            stats.skipped_missing_images,
            stats.skipped_unknown_categories,
        )

    splits: dict[str, str] = {split.value: f"{IMAGES_DIR_NAME}/{split.value}" for split in CONVERTED_SPLITS}
    for condition_name, image_paths in condition_lists.items():
        list_path: Path = config.output_dir / f"{DatasetSplit.VAL.value}_{condition_name}.txt"
        list_path.write_text(data="".join(f"{path}\n" for path in image_paths), encoding="utf-8")
        splits[f"{DatasetSplit.VAL.value}_{condition_name}"] = list_path.resolve().as_posix()
        LOGGER.info("condition split val_%s: %d images", condition_name, len(image_paths))

    yaml_path: Path = config.output_dir / DATASET_YAML_NAME
    write_dataset_yaml(
        yaml_path=yaml_path,
        root=config.output_dir,
        splits=splits,
        names={index: name for index, name in enumerate(config.class_names)},
    )
    LOGGER.info("dataset YAML written to %s", yaml_path)
    return yaml_path


def read_supervisely_frames(*, split_dir: Path, max_frames: int | None = None) -> list[BddFrame]:
    """Image-level tags (``timeofday``, ``weather``, ``scene``) become ``attributes``; only rectangles become boxes.

    ``max_frames`` limits how many annotation files are opened (sorted by name), so a quick
    run on the full 70k-image train split does not have to parse every JSON first.
    """
    annotations_dir: Path = split_dir / SUPERVISELY_ANNOTATIONS_DIR_NAME
    images_dir: Path = split_dir / SUPERVISELY_IMAGES_DIR_NAME
    for required_dir in (annotations_dir, images_dir):
        if not required_dir.is_dir():
            raise FileNotFoundError(f"Supervisely folder not found: {required_dir}")
    annotation_paths: list[Path] = sorted(annotations_dir.glob(pattern=f"*{SUPERVISELY_ANNOTATION_SUFFIX}"))
    frames: list[BddFrame] = []
    for annotation_path in annotation_paths[:max_frames]:
        with annotation_path.open(mode="r", encoding="utf-8") as handle:
            annotation: dict[str, object] = json.load(fp=handle)
        name: str = annotation_path.name[: -len(SUPERVISELY_ANNOTATION_SUFFIX)]
        size: dict[str, object] = annotation["size"]  # type: ignore[assignment]
        boxes: list[BddBox] = []
        for raw_object in annotation.get("objects") or []:  # type: ignore[union-attr]
            obj: dict[str, object] = raw_object  # type: ignore[assignment]
            # Compared, not parsed with SuperviselyGeometry(...): an unlisted geometry type is skipped, not fatal.
            if obj.get("geometryType") != SuperviselyGeometry.RECTANGLE:
                continue
            (first_x, first_y), (second_x, second_y) = obj["points"]["exterior"]  # type: ignore[index]
            boxes.append(
                BddBox(
                    category=str(obj.get("classTitle", "")),
                    x1=float(min(first_x, second_x)),
                    y1=float(min(first_y, second_y)),
                    x2=float(max(first_x, second_x)),
                    y2=float(max(first_y, second_y)),
                )
            )
        tags: list[dict[str, object]] = annotation.get("tags") or []  # type: ignore[assignment]
        frames.append(
            BddFrame(
                name=name,
                source_image=images_dir / name,
                image_width=int(str(size["width"])),
                image_height=int(str(size["height"])),
                boxes=tuple(boxes),
                attributes={str(tag["name"]): str(tag.get("value")) for tag in tags},
            )
        )
    return frames


def _convert_split(
    *,
    config: Bdd100kConfig,
    split: DatasetSplit,
    frames: list[BddFrame],
) -> tuple[SplitConversionStatistics, dict[str, list[str]]]:
    """Write YOLO images/labels for one split.

    Also returns, per condition split, the converted image paths matching it (only filled for ``VAL``).
    """
    class_ids: dict[str, int] = {name: index for index, name in enumerate(config.class_names)}
    condition_lists: dict[str, list[str]] = {name: [] for name in config.condition_splits}
    images_out: Path = config.output_dir / IMAGES_DIR_NAME / split.value
    labels_out: Path = config.output_dir / LABELS_DIR_NAME / split.value
    image_count: int = 0
    box_count: int = 0
    missing_images: int = 0
    unknown_categories: int = 0

    for frame in tqdm(iterable=frames, desc=f"BDD100K {split.value}"):
        if not frame.source_image.is_file():
            missing_images += 1
            continue
        labels: list[YoloLabel] = []
        for box in frame.boxes:
            category: str = config.category_aliases.get(box.category, box.category)
            if category not in class_ids:
                unknown_categories += 1
                continue
            labels.append(
                box_to_yolo(
                    class_id=class_ids[category],
                    box=box,
                    image_width=frame.image_width,
                    image_height=frame.image_height
                )
            )
        destination_image: Path = images_out / frame.name
        materialize_file(source=frame.source_image, destination=destination_image, mode=config.copy_mode)
        write_labels(label_path=labels_out / Path(frame.name).with_suffix(LABEL_EXTENSION).name, labels=labels)
        image_count += 1
        box_count += len(labels)

        if split is DatasetSplit.VAL:
            for condition_name, required in config.condition_splits.items():
                if all(frame.attributes.get(key) == value for key, value in required.items()):
                    condition_lists[condition_name].append(destination_image.resolve().as_posix())

    stats: SplitConversionStatistics = SplitConversionStatistics(
        split=split,
        images=image_count,
        boxes=box_count,
        skipped_missing_images=missing_images,
        skipped_unknown_categories=unknown_categories,
    )
    return stats, condition_lists


def box_to_yolo(*, class_id: int, box: BddBox, image_width: int, image_height: int) -> YoloLabel:
    x1: float = max(0.0, min(box.x1, image_width))
    y1: float = max(0.0, min(box.y1, image_height))
    x2: float = max(0.0, min(box.x2, image_width))
    y2: float = max(0.0, min(box.y2, image_height))
    return YoloLabel(
        class_id=class_id,
        center_x=(x1 + x2) / 2.0 / image_width,
        center_y=(y1 + y2) / 2.0 / image_height,
        width=(x2 - x1) / image_width,
        height=(y2 - y1) / image_height,
    )
