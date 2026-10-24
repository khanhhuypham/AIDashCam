"""Remap a dataset's class IDs into a model's class-ID space.

Needed to evaluate COCO-pretrained weights (``person``=0, ``car``=2, ...) on a dataset
labelled with BDD100K names (``pedestrian``=0, ``car``=2, ...) fairly (CV11).
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from pathlib import Path

from dashcam_ml.core.class_catalog import ClassCatalog
from dashcam_ml.data.yolo_dataset import (
    IMAGES_DIR_NAME,
    LABEL_EXTENSION,
    LABELS_DIR_NAME,
    YoloDatasetInfo,
    YoloLabel,
    label_path_for_image,
    read_dataset_yaml,
    read_labels,
    write_dataset_yaml,
    write_labels,
)
from dashcam_ml.domain.enums import CopyMode, DatasetSplit
from dashcam_ml.utils.files import iter_images, materialize_file
from dashcam_ml.utils.logging_utils import get_logger

LOGGER = get_logger(name=__name__)


def build_class_mapping(
    *, source_names: Mapping[int, str], target_names: Mapping[int, str], catalog: ClassCatalog
) -> dict[int, int]:
    """``source_id -> target_id`` for every source class that exists in the target model."""
    mapping: dict[int, int] = {}
    for source_id, source_name in source_names.items():
        target_id: int | None = catalog.match_index(class_name=source_name, candidates=target_names.items())
        if target_id is not None:
            mapping[source_id] = target_id
    return mapping


def prepare_dataset_for_model(
    *,
    dataset_yaml: Path,
    split: str,
    model_names: Mapping[int, str],
    catalog: ClassCatalog,
    work_dir: Path,
    copy_mode: CopyMode,
) -> Path:
    """Return a dataset YAML whose labels use ``model_names`` IDs (original YAML if no change needed)."""
    info: YoloDatasetInfo = read_dataset_yaml(yaml_path=dataset_yaml)
    if split not in info.splits:
        raise KeyError(f"split '{split}' not found in {dataset_yaml}; available: {sorted(info.splits)}")
    mapping: dict[int, int] = build_class_mapping(source_names=info.names, target_names=model_names, catalog=catalog)
    is_identity: bool = all(mapping.get(class_id) == class_id for class_id in info.names) and all(
        info.names[class_id].lower() == model_names.get(class_id, "").lower() for class_id in info.names
    )
    if is_identity:
        return dataset_yaml

    unmapped: list[str] = [name for class_id, name in info.names.items() if class_id not in mapping]
    if unmapped:
        LOGGER.info("classes not present in the model are dropped for this evaluation: %s", unmapped)

    fingerprint: str = hashlib.sha1(
        string=repr(sorted(mapping.items())).encode(encoding="utf-8"), usedforsecurity=False
    ).hexdigest()[:8]
    output_root: Path = work_dir / f"{dataset_yaml.parent.name}_{split}_{fingerprint}"
    split_images: Path = info.splits[split]
    images: list[Path] = (
        list(iter_images(directory=split_images))
        if split_images.is_dir()
        else [Path(line.strip()) for line in split_images.read_text(encoding="utf-8").splitlines() if line.strip()]
    )
    for image_path in images:
        destination_image: Path = output_root / IMAGES_DIR_NAME / DatasetSplit.TEST.value / image_path.name
        materialize_file(source=image_path, destination=destination_image, mode=copy_mode)
        remapped: list[YoloLabel] = [
            YoloLabel(
                class_id=mapping[label.class_id],
                center_x=label.center_x,
                center_y=label.center_y,
                width=label.width,
                height=label.height,
            )
            for label in read_labels(label_path=label_path_for_image(image_path=image_path))
            if label.class_id in mapping
        ]
        write_labels(
            label_path=output_root / LABELS_DIR_NAME / DatasetSplit.TEST.value / f"{image_path.stem}{LABEL_EXTENSION}",
            labels=remapped,
        )

    remapped_yaml: Path = output_root / "dataset.yaml"
    split_dir: str = f"{IMAGES_DIR_NAME}/{DatasetSplit.TEST.value}"
    write_dataset_yaml(
        yaml_path=remapped_yaml,
        root=output_root,
        splits={DatasetSplit.TRAIN.value: split_dir, DatasetSplit.VAL.value: split_dir, split: split_dir},
        names=model_names,
    )
    LOGGER.info("remapped %d images of split '%s' into %s", len(images), split, output_root)
    return remapped_yaml
