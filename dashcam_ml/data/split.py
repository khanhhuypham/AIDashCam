"""CV09 - split a labelled dataset into train/val/test *by source video*.

Frames of the same video never end up in two splits, which would leak near-identical
images between training and testing.
"""

from __future__ import annotations

import random
from collections import defaultdict
from pathlib import Path

from dashcam_ml.config.schema import SplitConfig
from dashcam_ml.data.yolo_dataset import IMAGES_DIR_NAME, LABEL_EXTENSION, LABELS_DIR_NAME, write_dataset_yaml
from dashcam_ml.domain.enums import DatasetSplit
from dashcam_ml.utils.files import iter_images, materialize_file, write_json
from dashcam_ml.utils.logging_utils import get_logger

LOGGER = get_logger(name=__name__)
DATASET_YAML_NAME: str = "dataset.yaml"
SPLIT_REPORT_NAME: str = "split_report.json"


def group_key(*, image_path: Path, separator: str) -> str:
    """``<video>__f000123.jpg`` -> ``<video>``; files without the separator form their own group."""
    stem: str = image_path.stem
    return stem.split(sep=separator, maxsplit=1)[0] if separator in stem else stem


def assign_groups(
    *, group_sizes: dict[str, int], ratios: dict[DatasetSplit, float], seed: int
) -> dict[str, DatasetSplit]:
    """Greedy: shuffle groups, then give each one to the split furthest below its target."""
    groups: list[str] = sorted(group_sizes)
    random.Random(x=seed).shuffle(x=groups)
    total: int = sum(group_sizes.values())
    targets: dict[DatasetSplit, float] = {split: ratio * total for split, ratio in ratios.items()}
    filled: dict[DatasetSplit, int] = {split: 0 for split in ratios}
    assignment: dict[str, DatasetSplit] = {}
    for group in groups:
        chosen: DatasetSplit = max(ratios, key=lambda split: targets[split] - filled[split])
        assignment[group] = chosen
        filled[chosen] += group_sizes[group]
    return assignment


def split_dataset(*, config: SplitConfig, seed: int) -> Path:
    images: list[Path] = list(iter_images(directory=config.images_dir))
    if not images:
        raise FileNotFoundError(f"no images under {config.images_dir}")
    grouped: dict[str, list[Path]] = defaultdict(list)
    for image_path in images:
        grouped[group_key(image_path=image_path, separator=config.name_separator)].append(image_path)

    assignment: dict[str, DatasetSplit] = assign_groups(
        group_sizes={group: len(paths) for group, paths in grouped.items()},
        ratios=config.ratios.as_dict(),
        seed=seed,
    )
    counts: dict[DatasetSplit, int] = {split: 0 for split in DatasetSplit}
    missing_labels: int = 0
    for group, paths in grouped.items():
        split: DatasetSplit = assignment[group]
        for image_path in paths:
            relative: Path = image_path.relative_to(config.images_dir)
            materialize_file(
                source=image_path,
                destination=config.output_dir / IMAGES_DIR_NAME / split.value / relative,
                mode=config.copy_mode,
            )
            label_source: Path = (config.labels_dir / relative).with_suffix(LABEL_EXTENSION)
            if label_source.is_file():
                materialize_file(
                    source=label_source,
                    destination=(config.output_dir / LABELS_DIR_NAME / split.value / relative).with_suffix(
                        LABEL_EXTENSION
                    ),
                    mode=config.copy_mode,
                )
            else:
                missing_labels += 1
            counts[split] += 1

    yaml_path: Path = config.output_dir / DATASET_YAML_NAME
    write_dataset_yaml(
        yaml_path=yaml_path,
        root=config.output_dir,
        splits={split.value: f"{IMAGES_DIR_NAME}/{split.value}" for split in DatasetSplit},
        names={index: name for index, name in enumerate(config.class_names)},
    )
    write_json(
        path=config.output_dir / SPLIT_REPORT_NAME,
        payload={
            "seed": seed,
            "images_per_split": {split.value: count for split, count in counts.items()},
            "videos_per_split": {
                split.value: sorted(group for group, assigned in assignment.items() if assigned is split)
                for split in DatasetSplit
            },
            "images_without_label": missing_labels,
        },
    )
    for split, ratio in config.ratios.as_dict().items():
        if ratio > 0.0 and counts[split] == 0:
            LOGGER.warning("split '%s' is empty: record more videos (%d so far)", split.value, len(grouped))
    LOGGER.info(
        "split done: %s (%d images without label = background)",
        {split.value: count for split, count in counts.items()},
        missing_labels,
    )
    return yaml_path
