from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from dashcam_ml.config.schema import Bdd100kConfig, DashcamConfig
from dashcam_ml.core.class_catalog import ClassCatalog
from dashcam_ml.data.bdd100k import BddBox, BddFrame, box_to_yolo, convert_bdd100k, read_supervisely_frames
from dashcam_ml.data.class_remap import build_class_mapping
from dashcam_ml.data.split import assign_groups, group_key
from dashcam_ml.data.yolo_dataset import YoloLabel, label_path_for_image, read_dataset_yaml, read_labels
from dashcam_ml.domain.enums import CopyMode, DatasetSplit, SuperviselyGeometry

COCO_SUBSET: dict[int, str] = {0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
BDD_NAMES: dict[int, str] = {0: "pedestrian", 1: "rider", 2: "car", 3: "truck", 4: "bus", 6: "motorcycle", 8: "traffic light"}


def test_group_key_uses_video_prefix() -> None:
    assert group_key(image_path=Path("trip_01__f000120.jpg"), separator="__") == "trip_01"
    assert group_key(image_path=Path("single.jpg"), separator="__") == "single"


def test_groups_never_split_and_ratios_are_close() -> None:
    sizes: dict[str, int] = {f"video_{index}": 100 for index in range(20)}
    assignment: dict[str, DatasetSplit] = assign_groups(
        group_sizes=sizes, ratios={DatasetSplit.TRAIN: 0.7, DatasetSplit.VAL: 0.15, DatasetSplit.TEST: 0.15}, seed=42
    )
    assert set(assignment) == set(sizes)
    counts: dict[DatasetSplit, int] = {split: 0 for split in DatasetSplit}
    for split in assignment.values():
        counts[split] += 1
    assert counts[DatasetSplit.TRAIN] == 14


def test_bdd_to_coco_mapping_uses_aliases(config: DashcamConfig) -> None:
    mapping: dict[int, int] = build_class_mapping(
        source_names=BDD_NAMES, target_names=COCO_SUBSET, catalog=ClassCatalog(specs=config.classes)
    )
    assert mapping == {0: 0, 2: 2, 3: 7, 4: 5, 6: 3}


def test_label_path_replaces_last_images_folder() -> None:
    image: Path = Path("/data/images/set/images/test/a.jpg")
    assert label_path_for_image(image_path=image) == Path("/data/images/set/labels/test/a.txt")


def _write_supervisely_annotation(*, split_dir: Path, image_name: str, annotation: dict[str, object]) -> None:
    (split_dir / "img").mkdir(parents=True, exist_ok=True)
    (split_dir / "ann").mkdir(parents=True, exist_ok=True)
    (split_dir / "img" / image_name).write_bytes(data=b"")
    (split_dir / "ann" / f"{image_name}.json").write_text(data=json.dumps(obj=annotation), encoding="utf-8")


def test_read_supervisely_frames_keeps_rectangles_and_tags(tmp_path: Path) -> None:
    _write_supervisely_annotation(
        split_dir=tmp_path / "val",
        image_name="a.jpg",
        annotation={
            "size": {"width": 1280, "height": 720},
            "tags": [{"name": "timeofday", "value": "night"}, {"name": "weather", "value": "rainy"}],
            "objects": [
                {"classTitle": "car", "geometryType": SuperviselyGeometry.RECTANGLE.value, "points": {"exterior": [[300, 200], [100, 50]]}},
                {"classTitle": "lane", "geometryType": SuperviselyGeometry.LINE.value, "points": {"exterior": [[0, 0], [10, 10], [20, 30]]}},
                {"classTitle": "car", "geometryType": "cuboid_3d", "points": {"exterior": []}},
            ],
        },
    )
    frames: list[BddFrame] = read_supervisely_frames(split_dir=tmp_path / "val")
    assert len(frames) == 1
    frame: BddFrame = frames[0]
    assert frame.name == "a.jpg"
    assert frame.source_image == tmp_path / "val" / "img" / "a.jpg"
    assert (frame.image_width, frame.image_height) == (1280, 720)
    assert frame.boxes == (BddBox(category="car", x1=100.0, y1=50.0, x2=300.0, y2=200.0),)
    assert frame.attributes == {"timeofday": "night", "weather": "rainy"}


def test_box_to_yolo_clamps_and_normalizes() -> None:
    label: YoloLabel = box_to_yolo(
        class_id=2, box=BddBox(category="car", x1=-10.0, y1=0.0, x2=640.0, y2=800.0), image_width=1280, image_height=720
    )
    assert label == YoloLabel(class_id=2, center_x=0.25, center_y=0.5, width=0.5, height=1.0)


def test_convert_supervisely_writes_labels_and_condition_splits(config: DashcamConfig, tmp_path: Path) -> None:
    root: Path = tmp_path / "sample"
    for split, timeofday in (("train", "daytime"), ("val", "night")):
        _write_supervisely_annotation(
            split_dir=root / split,
            image_name=f"{split}.jpg",
            annotation={
                "size": {"width": 1280, "height": 720},
                "tags": [{"name": "timeofday", "value": timeofday}],
                "objects": [
                    {"classTitle": "person", "geometryType": SuperviselyGeometry.RECTANGLE.value, "points": {"exterior": [[0, 0], [128, 72]]}},
                    {"classTitle": "drivable area", "geometryType": SuperviselyGeometry.RECTANGLE.value, "points": {"exterior": [[0, 0], [1, 1]]}},
                ],
            },
        )
    bdd_config: Bdd100kConfig = config.bdd100k.model_copy(
        update={
            "dataset_root": root,
            "output_dir": tmp_path / "yolo",
            "copy_mode": CopyMode.COPY,
            "max_images_per_split": None,
        }
    )
    yaml_path: Path = convert_bdd100k(config=bdd_config)

    pedestrian_id: int = bdd_config.class_names.index("pedestrian")
    labels: list[YoloLabel] = read_labels(label_path=tmp_path / "yolo" / "labels" / "val" / "val.txt")
    assert labels == [YoloLabel(class_id=pedestrian_id, center_x=0.05, center_y=0.05, width=0.1, height=0.1)]
    night_list: str = (tmp_path / "yolo" / "val_night.txt").read_text(encoding="utf-8")
    assert night_list.strip().endswith("images/val/val.jpg")
    assert (tmp_path / "yolo" / "val_daytime.txt").read_text(encoding="utf-8") == ""
    assert read_dataset_yaml(yaml_path=yaml_path).splits["train"] == (tmp_path / "yolo" / "images" / "train").resolve()


def test_bdd100k_config_requires_dataset_root(config: DashcamConfig) -> None:
    raw: dict[str, object] = config.bdd100k.model_dump()
    del raw["dataset_root"]
    with pytest.raises(ValidationError, match="dataset_root"):
        Bdd100kConfig.model_validate(obj=raw)


def test_read_supervisely_frames_fails_on_missing_folder(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="ann"):
        read_supervisely_frames(split_dir=tmp_path / "val")


def test_read_supervisely_frames_limits_files_in_name_order(tmp_path: Path) -> None:
    for image_name in ("c.jpg", "a.jpg", "b.jpg"):
        _write_supervisely_annotation(
            split_dir=tmp_path / "train",
            image_name=image_name,
            annotation={"size": {"width": 1280, "height": 720}, "tags": [], "objects": []},
        )
    frames: list[BddFrame] = read_supervisely_frames(split_dir=tmp_path / "train", max_frames=2)
    assert [frame.name for frame in frames] == ["a.jpg", "b.jpg"]
