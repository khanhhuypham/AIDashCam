"""One handler per :class:`Command`. Heavy libraries are imported inside handlers so
``--help`` and the pure-logic commands start instantly."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from dashcam_ml.config.schema import DashcamConfig
from dashcam_ml.core.class_catalog import ClassCatalog
from dashcam_ml.domain.enums import Command
from dashcam_ml.utils.logging_utils import get_logger

LOGGER = get_logger(name=__name__)


class CommandHandler(Protocol):
    def __call__(self, *, config: DashcamConfig) -> None: ...


def prepare_bdd100k(*, config: DashcamConfig) -> None:
    from dashcam_ml.data.bdd100k import convert_bdd100k

    convert_bdd100k(config=config.bdd100k)


def extract_frames(*, config: DashcamConfig) -> None:
    from dashcam_ml.data.frame_extraction import extract_frames as run_extraction

    run_extraction(config=config.frame_extraction)


def split_dataset(*, config: DashcamConfig) -> None:
    from dashcam_ml.data.split import split_dataset as run_split

    run_split(config=config.split, seed=config.project.seed)


def train(*, config: DashcamConfig) -> None:
    from dashcam_ml.training.trainer import train_detector

    train_detector(config=config.train, project=config.project)


def evaluate(*, config: DashcamConfig) -> None:
    from dashcam_ml.training.evaluator import evaluate_detectors

    evaluate_detectors(config=config.evaluate, project=config.project, catalog=ClassCatalog(specs=config.classes))


def run_video(*, config: DashcamConfig) -> None:
    from dashcam_ml.pipeline.video_runner import run_video as run_pipeline

    run_pipeline(config=config)


def export_coreml(*, config: DashcamConfig) -> None:
    from dashcam_ml.export.yolo_coreml import export_yolo_coreml

    export_yolo_coreml(config=config.coreml_export)


def export_twinlitenet(*, config: DashcamConfig) -> None:
    from dashcam_ml.export.twinlitenet_coreml import export_twinlitenet_coreml

    export_twinlitenet_coreml(model_config=config.twinlitenet, export_config=config.twinlitenet_export)


def verify_coreml(*, config: DashcamConfig) -> None:
    from dashcam_ml.export.verify import verify_coreml_models

    verify_coreml_models(
        verify_config=config.coreml_verify,
        models_dir=config.coreml_export.output_dir,
        twinlitenet_config=config.twinlitenet,
    )


def export_ios_config(*, config: DashcamConfig) -> None:
    from dashcam_ml.export.ios_config import export_ios_config as write_ios_config

    output_path: Path = write_ios_config(config=config)
    LOGGER.info("copy %s into the Xcode project", output_path.name)


COMMAND_HANDLERS: dict[Command, CommandHandler] = {
    Command.PREPARE_BDD100K: prepare_bdd100k,
    Command.EXTRACT_FRAMES: extract_frames,
    Command.SPLIT_DATASET: split_dataset,
    Command.TRAIN: train,
    Command.EVALUATE: evaluate,
    Command.RUN_VIDEO: run_video,
    Command.EXPORT_COREML: export_coreml,
    Command.EXPORT_TWINLITENET: export_twinlitenet,
    Command.VERIFY_COREML: verify_coreml,
    Command.EXPORT_IOS_CONFIG: export_ios_config,
}

COMMAND_HELP: dict[Command, str] = {
    Command.PREPARE_BDD100K: "CV02: chuyển nhãn BDD100K sang định dạng YOLO",
    Command.EXTRACT_FRAMES: "CV08: trích khung hình từ video tự quay",
    Command.SPLIT_DATASET: "CV09: chia train/val/test theo từng video",
    Command.TRAIN: "CV12: fine-tune YOLO",
    Command.EVALUATE: "CV11: đánh giá nhiều trọng số x kích thước ảnh",
    Command.RUN_VIDEO: "CV04/CV13: chạy detect + tracking + FCW (+ LDW) trên video",
    Command.EXPORT_COREML: "CV14: xuất YOLO sang Core ML theo các profile",
    Command.EXPORT_TWINLITENET: "CV14: xuất TwinLiteNet sang Core ML",
    Command.VERIFY_COREML: "CV14: so sánh Core ML với PyTorch (chỉ chạy trên macOS)",
    Command.EXPORT_IOS_CONFIG: "App iOS (sau pipeline): xuất ngưỡng cảnh báo ra JSON cho app Swift",
}
