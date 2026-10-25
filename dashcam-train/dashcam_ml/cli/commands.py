"""One handler per :class:`Command`. Heavy libraries are imported inside handlers so
``--help`` and the pure-logic commands start instantly."""

from __future__ import annotations

from typing import Protocol

from dashcam_ml.config.schema import DashcamConfig
from dashcam_ml.core.class_catalog import ClassCatalog
from dashcam_ml.domain.enums import Command


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


def export_tflite(*, config: DashcamConfig) -> None:
    from dashcam_ml.export.yolo_tflite import export_yolo_tflite

    export_yolo_tflite(config=config.tflite_export)


def verify_tflite(*, config: DashcamConfig) -> None:
    from dashcam_ml.export.verify import verify_tflite_models

    verify_tflite_models(verify_config=config.tflite_verify, models_dir=config.tflite_export.output_dir)


COMMAND_HANDLERS: dict[Command, CommandHandler] = {
    Command.PREPARE_BDD100K: prepare_bdd100k,
    Command.EXTRACT_FRAMES: extract_frames,
    Command.SPLIT_DATASET: split_dataset,
    Command.TRAIN: train,
    Command.EVALUATE: evaluate,
    Command.EXPORT_COREML: export_coreml,
    Command.EXPORT_TWINLITENET: export_twinlitenet,
    Command.VERIFY_COREML: verify_coreml,
    Command.EXPORT_TFLITE: export_tflite,
    Command.VERIFY_TFLITE: verify_tflite,
}

COMMAND_HELP: dict[Command, str] = {
    Command.PREPARE_BDD100K: "CV02: chuyển nhãn BDD100K sang định dạng YOLO",
    Command.EXTRACT_FRAMES: "CV08: trích khung hình từ video tự quay",
    Command.SPLIT_DATASET: "CV09: chia train/val/test theo từng video",
    Command.TRAIN: "CV12: fine-tune YOLO",
    Command.EVALUATE: "CV11: đánh giá nhiều trọng số x kích thước ảnh",
    Command.EXPORT_COREML: "CV14: xuất YOLO sang Core ML (iOS) theo các profile",
    Command.EXPORT_TWINLITENET: "CV14: xuất TwinLiteNet sang Core ML (iOS)",
    Command.VERIFY_COREML: "CV14: so sánh Core ML với PyTorch (chỉ chạy trên macOS)",
    Command.EXPORT_TFLITE: "xuất YOLO sang TFLite (Android) theo các profile (Linux/macOS, ví dụ Colab)",
    Command.VERIFY_TFLITE: "so sánh TFLite với PyTorch (Linux/macOS, ví dụ Colab)",
}
