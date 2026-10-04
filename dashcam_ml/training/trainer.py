"""CV12 - fine-tune a YOLO detector with Ultralytics."""

from __future__ import annotations

from pathlib import Path

from dashcam_ml.config.schema import ProjectConfig, TrainConfig
from dashcam_ml.utils.logging_utils import get_logger
from dashcam_ml.utils.runtime import resolve_device

LOGGER = get_logger(name=__name__)
TRAIN_SUBDIR: str = "train"
RESUME_WEIGHTS_NAME: str = "last.pt"

def train_detector(*, config: TrainConfig, project: ProjectConfig) -> Path:
    from ultralytics import YOLO

    if config.resume and config.model.name != RESUME_WEIGHTS_NAME:
        # resume từ trọng số gốc (vd. yolo11n.pt) không có gì để train tiếp
        raise ValueError(f"train.resume = true cần train.model trỏ tới {RESUME_WEIGHTS_NAME} của lần train trước, đang là {config.model}")
    model: YOLO = YOLO(model=str(config.model))
    model.train(
        data=str(config.data_yaml),
        epochs=config.epochs,
        imgsz=config.imgsz,
        batch=config.batch,
        workers=config.workers,
        patience=config.patience,
        optimizer=config.optimizer,
        lr0=config.lr0,
        cos_lr=config.cos_lr,
        amp=config.amp,
        cache=config.cache,
        resume=config.resume,
        seed=project.seed,
        deterministic=True,
        device=resolve_device(device=project.device),
        project=str(project.output_dir / TRAIN_SUBDIR),
        name=config.run_name,
        exist_ok=True,
        plots=True,
    )
    best_weights: Path = Path(str(model.trainer.best))
    LOGGER.info("best weights: %s", best_weights)
    return best_weights
