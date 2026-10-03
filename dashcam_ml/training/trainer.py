"""CV12 - fine-tune a YOLO detector with Ultralytics."""

from __future__ import annotations

from pathlib import Path

from dashcam_ml.config.schema import ProjectConfig, TrainConfig
from dashcam_ml.utils.logging_utils import get_logger
from dashcam_ml.utils.runtime import resolve_device

LOGGER = get_logger(name=__name__)
TRAIN_SUBDIR: str = "train"

def train_detector(*, config: TrainConfig, project: ProjectConfig) -> Path:
    from ultralytics import YOLO

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
