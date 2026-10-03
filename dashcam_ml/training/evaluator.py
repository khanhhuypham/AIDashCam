"""CV10 / CV11 - evaluate several weights x image sizes on the same test split."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

from dashcam_ml.config.schema import EvaluateConfig, ImageSize, ProjectConfig
from dashcam_ml.core.class_catalog import ClassCatalog
from dashcam_ml.data.class_remap import prepare_dataset_for_model
from dashcam_ml.models.yolo_detector import to_ultralytics_imgsz
from dashcam_ml.utils.files import write_json
from dashcam_ml.utils.logging_utils import get_logger
from dashcam_ml.utils.runtime import resolve_device

LOGGER = get_logger(name=__name__)
EVAL_SUBDIR: str = "eval"
REPORT_NAME: str = "evaluation_report.json"


@dataclass(frozen=True, slots=True)
class ClassMetrics:
    class_name: str
    precision: float
    recall: float
    map50: float
    map50_95: float


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    weights: str
    imgsz: ImageSize
    precision: float
    recall: float
    map50: float
    map50_95: float
    inference_ms_per_image: float
    per_class: list[ClassMetrics]


def evaluate_detectors(*, config: EvaluateConfig, project: ProjectConfig, catalog: ClassCatalog) -> Path:
    from ultralytics import YOLO

    eval_dir: Path = project.output_dir / EVAL_SUBDIR
    results: list[EvaluationResult] = []
    for weights in config.weights:
        model: YOLO = YOLO(model=str(weights))
        model_names: dict[int, str] = {int(index): str(name) for index, name in model.names.items()}
        data_yaml: Path = prepare_dataset_for_model(
            dataset_yaml=config.data_yaml,
            split=config.split,
            model_names=model_names,
            catalog=catalog,
            work_dir=eval_dir / "datasets",
            copy_mode=config.copy_mode,
        )
        for imgsz in config.imgsz_list:
            LOGGER.info("evaluating %s @ %s on split '%s'", weights.name, imgsz, config.split)
            metrics = model.val(
                data=str(data_yaml),
                split=config.split,
                imgsz=to_ultralytics_imgsz(imgsz=imgsz),
                conf=config.conf,
                iou=config.iou,
                batch=config.batch,
                device=resolve_device(device=project.device),
                project=str(eval_dir / "runs"),
                name=f"{weights.stem}_{imgsz}",
                exist_ok=True,
                plots=True,
                verbose=False,
            )
            results.append(
                _to_result(metrics=metrics, weights=weights, imgsz=imgsz, model_names=model_names)
            )

    report_path: Path = eval_dir / REPORT_NAME
    write_json(
        path=report_path,
        payload={"split": config.split, "data_yaml": str(config.data_yaml), "results": [asdict(r) for r in results]},
    )
    _log_table(results=results, focus_classes=config.focus_classes, catalog=catalog)
    LOGGER.info("report written to %s", report_path)
    return report_path


def _to_result(*, metrics: object, weights: Path, imgsz: ImageSize, model_names: dict[int, str]) -> EvaluationResult:
    box = metrics.box  # type: ignore[attr-defined]
    per_class: list[ClassMetrics] = []
    for position, class_index in enumerate(box.ap_class_index):
        precision, recall, ap50, ap = box.class_result(i=position)
        per_class.append(
            ClassMetrics(
                class_name=model_names[int(class_index)],
                precision=float(precision),
                recall=float(recall),
                map50=float(ap50),
                map50_95=float(ap),
            )
        )
    speed: dict[str, float] = metrics.speed  # type: ignore[attr-defined]
    return EvaluationResult(
        weights=str(weights),
        imgsz=imgsz,
        precision=float(box.mp),
        recall=float(box.mr),
        map50=float(box.map50),
        map50_95=float(box.map),
        inference_ms_per_image=float(speed.get("inference", 0.0)),
        per_class=per_class,
    )


def _log_table(*, results: list[EvaluationResult], focus_classes: tuple[str, ...], catalog: ClassCatalog) -> None:
    LOGGER.info("%-28s %-10s %7s %7s %7s %9s", "weights", "imgsz", "P", "R", "mAP50", "mAP50-95")
    for result in results:
        LOGGER.info(
            "%-28s %-10s %7.3f %7.3f %7.3f %9.3f",
            Path(result.weights).name,
            str(result.imgsz),
            result.precision,
            result.recall,
            result.map50,
            result.map50_95,
        )
        for focus in focus_classes:
            wanted: set[str] = catalog.equivalent_names(class_name=focus)
            for class_metrics in result.per_class:
                if class_metrics.class_name.lower() in wanted:
                    LOGGER.info(
                        "    └ %-22s P=%.3f R=%.3f mAP50=%.3f",
                        class_metrics.class_name,
                        class_metrics.precision,
                        class_metrics.recall,
                        class_metrics.map50,
                    )
