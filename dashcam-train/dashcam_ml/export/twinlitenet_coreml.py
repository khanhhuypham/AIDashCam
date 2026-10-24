"""CV14 (phương án B) - convert TwinLiteNet to Core ML with coremltools (target iOS 16)."""

from __future__ import annotations

from pathlib import Path

from dashcam_ml.config.schema import TwinLiteNetConfig, TwinLiteNetExportConfig
from dashcam_ml.domain.enums import ModelKind, WeightPrecision
from dashcam_ml.export.manifest import ModelManifestEntry, upsert_manifest
from dashcam_ml.export.yolo_coreml import MLPACKAGE_SUFFIX
from dashcam_ml.utils.files import remove_path
from dashcam_ml.utils.logging_utils import get_logger
from dashcam_ml.utils.runtime import require_coreml_conversion_platform

LOGGER = get_logger(name=__name__)
INPUT_NAME: str = "image"
DRIVABLE_OUTPUT_NAME: str = "drivable_prob"
LANE_OUTPUT_NAME: str = "lane_prob"
PIXEL_SCALE: float = 1.0 / 255.0


def export_twinlitenet_coreml(*, model_config: TwinLiteNetConfig, export_config: TwinLiteNetExportConfig) -> Path:
    require_coreml_conversion_platform()
    import coremltools as ct
    import torch

    from dashcam_ml.models.twinlitenet import TwinLiteNetProbabilityHead, load_twinlitenet

    head: TwinLiteNetProbabilityHead = TwinLiteNetProbabilityHead(backbone=load_twinlitenet(config=model_config)).eval()
    example: torch.Tensor = torch.rand(1, 3, model_config.input_height, model_config.input_width)
    with torch.no_grad():
        traced: torch.jit.ScriptModule = torch.jit.trace(func=head, example_inputs=example)

    mlmodel = ct.convert(
        model=traced,
        convert_to="mlprogram",
        inputs=[
            ct.ImageType(
                name=INPUT_NAME, shape=tuple(example.shape), scale=PIXEL_SCALE, color_layout=ct.colorlayout.RGB
            )
        ],
        outputs=[ct.TensorType(name=DRIVABLE_OUTPUT_NAME), ct.TensorType(name=LANE_OUTPUT_NAME)],
        minimum_deployment_target=ct.target.iOS16,
        compute_precision=(
            ct.precision.FLOAT32 if export_config.precision is WeightPrecision.FP32 else ct.precision.FLOAT16
        ),
    )
    if export_config.precision is WeightPrecision.INT8:
        from coremltools.optimize.coreml import OpLinearQuantizerConfig, OptimizationConfig, linear_quantize_weights

        mlmodel = linear_quantize_weights(
            mlmodel=mlmodel,
            config=OptimizationConfig(global_config=OpLinearQuantizerConfig(mode="linear_symmetric")),
        )

    mlmodel.short_description = "TwinLiteNet: drivable area + lane line probability maps (1 x H x W)."
    mlmodel.user_defined_metadata["mask_threshold"] = str(model_config.mask_threshold)
    export_config.output_dir.mkdir(parents=True, exist_ok=True)
    destination: Path = export_config.output_dir / f"{export_config.output_name}{MLPACKAGE_SUFFIX}"
    remove_path(path=destination)
    mlmodel.save(save_path=str(destination))
    LOGGER.info("saved %s", destination)

    entry: ModelManifestEntry = ModelManifestEntry(
        name=export_config.output_name,
        kind=ModelKind.LANE_SEGMENTER,
        file_name=destination.name,
        input_width=model_config.input_width,
        input_height=model_config.input_height,
        precision=export_config.precision,
        source_weights=model_config.weights.name,
        target_devices="iPhone 8 trở lên",
        input_names=[INPUT_NAME],
        output_names=[DRIVABLE_OUTPUT_NAME, LANE_OUTPUT_NAME],
    )
    return upsert_manifest(output_dir=export_config.output_dir, entries=[entry])
