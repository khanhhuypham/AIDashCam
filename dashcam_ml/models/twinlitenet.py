"""TwinLiteNet loader, probability head and frame segmenter (phương án B).

The network definition lives in the authors' repository (cloned to
``twinlitenet.repo_dir``); this module imports it dynamically so no model code is
copied into the project.
"""

from __future__ import annotations

import importlib
import sys
from types import ModuleType

import cv2
import numpy as np
import numpy.typing as npt
import torch
from torch import nn

from dashcam_ml.config.schema import TwinLiteNetConfig
from dashcam_ml.core.ldw import LaneSegmentation

BgrImage = npt.NDArray[np.uint8]
LANE_CLASS_INDEX: int = 1


class TwinLiteNetProbabilityHead(nn.Module):
    """Wraps TwinLiteNet to output the foreground probability of both heads.

    Output shapes are ``(1, H, W)``; used by the Python segmenter and by the Core ML
    export so both produce identical tensors.
    """

    def __init__(self, *, backbone: nn.Module) -> None:
        super().__init__()
        self.backbone: nn.Module = backbone

    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        drivable_logits, lane_logits = self.backbone(image)
        drivable_prob: torch.Tensor = torch.softmax(drivable_logits, dim=1)[:, LANE_CLASS_INDEX]
        lane_prob: torch.Tensor = torch.softmax(lane_logits, dim=1)[:, LANE_CLASS_INDEX]
        return drivable_prob, lane_prob


def load_twinlitenet(*, config: TwinLiteNetConfig) -> nn.Module:
    """Instantiate the authors' model class and load the checkpoint (on CPU, eval mode)."""
    if not config.repo_dir.is_dir():
        raise FileNotFoundError(
            f"TwinLiteNet repo not found at {config.repo_dir}. Clone it first (see README, phương án B)."
        )
    repo_path: str = str(config.repo_dir)
    if repo_path not in sys.path:
        sys.path.insert(0, repo_path)
    module: ModuleType = importlib.import_module(name=config.module)
    model_class: type[nn.Module] = getattr(module, config.class_name)
    model: nn.Module = model_class()

    checkpoint: object = torch.load(f=str(config.weights), map_location="cpu")
    state_dict: dict[str, torch.Tensor] = (
        checkpoint["state_dict"] if isinstance(checkpoint, dict) and "state_dict" in checkpoint else checkpoint  # type: ignore[assignment]
    )
    prefix: str = config.state_dict_prefix
    cleaned: dict[str, torch.Tensor] = {
        (key[len(prefix) :] if prefix and key.startswith(prefix) else key): value for key, value in state_dict.items()
    }
    model.load_state_dict(state_dict=cleaned)
    return model.eval()


def preprocess_frame(*, frame_bgr: BgrImage, config: TwinLiteNetConfig) -> torch.Tensor:
    """Resize to the model size, BGR -> RGB, scale to 0..1, NCHW float tensor."""
    resized: BgrImage = cv2.resize(
        src=frame_bgr, dsize=(config.input_width, config.input_height), interpolation=cv2.INTER_LINEAR
    )
    rgb: BgrImage = cv2.cvtColor(src=resized, code=cv2.COLOR_BGR2RGB)
    chw: npt.NDArray[np.float32] = np.ascontiguousarray(rgb.transpose(2, 0, 1), dtype=np.float32) / 255.0
    return torch.from_numpy(chw).unsqueeze(dim=0)


class TwinLiteNetSegmenter:
    def __init__(self, *, config: TwinLiteNetConfig, device: str) -> None:
        self._config: TwinLiteNetConfig = config
        self._device: torch.device = torch.device(device)
        self._model: TwinLiteNetProbabilityHead = (
            TwinLiteNetProbabilityHead(backbone=load_twinlitenet(config=config)).to(self._device).eval()
        )

    def segment(self, *, frame_bgr: BgrImage) -> LaneSegmentation:
        tensor: torch.Tensor = preprocess_frame(frame_bgr=frame_bgr, config=self._config).to(self._device)
        with torch.inference_mode():
            drivable_prob, lane_prob = self._model(tensor)
        threshold: float = self._config.mask_threshold
        return LaneSegmentation(
            drivable_mask=(drivable_prob[0] > threshold).cpu().numpy(),
            lane_mask=(lane_prob[0] > threshold).cpu().numpy(),
        )
