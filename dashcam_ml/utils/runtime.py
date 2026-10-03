"""Device selection and platform guards."""

from __future__ import annotations

import platform

AUTO_DEVICE: str = "auto"


def resolve_device(*, device: str) -> str | None:
    """``auto`` -> ``None`` so Ultralytics / torch pick the best device themselves."""
    return None if device.strip().lower() == AUTO_DEVICE else device


def resolve_torch_device(*, device: str) -> str:
    import torch

    if device.strip().lower() != AUTO_DEVICE:
        return f"cuda:{device}" if device.isdigit() else device
    if torch.cuda.is_available():
        return "cuda:0"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def require_coreml_conversion_platform() -> None:
    if platform.system() == "Windows":
        raise RuntimeError(
            "coremltools does not support Windows. Run the Core ML export on macOS, Linux, WSL2 or Google Colab."
        )


def require_macos(*, action: str) -> None:
    if platform.system() != "Darwin":
        raise RuntimeError(f"{action} needs macOS: Core ML models can only be executed on Apple platforms.")
