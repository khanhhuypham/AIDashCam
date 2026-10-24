"""Device selection."""

from __future__ import annotations

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
