"""Device selection and platform guards."""

from __future__ import annotations

import platform

AUTO_DEVICE: str = "auto"


def resolve_device(*, device: str) -> str | None:
    """``auto`` -> ``None`` so Ultralytics / torch pick the best device themselves."""
    return None if device.strip().lower() == AUTO_DEVICE else device


def require_coreml_conversion_platform() -> None:
    if platform.system() == "Windows":
        raise RuntimeError(
            "coremltools does not support Windows. Run the Core ML export on macOS, Linux, WSL2 or Google Colab."
        )


def require_litert_platform(*, action: str) -> None:
    """LiteRT (TFLite) export and runtime ship for Linux x86-64 and macOS only."""
    system: str = platform.system()
    if system == "Darwin" or (system == "Linux" and platform.machine().lower() in {"x86_64", "amd64"}):
        return
    raise RuntimeError(f"{action} needs Linux x86-64 or macOS (LiteRT has no Windows build). Run it on Google Colab.")


def require_macos(*, action: str) -> None:
    if platform.system() != "Darwin":
        raise RuntimeError(f"{action} needs macOS: Core ML models can only be executed on Apple platforms.")
