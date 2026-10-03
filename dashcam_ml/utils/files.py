from __future__ import annotations

import json
import os
import shutil
from collections.abc import Iterator
from pathlib import Path

import yaml

from dashcam_ml.domain.enums import CopyMode

IMAGE_EXTENSIONS: frozenset[str] = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".webp"})


def materialize_file(*, source: Path, destination: Path, mode: CopyMode) -> None:
    """Hard-link (falls back to copy across drives) or copy ``source`` to ``destination``."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        destination.unlink()
    if mode is CopyMode.HARDLINK:
        try:
            os.link(src=source, dst=destination)
            return
        except OSError:
            pass
    shutil.copy2(src=source, dst=destination)


def iter_images(*, directory: Path) -> Iterator[Path]:
    for path in sorted(directory.rglob(pattern="*")):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            yield path


def write_json(*, path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open(mode="w", encoding="utf-8") as handle:
        json.dump(obj=payload, fp=handle, ensure_ascii=False, indent=2)


def write_yaml(*, path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open(mode="w", encoding="utf-8") as handle:
        yaml.safe_dump(data=payload, stream=handle, allow_unicode=True, sort_keys=False)


def read_yaml(*, path: Path) -> dict[str, object]:
    with path.open(mode="r", encoding="utf-8") as handle:
        data: object = yaml.safe_load(stream=handle)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return data


def remove_path(*, path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path=path)
    elif path.exists():
        path.unlink()
