from __future__ import annotations

import json
from pathlib import Path


def write_json(*, path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open(mode="w", encoding="utf-8") as handle:
        json.dump(obj=payload, fp=handle, ensure_ascii=False, indent=2)


def read_json(*, path: Path) -> object:
    with path.open(mode="r", encoding="utf-8") as handle:
        return json.load(fp=handle)
