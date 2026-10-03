"""``manifest.json`` next to the exported models: everything the iOS app must know
about each ``.mlpackage`` (input size, precision, class names, IO names)."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from dashcam_ml.domain.enums import ModelKind, WeightPrecision
from dashcam_ml.utils.files import write_json

MANIFEST_NAME: str = "manifest.json"
MANIFEST_SCHEMA_VERSION: int = 1
MINIMUM_IOS_VERSION: str = "16.0"


@dataclass(frozen=True, slots=True)
class ModelManifestEntry:
    name: str
    kind: ModelKind
    file_name: str
    input_width: int
    input_height: int
    precision: WeightPrecision
    source_weights: str
    target_devices: str
    input_names: list[str]
    output_names: list[str]
    class_names: list[str] = field(default_factory=list)
    includes_nms: bool = False

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = asdict(self)
        payload["kind"] = self.kind.value
        payload["precision"] = self.precision.value
        return payload

    @staticmethod
    def from_dict(*, payload: dict[str, object]) -> ModelManifestEntry:
        values: dict[str, object] = dict(payload)
        values["kind"] = ModelKind(str(values["kind"]))
        values["precision"] = WeightPrecision(str(values["precision"]))
        return ModelManifestEntry(**values)  # type: ignore[arg-type]


def read_manifest(*, output_dir: Path) -> list[ModelManifestEntry]:
    manifest_path: Path = output_dir / MANIFEST_NAME
    if not manifest_path.is_file():
        return []
    with manifest_path.open(mode="r", encoding="utf-8") as handle:
        data: dict[str, object] = json.load(fp=handle)
    models: list[dict[str, object]] = data.get("models", [])  # type: ignore[assignment]
    return [ModelManifestEntry.from_dict(payload=item) for item in models]


def upsert_manifest(*, output_dir: Path, entries: list[ModelManifestEntry]) -> Path:
    """Insert or replace entries by ``name``; other models already listed are kept."""
    by_name: dict[str, ModelManifestEntry] = {entry.name: entry for entry in read_manifest(output_dir=output_dir)}
    for entry in entries:
        by_name[entry.name] = entry
    manifest_path: Path = output_dir / MANIFEST_NAME
    write_json(
        path=manifest_path,
        payload={
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "minimum_ios_version": MINIMUM_IOS_VERSION,
            "models": [by_name[name].to_dict() for name in sorted(by_name)],
        },
    )
    return manifest_path
