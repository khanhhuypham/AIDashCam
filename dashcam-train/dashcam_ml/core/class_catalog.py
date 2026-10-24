"""Lookup of :class:`ClassSpec` by model class name or alias (case-insensitive)."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from dashcam_ml.config.schema import ClassSpec


class ClassCatalog:
    def __init__(self, *, specs: Sequence[ClassSpec]) -> None:
        self._specs: tuple[ClassSpec, ...] = tuple(specs)
        self._by_name: dict[str, ClassSpec] = {}
        for spec in self._specs:
            for key in (spec.name, *spec.aliases):
                normalized: str = key.strip().lower()
                if normalized in self._by_name:
                    raise ValueError(f"class name or alias '{key}' is defined twice in config.classes")
                self._by_name[normalized] = spec

    @property
    def specs(self) -> tuple[ClassSpec, ...]:
        return self._specs

    def lookup(self, *, class_name: str) -> ClassSpec | None:
        return self._by_name.get(class_name.strip().lower())

    def contains(self, *, class_name: str) -> bool:
        return self.lookup(class_name=class_name) is not None

    def equivalent_names(self, *, class_name: str) -> set[str]:
        """All lower-case names that refer to the same class (itself included)."""
        normalized: str = class_name.strip().lower()
        spec: ClassSpec | None = self._by_name.get(normalized)
        if spec is None:
            return {normalized}
        return {normalized, spec.name.lower(), *(alias.lower() for alias in spec.aliases)}

    def match_index(self, *, class_name: str, candidates: Iterable[tuple[int, str]]) -> int | None:
        """Index of the first candidate ``(index, name)`` equivalent to ``class_name``."""
        wanted: set[str] = self.equivalent_names(class_name=class_name)
        for index, name in candidates:
            if name.strip().lower() in wanted:
                return index
        return None
