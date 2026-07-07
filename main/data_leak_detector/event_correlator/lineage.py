"""File lineage model for source-to-derived artifact tracking.

Leak reasoning needs to know that an uploaded derivative may still originate
from a sensitive source. This module keeps that graph compact and local before
it is converted into Datalog facts or persisted to Neo4j.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..io import normalize_path, same_file


@dataclass
class Lineage:
    """Source-to-derived file mapping used before symbolic reasoning."""

    direct: dict[str, str] = field(default_factory=dict)

    def add(self, derived: str, source: str) -> None:
        derived = normalize_path(derived)
        source = normalize_path(source)
        if derived and source and not same_file(derived, source):
            self.direct[derived] = source

    def root(self, file_path: str) -> str:
        current = normalize_path(file_path)
        seen: set[str] = set()
        while current in self.direct and current not in seen:
            seen.add(current)
            current = self.direct[current]
        return current

    def chain(self, file_path: str) -> list[str]:
        current = normalize_path(file_path)
        parts = [current] if current else []
        seen: set[str] = set()
        while current in self.direct and current not in seen:
            seen.add(current)
            current = self.direct[current]
            parts.append(current)
        return parts
