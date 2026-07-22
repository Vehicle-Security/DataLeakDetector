"""用于源到派生产物跟踪的文件谱系模型。

泄露推理需要知道：一个已上传的派生产物仍可能源自敏感文件。本模块在转换为 Datalog 事实
或持久化到 Neo4j 之前，先把这张图保持得紧凑且本地化。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from ..io import normalize_path


@dataclass
class Lineage:
    """符号推理前使用的源到派生文件映射。"""

    direct: dict[str, str] = field(default_factory=dict)

    def add(self, derived: str, source: str) -> None:
        derived = normalize_path(derived)
        source = self.resolve_artifact(source)
        if (
            not derived
            or not source
            or _same_artifact_path(derived, source)
            or any(_same_artifact_path(derived, existing) for existing in self.direct)
        ):
            return
        if any(_same_artifact_path(derived, item) for item in self.chain(source)):
            return
        self.direct[derived] = source

    def root(self, file_path: str) -> str:
        current = self.resolve_artifact(file_path)
        seen: set[str] = set()
        while current in self.direct and current not in seen:
            seen.add(current)
            current = self.resolve_artifact(self.direct[current])
        return current

    def chain(self, file_path: str) -> list[str]:
        current = self.resolve_artifact(file_path)
        parts = [current] if current else []
        seen: set[str] = set()
        while current in self.direct and current not in seen:
            seen.add(current)
            current = self.resolve_artifact(self.direct[current])
            parts.append(current)
        return parts

    def resolve_artifact(self, file_path: str) -> str:
        """Resolve a path or basename reference to one canonical known artifact."""

        reference = normalize_path(file_path)
        if not reference:
            return reference
        if "/" in reference:
            reference_key = reference.lower()
            matches = {
                derived
                for derived in self.direct
                if normalize_path(derived).lower() == reference_key
            }
            if len(matches) == 1:
                return next(iter(matches))
            return reference
        reference_key = reference.lower()
        matches = {
            derived
            for derived in self.direct
            if reference_key in _artifact_aliases(derived)
        }
        full_path_matches = {item for item in matches if "/" in normalize_path(item)}
        if len(full_path_matches) == 1:
            return next(iter(full_path_matches))
        # File-dialog monitors sometimes omit the extension for a selected
        # document (for example ``普通文件``). Resolve that stem only when it
        # identifies exactly one concrete derived artifact in this lineage.
        if not Path(reference).suffix:
            stem_matches = {
                derived
                for derived in self.direct
                if Path(normalize_path(derived)).stem.lower() == reference_key
            }
            if len(stem_matches) == 1:
                return next(iter(stem_matches))
        if reference in self.direct:
            return reference
        return next(iter(matches)) if len(matches) == 1 else reference


def _same_artifact_path(left: str, right: str) -> bool:
    return normalize_path(left).lower() == normalize_path(right).lower()


def _artifact_aliases(file_path: str) -> set[str]:
    name = Path(normalize_path(file_path)).name.lower()
    aliases = {name} if name else set()
    wrapped_name = Path(name).stem
    # Only unwrap container-style names such as ``2.ksheet.wpsonline``. A
    # regular document like ``secret.pdf`` must not acquire the vague alias
    # ``secret``.
    if Path(wrapped_name).suffix:
        aliases.add(wrapped_name)
    return aliases
