"""用于源到派生产物跟踪的文件谱系模型。

泄露推理需要知道：一个已上传的派生产物仍可能源自敏感文件。本模块在转换为 Datalog 事实
或持久化到 Neo4j 之前，先把这张图保持得紧凑且本地化。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..io import normalize_path, same_file


@dataclass
class Lineage:
    """符号推理前使用的源到派生文件映射。"""

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
