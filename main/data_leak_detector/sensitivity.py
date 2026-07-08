"""可配置地提取初始敏感源文件。

初始敏感文件来自数据集标注，不包含衍生文件。这个模块把字段名、JSON 路径和
兜底正则都放进配置里，换数据集时不用改事件关联逻辑。
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .io import normalize_path, read_text

DEFAULT_SOURCE_FIELDS = (
    "sensitive_file_path",
    "sensitive_file",
    "sensitive_path",
    "source_file",
)


@dataclass(frozen=True)
class SensitiveSourceConfig:
    fields: tuple[str, ...] = DEFAULT_SOURCE_FIELDS
    json_paths: tuple[str, ...] = ()
    regexes: tuple[str, ...] = (r'"sensitive_file_path"\s*:\s*"([^"]+)"',)
    allow_empty: bool = False

    @classmethod
    def from_env(cls) -> "SensitiveSourceConfig":
        return cls(
            fields=_env_tuple("DLD_SENSITIVE_SOURCE_FIELDS", DEFAULT_SOURCE_FIELDS),
            json_paths=_env_tuple("DLD_SENSITIVE_SOURCE_JSON_PATHS", ()),
            regexes=_env_tuple("DLD_SENSITIVE_SOURCE_REGEXES", (r'"sensitive_file_path"\s*:\s*"([^"]+)"',)),
            allow_empty=_env_bool("DLD_SENSITIVE_SOURCE_ALLOW_EMPTY", False),
        )


def extract_sensitive_sources(path: str | Path | None, config: SensitiveSourceConfig | None = None) -> tuple[str, ...]:
    """从数据集标注文件中提取初始敏感源路径。"""

    if path is None:
        return ()
    target = Path(path)
    if not target.exists():
        return ()

    config = config or SensitiveSourceConfig.from_env()
    text = read_text(target)
    values: list[str] = []

    payload = _loads_relaxed(text)
    if payload is not None:
        values.extend(_collect_by_field(payload, set(config.fields)))
        for json_path in config.json_paths:
            values.extend(_collect_by_path(payload, json_path))

    for pattern in config.regexes:
        try:
            values.extend(re.findall(pattern, text))
        except re.error:
            continue

    normalized: list[str] = []
    for value in values:
        path_text = normalize_path(value)
        if not path_text and not config.allow_empty:
            continue
        if path_text and path_text not in normalized:
            normalized.append(path_text)
    return tuple(normalized)


def _loads_relaxed(text: str) -> Any | None:
    candidates = [text]
    if '""' in text:
        candidates.append(text.replace('""', '"'))
    for candidate in candidates:
        try:
            return json.loads(candidate, strict=False)
        except json.JSONDecodeError:
            continue
    return None


def _collect_by_field(value: Any, fields: set[str]) -> list[str]:
    values: list[str] = []
    if isinstance(value, dict):
        for key, item in value.items():
            if key in fields:
                values.append(str(item or ""))
            values.extend(_collect_by_field(item, fields))
    elif isinstance(value, list):
        for item in value:
            values.extend(_collect_by_field(item, fields))
    return values


def _collect_by_path(payload: Any, json_path: str) -> list[str]:
    parts = tuple(part for part in json_path.strip().strip(".").split(".") if part)
    if not parts:
        return []
    return [str(item or "") for item in _walk_path(payload, parts)]


def _walk_path(value: Any, parts: tuple[str, ...]) -> list[Any]:
    if not parts:
        return [value]
    head, *tail = parts
    rest = tuple(tail)
    values: list[Any] = []
    if head == "*":
        if isinstance(value, list):
            for item in value:
                values.extend(_walk_path(item, rest))
        elif isinstance(value, dict):
            for item in value.values():
                values.extend(_walk_path(item, rest))
    elif isinstance(value, dict) and head in value:
        values.extend(_walk_path(value[head], rest))
    return values


def _env_tuple(name: str, default: tuple[str, ...]) -> tuple[str, ...]:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    return tuple(item.strip() for item in value.split(",") if item.strip())


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}
