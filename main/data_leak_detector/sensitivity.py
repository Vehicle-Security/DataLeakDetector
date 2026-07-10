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
DEFAULT_OPERATION_PATHS = ("operations",)
DEFAULT_OPERATION_TEXT_FIELDS = ("operation", "operation_type", "description", "label", "risk_type")
DEFAULT_SOURCE_OPERATION_TOKENS = ("正常操作", "打开", "查看", "阅读", "open", "read", "view", "normal")
DEFAULT_DERIVED_OPERATION_TOKENS = (
    "潜在隐藏行为",
    "隐藏行为",
    "直接外发",
    "屏幕截图",
    "截图",
    "录屏",
    "上传",
    "发送",
    "外发",
    "导出",
    "复制",
    "压缩",
    "派生",
    "screenshot",
    "upload",
    "send",
    "export",
    "copy",
)


@dataclass(frozen=True)
class SensitiveSourceConfig:
    fields: tuple[str, ...] = DEFAULT_SOURCE_FIELDS
    operation_paths: tuple[str, ...] = DEFAULT_OPERATION_PATHS
    operation_text_fields: tuple[str, ...] = DEFAULT_OPERATION_TEXT_FIELDS
    source_operation_tokens: tuple[str, ...] = DEFAULT_SOURCE_OPERATION_TOKENS
    derived_operation_tokens: tuple[str, ...] = DEFAULT_DERIVED_OPERATION_TOKENS
    json_paths: tuple[str, ...] = ()
    regexes: tuple[str, ...] = (r'"sensitive_file_path"\s*:\s*"([^"]+)"',)
    allow_empty: bool = False

    @classmethod
    def from_env(cls) -> "SensitiveSourceConfig":
        return cls(
            fields=_env_tuple("DLD_SENSITIVE_SOURCE_FIELDS", DEFAULT_SOURCE_FIELDS),
            operation_paths=_env_tuple("DLD_SENSITIVE_SOURCE_OPERATION_PATHS", DEFAULT_OPERATION_PATHS),
            operation_text_fields=_env_tuple("DLD_SENSITIVE_SOURCE_OPERATION_TEXT_FIELDS", DEFAULT_OPERATION_TEXT_FIELDS),
            source_operation_tokens=_env_tuple("DLD_SENSITIVE_SOURCE_OPERATION_TOKENS", DEFAULT_SOURCE_OPERATION_TOKENS),
            derived_operation_tokens=_env_tuple("DLD_SENSITIVE_DERIVED_OPERATION_TOKENS", DEFAULT_DERIVED_OPERATION_TOKENS),
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
        values.extend(_collect_source_operations(payload, config))
        if not values and not _has_operation_payload(payload, config):
            values.extend(_collect_by_field(payload, set(config.fields)))
        for json_path in config.json_paths:
            values.extend(_collect_by_path(payload, json_path))

    if payload is None or not values:
        for pattern in config.regexes:
            try:
                values.extend(re.findall(pattern, text))
            except re.error:
                continue

    normalized: list[str] = []
    for value in values:
        path_text = normalize_path(value)
        if not _is_valid_source_path(path_text):
            continue
        if not path_text and not config.allow_empty:
            continue
        if path_text and path_text not in normalized:
            normalized.append(path_text)
    return tuple(normalized)


def _collect_source_operations(payload: Any, config: SensitiveSourceConfig) -> list[str]:
    items: list[Any] = []
    for json_path in config.operation_paths:
        items.extend(_collect_raw_by_path(payload, json_path))
    if not items and isinstance(payload, list):
        items = payload

    flattened = []
    for item in items:
        if isinstance(item, list):
            flattened.extend(item)
        else:
            flattened.append(item)

    values: list[str] = []
    fields = set(config.fields)
    for item in flattened:
        if not isinstance(item, dict):
            continue
        operation_text = _operation_text(item, config.operation_text_fields)
        if not _looks_initial_source_operation(operation_text, config):
            continue
        for field in fields:
            value = item.get(field)
            if value:
                values.append(str(value))
    if not values and flattened:
        first = next((item for item in flattened if isinstance(item, dict)), None)
        if first:
            for field in config.fields:
                value = first.get(field)
                if value:
                    return [str(value)]
    return values


def _has_operation_payload(payload: Any, config: SensitiveSourceConfig) -> bool:
    for json_path in config.operation_paths:
        if _collect_raw_by_path(payload, json_path):
            return True
    return isinstance(payload, list)


def _operation_text(item: dict[str, Any], fields: tuple[str, ...]) -> str:
    return " ".join(str(item.get(field) or "").strip() for field in fields if str(item.get(field) or "").strip())


def _looks_initial_source_operation(text: str, config: SensitiveSourceConfig) -> bool:
    normalized = text.casefold()
    if not normalized:
        return True
    if any(token.casefold() in normalized for token in config.derived_operation_tokens):
        return False
    return any(token.casefold() in normalized for token in config.source_operation_tokens)


def _is_valid_source_path(value: str) -> bool:
    normalized = normalize_path(value).strip().strip("\"'").lower()
    if not normalized:
        return False
    if normalized in {"n/a", "na", "none", "null", "unknown", "-", "无", "空"} or normalized.startswith("n/a "):
        return False
    return "/" in normalized or "\\" in value


def _loads_relaxed(text: str) -> Any | None:
    candidates = [text]
    if '""' in text:
        candidates.append(text.replace('""', '"'))
    for candidate in list(candidates):
        candidates.append(_repair_backslashes(candidate))
    for candidate in candidates:
        try:
            return json.loads(candidate, strict=False)
        except json.JSONDecodeError:
            continue
    return None


def _repair_backslashes(text: str) -> str:
    return re.sub(r'(?<!\\)\\(?!["\\/bfnrtu])', r"\\\\", text)


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


def _collect_raw_by_path(payload: Any, json_path: str) -> list[Any]:
    parts = tuple(part for part in json_path.strip().strip(".").split(".") if part)
    if not parts:
        return []
    return _walk_path(payload, parts)


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
