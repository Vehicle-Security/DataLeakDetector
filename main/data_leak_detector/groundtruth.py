"""按数据集标注口径生成泄密判定。

当前真实样本以 `groundtruth.json` 作为“是否泄密”的评估口径。这个模块只负责
解释标注，不参与日志/OCR/VLM 证据推理；以后换数据集时，优先改
`spec/config/groundtruth_policy.json`，而不是改流水线代码。
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .io import read_text
from .policy import normalize_text


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_GROUNDTRUTH_POLICY_PATH = REPO_ROOT / "spec" / "config" / "groundtruth_policy.json"

FALLBACK_GROUNDTRUTH_POLICY: dict[str, Any] = {
    "operation_paths": ("operations",),
    "operation_text_fields": ("operation", "operation_type", "description", "label", "risk_type"),
    "leak_tokens": (
        "直接外发",
        "邮箱外发",
        "复制内容外发",
        "内容外发",
        "完成外传",
        "泄露",
        "外传",
        "上传",
        "发送",
        "屏幕共享展示敏感文件",
        "共享屏幕展示敏感文件",
        "exfiltration",
        "leak",
    ),
    "non_leak_tokens": ("正常操作", "normal"),
    "unknown_risk_tokens": ("潜在隐藏行为", "可疑", "隐藏行为"),
}


@dataclass(frozen=True)
class GroundTruthPolicy:
    operation_paths: tuple[str, ...]
    operation_text_fields: tuple[str, ...]
    leak_tokens: tuple[str, ...]
    non_leak_tokens: tuple[str, ...]
    unknown_risk_tokens: tuple[str, ...]


@dataclass(frozen=True)
class GroundTruthOperation:
    index: int
    timestamp: str
    sensitive_file: str
    operation: str
    label: str


@dataclass(frozen=True)
class GroundTruthVerdict:
    available: bool
    source: str
    conclusion: str
    total_operations: int
    leak_operations: tuple[GroundTruthOperation, ...]
    non_leak_operations: tuple[GroundTruthOperation, ...]
    unknown_risk_operations: tuple[GroundTruthOperation, ...]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["leak_operations"] = [asdict(item) for item in self.leak_operations]
        payload["non_leak_operations"] = [asdict(item) for item in self.non_leak_operations]
        payload["unknown_risk_operations"] = [asdict(item) for item in self.unknown_risk_operations]
        return payload


def load_groundtruth_policy(path: str | Path | None = None) -> GroundTruthPolicy:
    """加载 groundtruth 判定策略；配置缺失时使用当前数据集的最小口径。"""

    config_path = Path(path or os.getenv("DLD_GROUNDTRUTH_POLICY_CONFIG") or DEFAULT_GROUNDTRUTH_POLICY_PATH)
    raw = dict(FALLBACK_GROUNDTRUTH_POLICY)
    if config_path.exists():
        raw.update(json.loads(config_path.read_text(encoding="utf-8")))
    return GroundTruthPolicy(
        operation_paths=_tuple(raw.get("operation_paths")),
        operation_text_fields=_tuple(raw.get("operation_text_fields")),
        leak_tokens=_tokens(raw, "leak_tokens", "DLD_GROUNDTRUTH_LEAK_TOKENS"),
        non_leak_tokens=_tokens(raw, "non_leak_tokens", "DLD_GROUNDTRUTH_NON_LEAK_TOKENS"),
        unknown_risk_tokens=_tokens(raw, "unknown_risk_tokens", "DLD_GROUNDTRUTH_UNKNOWN_RISK_TOKENS"),
    )


def evaluate_groundtruth(path: str | Path | None, policy: GroundTruthPolicy | None = None) -> GroundTruthVerdict:
    """根据 groundtruth 标注生成当前样本的泄密结论。"""

    if path is None or not Path(path).exists():
        return GroundTruthVerdict(
            available=False,
            source="missing",
            conclusion="unknown",
            total_operations=0,
            leak_operations=(),
            non_leak_operations=(),
            unknown_risk_operations=(),
        )

    policy = policy or load_groundtruth_policy()
    text = read_text(path)
    payload = _loads_relaxed(text)
    operations = _extract_operations(payload, policy) if payload is not None else _scan_text_operations(text, policy)
    leak_ops = tuple(item for item in operations if item.label == "leak")
    non_leak_ops = tuple(item for item in operations if item.label == "non_leak")
    unknown_ops = tuple(item for item in operations if item.label == "unknown_risk")
    return GroundTruthVerdict(
        available=True,
        source=str(path),
        conclusion=_groundtruth_conclusion(leak_ops, unknown_ops),
        total_operations=len(operations),
        leak_operations=leak_ops,
        non_leak_operations=non_leak_ops,
        unknown_risk_operations=unknown_ops,
    )


def _groundtruth_conclusion(
    leak_ops: tuple[GroundTruthOperation, ...],
    unknown_ops: tuple[GroundTruthOperation, ...],
) -> str:
    if leak_ops:
        return "data_leak_risk_detected"
    if unknown_ops:
        return "suspicious_behavior_detected"
    return "no_confirmed_data_leak"


def _extract_operations(payload: Any, policy: GroundTruthPolicy) -> tuple[GroundTruthOperation, ...]:
    raw_items: list[Any] = []
    for json_path in policy.operation_paths:
        raw_items.extend(_collect_by_path(payload, json_path))
    raw_items = _flatten_operation_items(raw_items)
    if not raw_items and isinstance(payload, list):
        raw_items = payload

    operations: list[GroundTruthOperation] = []
    for index, item in enumerate(raw_items):
        if not isinstance(item, dict):
            continue
        operation_text = _operation_text(item, policy)
        operations.append(
            GroundTruthOperation(
                index=index,
                timestamp=str(item.get("operation_time") or item.get("timestamp") or item.get("time") or ""),
                sensitive_file=str(item.get("sensitive_file_path") or item.get("sensitive_file") or item.get("source_file") or ""),
                operation=operation_text,
                label=_label_operation(operation_text, policy),
            )
        )
    return tuple(operations)


def _flatten_operation_items(items: list[Any]) -> list[Any]:
    flattened: list[Any] = []
    for item in items:
        if isinstance(item, list):
            flattened.extend(item)
        else:
            flattened.append(item)
    return flattened


def _scan_text_operations(text: str, policy: GroundTruthPolicy) -> tuple[GroundTruthOperation, ...]:
    label = _label_operation(text, policy)
    if label == "none":
        return ()
    return (
        GroundTruthOperation(
            index=0,
            timestamp="",
            sensitive_file="",
            operation=text[:500],
            label=label,
        ),
    )


def _operation_text(item: dict[str, Any], policy: GroundTruthPolicy) -> str:
    values = [str(item.get(field) or "") for field in policy.operation_text_fields]
    return " ".join(value for value in values if value)


def _label_operation(text: str, policy: GroundTruthPolicy) -> str:
    normalized = normalize_text(text)
    if _contains_any(normalized, policy.non_leak_tokens):
        return "non_leak"
    if _contains_any(normalized, policy.leak_tokens):
        return "leak"
    if _contains_any(normalized, policy.unknown_risk_tokens):
        return "unknown_risk"
    return "none"


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


def _collect_by_path(payload: Any, json_path: str) -> list[Any]:
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


def _contains_any(text: str, tokens: tuple[str, ...]) -> bool:
    return any(normalize_text(token) in text for token in tokens if normalize_text(token))


def _tokens(raw: dict[str, Any], key: str, env_name: str) -> tuple[str, ...]:
    tokens = list(_tuple(raw.get(key)))
    for token in os.getenv(env_name, "").split(","):
        token = token.strip()
        if token:
            tokens.append(token)
    return tuple(dict.fromkeys(tokens))


def _tuple(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    if isinstance(value, list | tuple | set):
        return tuple(str(item).strip() for item in value if str(item).strip())
    return (str(value).strip(),) if str(value).strip() else ()
