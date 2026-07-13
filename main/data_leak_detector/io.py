"""监控日志的输入加载与规范化辅助工具。

流水线的其他部分不需要关心日志来自 JSON、JSON Lines、UTF-8、GB18030、
嵌套进程元数据还是不一致的路径分隔符。本模块在项目边界处隔离这些边缘情况。
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any

from .models import LogEvent
from .policy import SENSITIVE_TOKENS


def read_text(path: str | Path) -> str:
    """按 Windows 主机上常见的编码读取采集到的日志。"""

    target = Path(path)
    for encoding in ("utf-8-sig", "utf-8", "gb18030"):
        try:
            return target.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
    return target.read_text(encoding="utf-8", errors="replace")


def load_json_records(path: str | Path) -> list[dict[str, Any]]:
    """读取 JSON 数组或 JSON Lines 文件。"""

    text = read_text(path).strip()
    if not text:
        return []

    if text.startswith("["):
        parsed = json.loads(_repair_json(text), strict=False)
        return _only_objects(parsed)

    records: list[dict[str, Any]] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        records.extend(_only_objects(json.loads(_repair_json(line), strict=False)))
    return records


def normalize_logs(records: list[dict[str, Any]], session_start_ms: int | None = None) -> list[LogEvent]:
    """把异构的原始日志记录映射成流水线事件结构。"""

    events: list[LogEvent] = []
    parsed_times = [parse_timestamp_ms(record.get("timestamp") or record.get("time") or "") for record in records]
    inferred_start_ms = next((item for item in parsed_times if item), 0)
    effective_start_ms = int(session_start_ms or 0) or inferred_start_ms
    explicit_session_start = bool(session_start_ms)
    for index, record in enumerate(records):
        process = record.get("process_info") if isinstance(record.get("process_info"), dict) else {}
        window = record.get("window_info") if isinstance(record.get("window_info"), dict) else {}
        upload = record.get("upload_detection") if isinstance(record.get("upload_detection"), dict) else {}
        extra = record.get("extra") if isinstance(record.get("extra"), dict) else {}

        timestamp = str(record.get("timestamp") or record.get("time") or "")
        timestamp_ms = parsed_times[index] if index < len(parsed_times) else parse_timestamp_ms(timestamp)
        file_path = normalize_path(
            record.get("file_path")
            or record.get("path")
            or record.get("destination_path")
            or upload.get("temp_file")
            or upload.get("original_file")
            or ""
        )
        process_name = str(record.get("process_name") or process.get("process_name") or "")
        app_name = str(record.get("app_name") or process.get("app_name") or process_name)
        window_title = str(record.get("window_title") or window.get("window_title") or "")
        description = _event_description(record, upload, extra)

        events.append(
            LogEvent(
                event_id=str(record.get("event_id") or f"log_{index}"),
                timestamp=timestamp,
                timestamp_ms=timestamp_ms,
                video_time_ms=_video_time_ms(record, timestamp_ms, effective_start_ms, explicit_session_start),
                event_type=str(record.get("event_type") or record.get("type") or "").lower(),
                file_path=file_path,
                process_name=process_name,
                app_name=app_name,
                window_title=window_title,
                description=description,
                raw=record,
            )
        )
    return events


def _event_description(record: dict[str, Any], upload: dict[str, Any], extra: dict[str, Any]) -> str:
    parts = [
        record.get("description"),
        record.get("operation"),
        record.get("content_preview"),
        upload.get("upload_type"),
        upload.get("original_file"),
        extra.get("category"),
        extra.get("risk_level"),
        extra.get("operation_detail"),
        extra.get("source"),
        extra.get("raw_operation"),
    ]
    return " ".join(str(item).strip() for item in parts if str(item or "").strip())


def _video_time_ms(
    record: dict[str, Any],
    timestamp_ms: int,
    session_start_ms: int,
    explicit_session_start: bool = False,
) -> int:
    extra = record.get("extra") if isinstance(record.get("extra"), dict) else {}
    relative = extra.get("relative_timestamp")
    try:
        return max(int(float(relative) * 1000), 0)
    except (TypeError, ValueError):
        pass
    if timestamp_ms and session_start_ms:
        delta = timestamp_ms - session_start_ms
        if delta < 0 and explicit_session_start:
            return -1
        return max(delta, 0)
    return -1


def flatten_text(value: Any) -> str:
    if isinstance(value, dict):
        return " ".join(flatten_text(item) for item in value.values())
    if isinstance(value, list):
        return " ".join(flatten_text(item) for item in value)
    return str(value or "")


def normalize_path(value: object) -> str:
    text = _repair_mojibake(str(value or "")).strip().strip('"').replace("\\", "/")
    text = re.sub(r"^([A-Za-z]:)/+", r"\1/", text)
    return re.sub(r"/{2,}", "/", text)


@lru_cache(maxsize=16_384)
def _repair_mojibake(text: str) -> str:
    """Recover common Windows collector encoding mistakes conservatively."""

    if not text or text.isascii():
        return text
    original_cjk = _cjk_count(text)
    candidates = [text]
    for source_encoding in ("latin1", "cp1252"):
        for target_encoding in ("utf-8", "gb18030"):
            try:
                candidates.append(text.encode(source_encoding).decode(target_encoding))
            except (UnicodeEncodeError, UnicodeDecodeError):
                continue
    best = max(candidates, key=lambda item: (_cjk_count(item), -item.count("\ufffd"), -len(item)))
    return best if _cjk_count(best) >= original_cjk + 2 else text


def _cjk_count(text: str) -> int:
    return sum("\u3400" <= char <= "\u9fff" for char in text)


def basename(value: object) -> str:
    return Path(normalize_path(value)).name


def same_file(left: object, right: object) -> bool:
    """先按规范化后的完整路径比较，再按文件名作务实的兜底比较。"""

    lhs = normalize_path(left).lower()
    rhs = normalize_path(right).lower()
    if not lhs or not rhs or _is_placeholder_file_ref(lhs) or _is_placeholder_file_ref(rhs):
        return False
    if lhs == rhs:
        return True
    lhs_name = basename(lhs).lower()
    rhs_name = basename(rhs).lower()
    if _safe_basename_match(lhs_name) and _safe_basename_match(rhs_name):
        return _basename_match_key(lhs_name) == _basename_match_key(rhs_name)
    return False


def looks_sensitive(value: object) -> bool:
    text = str(value or "").lower()
    return any(token.lower() in text for token in SENSITIVE_TOKENS)


def _is_placeholder_file_ref(value: str) -> bool:
    normalized = value.strip().strip("\"'").lower()
    return normalized in {"n/a", "na", "none", "null", "unknown", "-", "无", "空"} or normalized.startswith("n/a ")


def _safe_basename_match(name: str) -> bool:
    if _is_placeholder_file_ref(name):
        return False
    path = Path(name)
    return bool(path.suffix and len(path.stem) >= 2)


def _basename_match_key(name: str) -> str:
    """Ignore accidental whitespace immediately before a file extension."""
    return re.sub(r"\s+(\.[^.]+)$", r"\1", name)


def parse_timestamp_ms(value: object) -> int:
    text = str(value or "").strip()
    if not text:
        return 0
    try:
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        parsed = datetime.fromisoformat(text)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return int(parsed.timestamp() * 1000)
    except ValueError:
        pass

    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y/%m/%d %H:%M:%S"):
        try:
            parsed = datetime.strptime(text[:19], fmt).replace(tzinfo=timezone.utc)
            return int(parsed.timestamp() * 1000)
        except ValueError:
            continue
    return 0


def iso_now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def _only_objects(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, dict):
        return [value]
    if isinstance(value, list):
        return [item for item in value if isinstance(item, dict)]
    return []


def _repair_json(text: str) -> str:
    repaired = re.sub(r",(\s*[}\]])", r"\1", text)
    repaired = re.sub(r'(?<!\\)\\(?!["\\/bfnrtu])', r"\\\\", repaired)
    return repaired
