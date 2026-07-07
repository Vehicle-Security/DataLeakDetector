from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .models import LogEvent


SENSITIVE_NAME_TOKENS = (
    "salary",
    "payroll",
    "confidential",
    "secret",
    "contract",
    "finance",
    "customer",
    "password",
    "budget",
    "strategy",
    "internal",
    "薪资",
    "工资",
    "机密",
    "绝密",
    "合同",
    "财务",
    "客户",
    "密码",
    "预算",
    "战略",
    "内部",
)


def read_text(path: str | Path) -> str:
    """Read text with the encodings commonly seen in collected Windows logs."""

    target = Path(path)
    for encoding in ("utf-8-sig", "utf-8", "gb18030"):
        try:
            return target.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
    return target.read_text(encoding="utf-8", errors="replace")


def load_json_records(path: str | Path) -> list[dict[str, Any]]:
    """Load JSON array or JSONL logs and keep only object records."""

    text = read_text(path).strip()
    if not text:
        return []

    parsed: Any
    if text.startswith("["):
        parsed = json.loads(_repair_json(text), strict=False)
    else:
        parsed = []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            parsed.append(json.loads(_repair_json(line), strict=False))

    if isinstance(parsed, dict):
        parsed = [parsed]
    return [item for item in parsed if isinstance(item, dict)]


def normalize_logs(records: list[dict[str, Any]]) -> list[LogEvent]:
    """Project raw records into the small event shape used by the pipeline."""

    events: list[LogEvent] = []
    for index, record in enumerate(records):
        process_info = record.get("process_info") if isinstance(record.get("process_info"), dict) else {}
        window_info = record.get("window_info") if isinstance(record.get("window_info"), dict) else {}
        timestamp = str(record.get("timestamp") or record.get("time") or "")
        file_path = normalize_path(str(record.get("file_path") or record.get("path") or ""))
        file_name = str(record.get("file_name") or Path(file_path).name or "")
        process_name = str(record.get("process_name") or process_info.get("process_name") or "")
        app_name = str(record.get("app_name") or process_info.get("app_name") or process_name or "")
        window_title = str(record.get("window_title") or window_info.get("window_title") or "")
        events.append(
            LogEvent(
                event_id=str(record.get("event_id") or f"log_{index}"),
                timestamp=timestamp,
                timestamp_ms=parse_timestamp_ms(timestamp),
                event_type=str(record.get("event_type") or record.get("type") or "").lower(),
                file_path=file_path,
                file_name=file_name,
                process_name=process_name,
                app_name=app_name,
                window_title=window_title,
                raw=record,
            )
        )
    return events


def normalize_path(value: object) -> str:
    """Normalize separators without changing drive letters or casing."""

    text = str(value or "").strip().strip('"')
    return text.replace("\\", "/")


def same_file(left: object, right: object) -> bool:
    """Compare paths by normalized full path, falling back to basename."""

    lhs = normalize_path(left).lower()
    rhs = normalize_path(right).lower()
    if not lhs or not rhs:
        return False
    return lhs == rhs or Path(lhs).name == Path(rhs).name


def looks_sensitive(value: object) -> bool:
    text = str(value or "").lower()
    return any(token.lower() in text for token in SENSITIVE_NAME_TOKENS)


def flatten_text(value: Any) -> str:
    if isinstance(value, dict):
        return " ".join(flatten_text(item) for item in value.values())
    if isinstance(value, list):
        return " ".join(flatten_text(item) for item in value)
    return str(value or "")


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


def _repair_json(text: str) -> str:
    repaired = re.sub(r",(\s*[}\]])", r"\1", text)
    repaired = re.sub(r'\\(?!["\\/bfnrtu])', r"\\\\", repaired)
    return repaired
