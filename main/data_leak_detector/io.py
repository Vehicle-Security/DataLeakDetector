"""Input loading and normalization helpers for monitor logs.

The rest of the pipeline should not need to know whether a log came from JSON,
JSON Lines, UTF-8, GB18030, nested process metadata, or inconsistent path
separators. This module isolates those edge cases at the project boundary.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .models import LogEvent
from .policy import SENSITIVE_TOKENS


def read_text(path: str | Path) -> str:
    """Read collected logs with the encodings usually seen on Windows hosts."""

    target = Path(path)
    for encoding in ("utf-8-sig", "utf-8", "gb18030"):
        try:
            return target.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
    return target.read_text(encoding="utf-8", errors="replace")


def load_json_records(path: str | Path) -> list[dict[str, Any]]:
    """Load either a JSON array or JSON Lines file."""

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


def normalize_logs(records: list[dict[str, Any]]) -> list[LogEvent]:
    """Project heterogeneous raw log records into the pipeline event shape."""

    events: list[LogEvent] = []
    for index, record in enumerate(records):
        process = record.get("process_info") if isinstance(record.get("process_info"), dict) else {}
        window = record.get("window_info") if isinstance(record.get("window_info"), dict) else {}
        upload = record.get("upload_detection") if isinstance(record.get("upload_detection"), dict) else {}

        timestamp = str(record.get("timestamp") or record.get("time") or "")
        file_path = normalize_path(record.get("file_path") or record.get("path") or upload.get("temp_file") or "")
        process_name = str(record.get("process_name") or process.get("process_name") or "")
        app_name = str(record.get("app_name") or process.get("app_name") or process_name)
        window_title = str(record.get("window_title") or window.get("window_title") or "")
        description = str(record.get("description") or upload.get("upload_type") or "")

        events.append(
            LogEvent(
                event_id=str(record.get("event_id") or f"log_{index}"),
                timestamp=timestamp,
                timestamp_ms=parse_timestamp_ms(timestamp),
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


def flatten_text(value: Any) -> str:
    if isinstance(value, dict):
        return " ".join(flatten_text(item) for item in value.values())
    if isinstance(value, list):
        return " ".join(flatten_text(item) for item in value)
    return str(value or "")


def normalize_path(value: object) -> str:
    return str(value or "").strip().strip('"').replace("\\", "/")


def basename(value: object) -> str:
    return Path(normalize_path(value)).name


def same_file(left: object, right: object) -> bool:
    """Compare by normalized full path, then by basename as a pragmatic fallback."""

    lhs = normalize_path(left).lower()
    rhs = normalize_path(right).lower()
    if not lhs or not rhs:
        return False
    return lhs == rhs or basename(lhs).lower() == basename(rhs).lower()


def looks_sensitive(value: object) -> bool:
    text = str(value or "").lower()
    return any(token.lower() in text for token in SENSITIVE_TOKENS)


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
    repaired = re.sub(r'\\(?!["\\/bfnrtu])', r"\\\\", repaired)
    return repaired
