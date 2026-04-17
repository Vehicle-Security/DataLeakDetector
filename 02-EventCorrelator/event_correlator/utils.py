from __future__ import annotations

import re
from datetime import datetime
from typing import Any, Optional


TIMESTAMP_FORMATS = (
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%dT%H:%M:%S",
    "%Y-%m-%dT%H:%M:%S.%f",
    "%Y-%m-%d %H:%M:%S.%f",
)


UNKNOWN_RESOURCE_MARKERS = {"", "未知", "??", "unknown", "unk", "n/a"}


def normalize_file_path(file_path: str) -> str:
    if not file_path:
        return ""

    normalized = str(file_path).strip().replace("\\", "/")
    while "//" in normalized:
        normalized = normalized.replace("//", "/")
    return normalized


def is_unknown_resource(value: str) -> bool:
    text = str(value or "").strip()
    if not text:
        return True
    return text in UNKNOWN_RESOURCE_MARKERS or text.lower() in UNKNOWN_RESOURCE_MARKERS


def get_path_basename(file_path: str) -> str:
    normalized = normalize_file_path(file_path)
    return normalized.rsplit("/", 1)[-1] if normalized else ""


def normalize_app_name(app_name: str) -> str:
    return str(app_name or "").strip()


def normalize_timestamp_text(timestamp: str) -> str:
    if not timestamp:
        return ""

    text = str(timestamp).strip().replace("T", " ")
    if text.endswith("Z"):
        text = text[:-1]
    if "." in text:
        text = text.split(".", 1)[0]
    return text


def parse_timestamp(value: str) -> Optional[datetime]:
    if not value:
        return None

    text = str(value).strip()
    if text.endswith("Z"):
        text = text[:-1]

    for fmt in TIMESTAMP_FORMATS:
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue

    normalized = normalize_timestamp_text(text)
    for fmt in TIMESTAMP_FORMATS:
        try:
            return datetime.strptime(normalized, fmt)
        except ValueError:
            continue
    return None


def parse_time_range(time_range: str) -> tuple[Optional[datetime], Optional[datetime]]:
    if not time_range:
        return None, None

    parts = re.split(r"\s+-\s+", str(time_range).strip(), maxsplit=1)
    if len(parts) != 2:
        return None, None

    return parse_timestamp(parts[0]), parse_timestamp(parts[1])


def minute_bucket(value: str) -> str:
    normalized = normalize_timestamp_text(value)
    return normalized[:16] if normalized else ""


def choose_non_empty(*values: Any) -> str:
    for value in values:
        text = str(value or "").strip()
        if text:
            return text
    return ""


def extract_parent_candidates(raw_event: dict[str, Any]) -> list[str]:
    upload_detection = raw_event.get("upload_detection", {}) or {}
    extra = raw_event.get("extra", {}) or {}
    candidates = [
        raw_event.get("source_path"),
        raw_event.get("src_path"),
        raw_event.get("source_file"),
        raw_event.get("original_file"),
        raw_event.get("original_path"),
        raw_event.get("old_path"),
        raw_event.get("old_file_path"),
        raw_event.get("parent_path"),
        raw_event.get("from_path"),
        upload_detection.get("original_file"),
        upload_detection.get("source_file"),
        extra.get("original_file"),
        extra.get("source_file"),
    ]
    normalized: list[str] = []
    seen = set()
    for candidate in candidates:
        path = normalize_file_path(str(candidate or ""))
        if not path or path in seen:
            continue
        seen.add(path)
        normalized.append(path)
    return normalized


def extract_resource_tokens(*values: Any) -> set[str]:
    tokens: set[str] = set()
    for value in values:
        if isinstance(value, (list, tuple, set)):
            for item in value:
                tokens.update(extract_resource_tokens(item))
            continue

        text = str(value or "").strip()
        if not text:
            continue

        normalized_text = normalize_file_path(text).lower()
        tokens.add(normalized_text)
        tokens.add(get_path_basename(normalized_text))
    return {token for token in tokens if token}
