# -*- coding: utf-8 -*-
"""
Helpers for the final keyevents export contract.

`logs.json` keeps raw monitor output. `keyevents.json` is stricter:
- file events must keep a precise full file path
- app_switch / website_visit may be kept only after being linked to one
  verifiable precise file path from nearby file events
"""

from __future__ import annotations

import copy
import ntpath
import os
import re
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Tuple

from .log_contract import clean_text, is_non_file_event, looks_like_full_path


CONTEXT_WINDOW_EVENT_TYPES = {"app_switch", "website_visit"}
INTERNAL_END_EVENT_TYPES = {"closed", "window_closed"}
DURATION_FILE_EVENT_TYPES = {"opened", "read"}
INSTANT_END_EVENT_TYPES = {
    "created",
    "modified",
    "deleted",
    "renamed",
    "file_selected",
    "upload_detected",
    "inferred_upload",
    "manual_note",
}
DEFAULT_CORRELATION_WINDOW_SECONDS = 30.0

SYSTEM_PATH_PATTERNS = [
    r"c:\\windows\\",
    r"c:\\program files\\",
    r"c:\\program files \(x86\)\\",
    r"c:\\programdata\\",
    r"\\appdata\\local\\temp\\",
    r"\\appdata\\local\\google\\chrome\\user data\\",
    r"\\appdata\\local\\microsoft\\",
    r"\\appdata\\roaming\\microsoft\\windows\\recent\\",
    r"\\google\\chrome\\application\\",
    r"\\mozilla firefox\\",
    r"\\microsoft\\edge\\",
]

SYSTEM_EXTENSIONS = {
    ".dll",
    ".exe",
    ".sys",
    ".drv",
    ".ocx",
    ".pf",
    ".sdb",
    ".nls",
    ".mui",
    ".cat",
    ".etl",
    ".log",
    ".bak",
    ".tmp",
    ".temp",
    ".lnk",
    ".url",
    ".dat",
    ".db",
    ".sqlite",
    ".db-journal",
    ".manifest",
    ".config",
    ".crdownload",
    ".partial",
}

MEANINGFUL_EXTENSIONS = {
    ".doc",
    ".docx",
    ".pdf",
    ".txt",
    ".rtf",
    ".odt",
    ".wps",
    ".xls",
    ".xlsx",
    ".csv",
    ".ods",
    ".ppt",
    ".pptx",
    ".odp",
    ".jpg",
    ".jpeg",
    ".png",
    ".gif",
    ".bmp",
    ".tiff",
    ".webp",
    ".svg",
    ".zip",
    ".rar",
    ".7z",
    ".tar",
    ".gz",
    ".py",
    ".js",
    ".java",
    ".cpp",
    ".c",
    ".h",
    ".cs",
    ".go",
    ".rs",
    ".html",
    ".css",
    ".json",
    ".xml",
    ".yaml",
    ".yml",
    ".md",
    ".mp3",
    ".mp4",
    ".avi",
    ".mov",
    ".mkv",
}

USER_PATH_PATTERNS = [
    r"\\users\\[^\\]+\\documents\\",
    r"\\users\\[^\\]+\\desktop\\",
    r"\\users\\[^\\]+\\downloads\\",
    r"d:\\",
]


def _safe_lower(value: Any) -> str:
    return clean_text(value).lower()


def _basename(path: str) -> str:
    return ntpath.basename(clean_text(path).rstrip("\\/"))


def _splitext(path: str) -> Tuple[str, str]:
    return ntpath.splitext(_basename(path))


def _path_key(path: str) -> str:
    text = clean_text(path)
    return ntpath.normcase(ntpath.normpath(text)) if text else ""


def _parse_timestamp(timestamp_text: str) -> Optional[datetime]:
    text = clean_text(timestamp_text)
    if not text:
        return None

    normalized = text.replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(normalized)
        if parsed.tzinfo is not None:
            parsed = parsed.replace(tzinfo=None)
        return parsed
    except ValueError:
        pass

    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S"):
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue
    return None


def _format_timestamp(value: datetime) -> str:
    return value.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3]


def is_context_window_event(event_type: str) -> bool:
    return clean_text(event_type) in CONTEXT_WINDOW_EVENT_TYPES


def is_meaningful_file_event_for_keyevents(event: Dict[str, Any]) -> bool:
    event_type = clean_text(event.get("event_type", ""))
    if is_context_window_event(event_type) or is_non_file_event(event_type):
        return False

    file_path = clean_text(event.get("file_path", ""))
    if not file_path or not looks_like_full_path(file_path):
        return False

    file_path_lower = file_path.lower()
    for pattern in SYSTEM_PATH_PATTERNS:
        if re.search(pattern, file_path_lower):
            return False

    file_ext = clean_text(event.get("file_extension", "")) or _splitext(file_path)[1]
    file_ext_lower = file_ext.lower()

    if file_ext_lower in SYSTEM_EXTENSIONS:
        return False

    if not file_ext or file_ext == ".":
        basename = _basename(file_path)
        if basename and "." not in basename:
            return False

    if file_path.endswith("\\") or file_path.endswith("/"):
        return False

    if file_ext_lower in MEANINGFUL_EXTENSIONS:
        return True

    for pattern in USER_PATH_PATTERNS:
        if re.search(pattern, file_path_lower):
            return True

    return False


def _same_process_or_app(window_event: Dict[str, Any], candidate: Dict[str, Any]) -> bool:
    window_proc = window_event.get("process_info", {}) or {}
    candidate_proc = candidate.get("process_info", {}) or {}

    window_pid = clean_text(window_proc.get("pid", ""))
    candidate_pid = clean_text(candidate_proc.get("pid", ""))
    if window_pid and candidate_pid and window_pid == candidate_pid:
        return True

    window_process_name = _safe_lower(window_proc.get("process_name", ""))
    candidate_process_name = _safe_lower(candidate_proc.get("process_name", ""))
    if window_process_name and candidate_process_name and window_process_name == candidate_process_name:
        return True

    window_app_name = _safe_lower(window_event.get("app_name", ""))
    candidate_app_name = _safe_lower(candidate.get("app_name", ""))
    return bool(window_app_name and candidate_app_name and window_app_name == candidate_app_name)


def _title_matches_candidate(window_event: Dict[str, Any], candidate: Dict[str, Any]) -> bool:
    title = _safe_lower((window_event.get("window_info", {}) or {}).get("window_title", ""))
    basename = _safe_lower(_basename(candidate.get("file_path", "")))
    return bool(title and basename and basename in title)


def _candidate_time_distance_seconds(window_event: Dict[str, Any], candidate: Dict[str, Any]) -> Optional[float]:
    window_ts = _parse_timestamp(window_event.get("timestamp", ""))
    candidate_ts = _parse_timestamp(candidate.get("timestamp", ""))
    if window_ts is None or candidate_ts is None:
        return None
    return abs((candidate_ts - window_ts).total_seconds())


def _choose_context_candidate(
    window_event: Dict[str, Any],
    candidates: Iterable[Dict[str, Any]],
    correlation_window_seconds: float,
) -> Optional[Dict[str, Any]]:
    matched: List[Tuple[float, Dict[str, Any]]] = []

    for candidate in candidates:
        if not _same_process_or_app(window_event, candidate):
            continue

        distance = _candidate_time_distance_seconds(window_event, candidate)
        if distance is None or distance > correlation_window_seconds:
            continue

        matched.append((distance, candidate))

    if not matched:
        return None

    title_matched = [(distance, candidate) for distance, candidate in matched if _title_matches_candidate(window_event, candidate)]
    shortlisted = title_matched if title_matched else matched

    unique_paths = {_path_key(candidate.get("file_path", "")) for _, candidate in shortlisted}
    unique_paths.discard("")
    if len(unique_paths) != 1:
        return None

    shortlisted.sort(key=lambda item: item[0])
    return shortlisted[0][1]


def bind_context_window_event_paths(
    events: Iterable[Dict[str, Any]],
    correlation_window_seconds: float = DEFAULT_CORRELATION_WINDOW_SECONDS,
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    bound_events = [copy.deepcopy(event) for event in events]
    candidates = [event for event in bound_events if is_meaningful_file_event_for_keyevents(event)]

    stats = {
        "bound_window_events": 0,
        "unbound_window_events": 0,
    }

    for event in bound_events:
        if not is_context_window_event(event.get("event_type", "")):
            continue

        if clean_text(event.get("file_path", "")):
            continue

        candidate = _choose_context_candidate(event, candidates, correlation_window_seconds)
        if candidate is None:
            stats["unbound_window_events"] += 1
            continue

        file_path = clean_text(candidate.get("file_path", ""))
        event["file_path"] = file_path
        event["file_name"] = _basename(file_path)
        event["file_extension"] = clean_text(candidate.get("file_extension", "")) or _splitext(file_path)[1]
        event["file_size"] = int(candidate.get("file_size", 0) or 0)

        disk_info = event.get("disk_info")
        if not isinstance(disk_info, dict):
            disk_info = {}
            event["disk_info"] = disk_info
        if not clean_text(disk_info.get("drive_letter", "")) and len(file_path) >= 2 and file_path[1] == ":":
            disk_info["drive_letter"] = file_path[:2]

        stats["bound_window_events"] += 1

    return bound_events, stats


def should_keep_keyevent(event: Dict[str, Any]) -> bool:
    event_type = clean_text(event.get("event_type", ""))

    if event_type in INTERNAL_END_EVENT_TYPES:
        return False

    if is_context_window_event(event_type):
        file_path = clean_text(event.get("file_path", ""))
        if file_path:
            return looks_like_full_path(file_path)
        process_info = event.get("process_info", {}) or {}
        window_info = event.get("window_info", {}) or {}
        return bool(
            clean_text(window_info.get("window_title", "")) or
            clean_text(process_info.get("process_name", "")) or
            clean_text(event.get("app_name", ""))
        )

    if is_non_file_event(event_type):
        return True

    return is_meaningful_file_event_for_keyevents(event)


def _process_key(event: Dict[str, Any]) -> str:
    process_info = event.get("process_info", {}) or {}
    pid = clean_text(process_info.get("pid", ""))
    process_name = _safe_lower(process_info.get("process_name", ""))
    return pid or process_name


def _file_timeline_key(event: Dict[str, Any]) -> Tuple[str, str]:
    return (_path_key(event.get("file_path", "")), _process_key(event))


def _window_timeline_key(event: Dict[str, Any]) -> Tuple[str, str, str]:
    window_info = event.get("window_info", {}) or {}
    return (
        clean_text(window_info.get("window_handle", "")),
        _process_key(event),
        _safe_lower(window_info.get("window_title", "")),
    )


def _set_end_time(event: Dict[str, Any], end_time: datetime, source: str) -> None:
    event["end_time"] = _format_timestamp(end_time)
    extra = event.get("extra")
    if not isinstance(extra, dict):
        extra = {}
        event["extra"] = extra
    extra["end_time_source"] = source


def _first_after(
    candidates: Iterable[Dict[str, Any]],
    start_time: datetime,
) -> Optional[Tuple[datetime, Dict[str, Any]]]:
    matched: List[Tuple[datetime, Dict[str, Any]]] = []
    for candidate in candidates:
        candidate_time = _parse_timestamp(candidate.get("timestamp", ""))
        if candidate_time is not None and candidate_time >= start_time:
            matched.append((candidate_time, candidate))
    if not matched:
        return None
    matched.sort(key=lambda item: item[0])
    return matched[0]


def annotate_keyevent_end_times(
    keyevents: Iterable[Dict[str, Any]],
    timeline_events: Iterable[Dict[str, Any]],
    session_end_time: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Add end_time to every exported keyevent."""
    result = [copy.deepcopy(event) for event in keyevents]
    timeline = [copy.deepcopy(event) for event in timeline_events]

    session_end = _parse_timestamp(session_end_time or "")

    closed_by_file: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    next_file_by_key: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    window_closed_by_key: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = {}
    window_events: List[Dict[str, Any]] = []

    for event in timeline:
        event_type = clean_text(event.get("event_type", ""))
        if event_type == "closed":
            closed_by_file.setdefault(_file_timeline_key(event), []).append(event)
        elif event_type == "window_closed":
            window_closed_by_key.setdefault(_window_timeline_key(event), []).append(event)
        elif event_type in DURATION_FILE_EVENT_TYPES and clean_text(event.get("file_path", "")):
            next_file_by_key.setdefault(_file_timeline_key(event), []).append(event)
        elif is_context_window_event(event_type):
            window_events.append(event)

    for event in result:
        event_type = clean_text(event.get("event_type", ""))
        start_time = _parse_timestamp(event.get("timestamp", ""))
        if start_time is None:
            fallback = session_end or datetime.now()
            _set_end_time(event, fallback, "session_stop")
            continue

        if event_type in DURATION_FILE_EVENT_TYPES:
            key = _file_timeline_key(event)
            closed = _first_after(closed_by_file.get(key, []), start_time)
            if closed:
                source = "cpp_etw_close" if (closed[1].get("extra", {}) or {}).get("source") == "etw_monitor" else "etw_close"
                _set_end_time(event, closed[0], source)
                continue

            later_file_events = [
                candidate for candidate in next_file_by_key.get(key, [])
                if candidate is not event and clean_text(candidate.get("timestamp", "")) != clean_text(event.get("timestamp", ""))
            ]
            next_file = _first_after(later_file_events, start_time)
            if next_file:
                _set_end_time(event, next_file[0], "next_file_event")
                continue

            _set_end_time(event, session_end or start_time, "session_stop")
            continue

        if is_context_window_event(event_type):
            key = _window_timeline_key(event)
            closed = _first_after(window_closed_by_key.get(key, []), start_time)
            if closed:
                _set_end_time(event, closed[0], "window_closed")
                continue

            later_windows = [
                candidate for candidate in window_events
                if clean_text(candidate.get("timestamp", "")) != clean_text(event.get("timestamp", ""))
            ]
            next_window = _first_after(later_windows, start_time)
            if next_window:
                _set_end_time(event, next_window[0], "next_window_event")
                continue

            _set_end_time(event, session_end or start_time, "session_stop")
            continue

        _set_end_time(event, start_time, "instant")

    return result


def finalize_keyevents(
    events: Iterable[Dict[str, Any]],
    correlation_window_seconds: float = DEFAULT_CORRELATION_WINDOW_SECONDS,
    session_end_time: Optional[str] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    bound_events, stats = bind_context_window_event_paths(
        events,
        correlation_window_seconds=correlation_window_seconds,
    )

    filtered = [event for event in bound_events if should_keep_keyevent(event)]

    seen = set()
    unique: List[Dict[str, Any]] = []
    for event in filtered:
        dedupe_key = (
            clean_text(event.get("timestamp", "")),
            clean_text(event.get("file_path", "")),
            clean_text(event.get("event_type", "")),
            clean_text(event.get("destination_path", "")),
        )
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        unique.append(event)

    unique.sort(key=lambda item: clean_text(item.get("timestamp", "")))

    stats["dropped_unbound_window_events"] = sum(
        1
        for event in bound_events
        if is_context_window_event(event.get("event_type", "")) and not clean_text(event.get("file_path", ""))
    )
    stats["deduplicated_events"] = len(filtered) - len(unique)
    unique = annotate_keyevent_end_times(
        unique,
        bound_events,
        session_end_time=session_end_time,
    )
    stats["final_events"] = len(unique)
    return unique, stats
