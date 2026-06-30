from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

from .frontend import classify_frontend_app
from .schema import CorrelatorContext, NormalizedLogEvent
from .utils import get_path_basename, normalize_file_path


SENSITIVE_KEYWORDS = (
    "\u85aa\u8d44",
    "\u5de5\u8d44",
    "\u673a\u5bc6",
    "\u7edd\u5bc6",
    "\u5408\u540c",
    "\u8d22\u52a1",
    "\u5ba2\u6237",
    "\u5bc6\u7801",
    "\u6838\u5fc3",
    "\u79d8\u5bc6",
    "\u5185\u90e8",
    "\u62a5\u8868",
    "\u9884\u7b97",
    "\u6218\u7565",
    "\u89c4\u5212",
    "\u4f1a\u8bae\u7eaa\u8981",
    "\u5458\u5de5",
    "confidential",
    "salary",
    "payroll",
    "contract",
    "customer",
    "client",
    "finance",
    "budget",
    "strategy",
    "roadmap",
)

DEFAULT_WHITELIST_APPS = {
    "system",
    "idle",
    "svchost.exe",
    "winmonitor",
    "winmonitor.exe",
    "python",
    "python.exe",
    "python3.11",
    "python3.11.exe",
}


@dataclass
class SensitiveWindowAccumulator:
    sensitive_file: str
    match_types: set[str] = field(default_factory=set)
    start_dt: datetime | None = None
    end_dt: datetime | None = None
    anchor_events: list[dict[str, Any]] = field(default_factory=list)
    cooccur_apps: set[str] = field(default_factory=set)
    frontend_categories: set[str] = field(default_factory=set)
    candidate_events: list[dict[str, Any]] = field(default_factory=list)

    def include_anchor(self, log_event: NormalizedLogEvent, match_types: set[str]) -> None:
        self.match_types.update(match_types)
        self._include_time(log_event.timestamp)
        self.anchor_events.append(_candidate_event(log_event, reason="sensitive_anchor"))

    def include_related(self, log_event: NormalizedLogEvent, reason: str) -> None:
        self._include_time(log_event.timestamp)
        frontend = classify_frontend_app(log_event.raw)
        if frontend.get("is_external"):
            self.frontend_categories.update(frontend.get("categories", []) or [frontend.get("category", "")])
        app_name = str(frontend.get("display_name") or log_event.app_name or log_event.process_name)
        if app_name:
            self.cooccur_apps.add(app_name)
        self.candidate_events.append(_candidate_event(log_event, reason=reason, frontend=frontend))

    def _include_time(self, value: datetime) -> None:
        self.start_dt = value if self.start_dt is None else min(self.start_dt, value)
        self.end_dt = value if self.end_dt is None else max(self.end_dt, value)


def build_sensitive_windows(
    context: CorrelatorContext,
    correlation_config: dict[str, Any],
) -> list[dict[str, Any]]:
    post_buffer = int(correlation_config.get("post_buffer_seconds", 10) or 10)
    external_followup_seconds = int(correlation_config.get("external_followup_seconds", 120) or 120)
    cooccur_seconds = int(correlation_config.get("cooccur_seconds", 15) or 15)
    whitelist_apps = {
        str(item or "").strip().lower()
        for item in correlation_config.get("whitelist_apps", []) or []
        if str(item or "").strip()
    } | DEFAULT_WHITELIST_APPS

    anchors: dict[str, SensitiveWindowAccumulator] = {}
    for log_event in context.normalized_logs:
        sensitive_file, match_types = _match_sensitive_file(log_event, context.sensitive_files)
        if not sensitive_file:
            continue
        key = normalize_file_path(sensitive_file).lower()
        accumulator = anchors.setdefault(key, SensitiveWindowAccumulator(sensitive_file=sensitive_file))
        accumulator.include_anchor(log_event, match_types)

    if not anchors:
        return []

    for accumulator in anchors.values():
        if accumulator.start_dt is None or accumulator.end_dt is None:
            continue
        cooccur_start = accumulator.start_dt - timedelta(seconds=cooccur_seconds)
        followup_end = accumulator.end_dt + timedelta(seconds=external_followup_seconds)
        for log_event in context.normalized_logs:
            if log_event.timestamp < cooccur_start or log_event.timestamp > followup_end:
                continue
            if _is_whitelisted(log_event, whitelist_apps):
                continue

            frontend = classify_frontend_app(log_event.raw)
            within_anchor = log_event.timestamp <= accumulator.end_dt + timedelta(seconds=cooccur_seconds)
            if within_anchor:
                accumulator.include_related(log_event, reason="cooccur_non_whitelist_app")
            elif frontend.get("is_external") or frontend.get("visual_review"):
                accumulator.include_related(log_event, reason="external_frontend_followup")

    windows: list[dict[str, Any]] = []
    for index, accumulator in enumerate(sorted(anchors.values(), key=lambda item: item.start_dt or datetime.min), 1):
        if accumulator.start_dt is None or accumulator.end_dt is None:
            continue
        end_dt = accumulator.end_dt + timedelta(seconds=post_buffer)
        windows.append(
            {
                "window_id": f"sensitive_window_{index}",
                "sensitive_file": accumulator.sensitive_file,
                "start": accumulator.start_dt.strftime("%Y-%m-%d %H:%M:%S"),
                "end": end_dt.strftime("%Y-%m-%d %H:%M:%S"),
                "match_types": sorted(accumulator.match_types),
                "cooccur_apps": sorted(accumulator.cooccur_apps),
                "frontend_categories": sorted(item for item in accumulator.frontend_categories if item),
                "candidate_events": (accumulator.anchor_events + accumulator.candidate_events)[:24],
                "post_buffer_seconds": post_buffer,
            }
        )
    return windows


def _match_sensitive_file(log_event: NormalizedLogEvent, sensitive_files: list[str]) -> tuple[str, set[str]]:
    path = normalize_file_path(log_event.file_path)
    path_key = path.lower()
    basename = get_path_basename(path).lower()
    text = " ".join(
        str(part or "")
        for part in (
            log_event.file_path,
            log_event.file_name,
            log_event.raw.get("content_preview", ""),
            log_event.raw.get("window_info", {}).get("window_title", ""),
        )
    ).lower()
    compact_text = _compact(text)

    for sensitive_file in sensitive_files:
        normalized = normalize_file_path(sensitive_file)
        sensitive_key = normalized.lower()
        sensitive_base = get_path_basename(normalized).lower()
        sensitive_stem = sensitive_base.rsplit(".", 1)[0]
        match_types: set[str] = set()

        if path_key and path_key == sensitive_key:
            match_types.add("exact_path")
        if basename and basename == sensitive_base:
            match_types.add("filename")
        if sensitive_key and sensitive_key in text:
            match_types.add("window_title")
        if sensitive_stem and len(sensitive_stem) >= 4 and _compact(sensitive_stem) in compact_text:
            match_types.add("window_title")
        if path_key and _under_sensitive_stem_dir(path_key, sensitive_stem):
            match_types.add("derived_under_sensitive_stem_dir")
        if any(keyword in text for keyword in SENSITIVE_KEYWORDS):
            match_types.add("keyword")

        if match_types:
            return normalized, match_types

    if any(keyword in text for keyword in SENSITIVE_KEYWORDS):
        return path or get_path_basename(text), {"keyword"}
    return "", set()


def _under_sensitive_stem_dir(path_key: str, sensitive_stem: str) -> bool:
    if not path_key or not sensitive_stem:
        return False
    parts = [part.lower() for part in path_key.split("/") if part]
    return any(sensitive_stem in part for part in parts[:-1])


def _compact(text: str) -> str:
    return "".join(ch for ch in str(text or "").lower() if ch.isalnum() or "\u4e00" <= ch <= "\u9fff")


def _is_whitelisted(log_event: NormalizedLogEvent, whitelist_apps: set[str]) -> bool:
    text = f"{log_event.app_name} {log_event.process_name}".lower()
    return any(app and app in text for app in whitelist_apps)


def _candidate_event(
    log_event: NormalizedLogEvent,
    reason: str,
    frontend: dict[str, Any] | None = None,
) -> dict[str, Any]:
    frontend = frontend if frontend is not None else classify_frontend_app(log_event.raw)
    return {
        "timestamp": log_event.timestamp_text,
        "event_type": log_event.event_type,
        "file_path": log_event.file_path,
        "app_name": log_event.app_name,
        "frontend_app": frontend,
        "reason": reason,
    }
