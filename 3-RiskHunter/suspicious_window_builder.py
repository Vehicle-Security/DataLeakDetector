"""Build VLM review windows from log evidence.

This module keeps the log layer in its lane: it finds time ranges and reasons
worth visual review, but it does not decide whether a selected/attached file was
actually sent. Completion evidence belongs to keyframes/VLM.
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta
from typing import Any, Dict, Iterable, List, Optional

from log_first_detector import file_hint_from_log, flatten_log_text, is_sensitive_name, normalize_path


AI_CONTEXT_TOKENS = (
    "chatgpt",
    "claude",
    "gemini",
    "deepseek",
    "kimi",
    "poe",
    "doubao",
    "tongyi",
    "yiyan",
    "copilot",
    " ai ",
    "\u4eba\u5de5\u667a\u80fd",
    "\u5927\u6a21\u578b",
)

EXFIL_REVIEW_TOKENS = (
    "file_selected",
    "file picker",
    "choose file",
    "selected file",
    "clipboard",
    "paste",
    "copy",
    "send",
    "share",
    "upload",
    "attach",
    "attachment",
    "compose",
    "email",
    "mail",
    "screenshot",
    "screen capture",
    "record",
    "recording",
    "screen share",
    "share screen",
    "meeting",
    "zip",
    "compress",
    "archive",
    "extract",
    "convert",
    "rename",
    "export",
    "dropbox",
    "onedrive",
    "google drive",
    "icloud drive",
    "usb",
    "removable",
    "\u7c98\u8d34",
    "\u590d\u5236",
    "\u53d1\u9001",
    "\u5206\u4eab",
    "\u4e0a\u4f20",
    "\u9644\u4ef6",
    "\u90ae\u7bb1",
    "\u5199\u4fe1",
    "\u622a\u56fe",
    "\u5f55\u5c4f",
    "\u5171\u4eab\u5c4f\u5e55",
    "\u5c4f\u5e55\u5171\u4eab",
    "\u4f1a\u8bae",
    "\u538b\u7f29",
    "\u89e3\u538b",
    "\u8f6c\u6362",
    "\u91cd\u547d\u540d",
    "\u5bfc\u51fa",
    "\u4e91\u76d8",
    "u\u76d8",
)

HIDDEN_REVIEW_EVENT_TYPES = {
    "created",
    "modified",
    "opened",
    "renamed",
    "moved",
    "copied",
    "compressed",
    "converted",
    "clipboard_text",
    "clipboard_image",
    "screenshot_capture",
    "screen_recording_started",
}

BENIGN_COMPLETION_TOKENS = (
    "cancel",
    "cancelled",
    "canceled",
    "discard",
    "draft",
    "\u53d6\u6d88",
    "\u653e\u5f03",
    "\u8349\u7a3f",
)


def _get_int_env(name: str, default: int, minimum: int = 0) -> int:
    try:
        return max(minimum, int(os.getenv(name, str(default))))
    except ValueError:
        return default


def parse_timestamp(timestamp: str) -> Optional[datetime]:
    if not timestamp:
        return None
    text = str(timestamp).strip().replace("Z", "").replace("T", " ")
    for fmt in ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue
    return None


def format_timestamp(value: datetime) -> str:
    return value.strftime("%Y-%m-%d %H:%M:%S")


def _contains_any(text: str, tokens: Iterable[str]) -> bool:
    lowered = f" {str(text or '').lower()} "
    return any(token.lower() in lowered for token in tokens)


def _app_name(log: Dict[str, Any]) -> str:
    return (
        str(log.get("app_name") or "")
        or str(log.get("process_info", {}).get("process_name") or "")
        or "unknown"
    )


def _is_whitelisted(log: Dict[str, Any], whitelist_apps: Iterable[str]) -> bool:
    text = f"{_app_name(log)} {log.get('process_info', {}).get('process_name', '')}".lower()
    return any(str(app).lower() in text for app in whitelist_apps if app)


def _is_sensitive_anchor(log: Dict[str, Any]) -> bool:
    upload_detection = log.get("upload_detection")
    if isinstance(upload_detection, dict) and upload_detection.get("sensitivity"):
        return True
    hint = file_hint_from_log(log)
    path = normalize_path(log.get("file_path", "") or hint)
    name = log.get("file_name") or path
    title = log.get("window_info", {}).get("window_title", "")
    content = log.get("content_preview", "")
    return bool(is_sensitive_name(f"{name} {hint} {title} {content}"))


def _is_review_signal(log: Dict[str, Any], whitelist_apps: Iterable[str]) -> bool:
    if _is_whitelisted(log, whitelist_apps):
        return False
    text = flatten_log_text(log)
    event_type = str(log.get("event_type", "")).lower()
    if event_type in HIDDEN_REVIEW_EVENT_TYPES and _is_sensitive_anchor(log):
        return True
    return _contains_any(text, AI_CONTEXT_TOKENS) or _contains_any(text, EXFIL_REVIEW_TOKENS)


def _merge_windows(windows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not windows:
        return []
    ordered = sorted(windows, key=lambda item: item["start_dt"])
    merged = [ordered[0]]
    for window in ordered[1:]:
        last = merged[-1]
        if window["start_dt"] <= last["end_dt"]:
            last["end_dt"] = max(last["end_dt"], window["end_dt"])
            last["reasons"] = sorted(set(last["reasons"]) | set(window["reasons"]))
            last["anchor_files"] = sorted(set(last["anchor_files"]) | set(window["anchor_files"]))
            last["candidate_events"].extend(window["candidate_events"])
            last["requires_completion_evidence"] = (
                last["requires_completion_evidence"] or window["requires_completion_evidence"]
            )
        else:
            merged.append(window)
    return merged


def build_analysis_windows(
    logs: List[Dict[str, Any]],
    log_first_result: Optional[Dict[str, Any]] = None,
    whitelist_apps: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    if isinstance(log_first_result, dict):
        meta = log_first_result.get("log_first", {})
        if isinstance(meta, dict) and int(meta.get("sensitive_events", 0) or 0) <= 0:
            return []
    if whitelist_apps is None and isinstance(log_first_result, dict):
        meta = log_first_result.get("log_first", {})
        if isinstance(meta, dict):
            whitelist_apps = list(meta.get("whitelist_apps", []) or [])
    whitelist_apps = whitelist_apps or []
    pre_seconds = _get_int_env("DLD_ANALYSIS_PRE_SECONDS", 8)
    post_seconds = _get_int_env("DLD_ANALYSIS_POST_SECONDS", 90, minimum=1)
    fallback_window = _get_int_env("DLD_VLM_FALLBACK_WINDOW_SEC", 300, minimum=1)
    correlation_seconds = _get_int_env("DLD_LOG_CORRELATION_SECONDS", fallback_window, minimum=1)

    parsed_logs = []
    sensitive_times: List[datetime] = []
    for log in logs:
        dt = parse_timestamp(log.get("timestamp", ""))
        if not dt:
            continue
        parsed_logs.append((dt, log))
        if _is_sensitive_anchor(log):
            sensitive_times.append(dt)

    if not sensitive_times and log_first_result:
        for record in log_first_result.get("operation_records", []):
            dt = parse_timestamp(record.get("operation_time", ""))
            if dt:
                sensitive_times.append(dt)

    if not sensitive_times:
        return []

    windows = []
    for dt, log in parsed_logs:
        if not _is_review_signal(log, whitelist_apps):
            continue
        nearest = min(abs((dt - anchor).total_seconds()) for anchor in sensitive_times)
        if nearest > correlation_seconds:
            continue

        text = flatten_log_text(log)
        reasons = []
        if _contains_any(text, AI_CONTEXT_TOKENS):
            reasons.append("ai_context_near_sensitive_log")
        if _contains_any(text, EXFIL_REVIEW_TOKENS):
            reasons.append("ambiguous_exfil_context_near_sensitive_log")
        if str(log.get("event_type", "")).lower() in HIDDEN_REVIEW_EVENT_TYPES and _is_sensitive_anchor(log):
            reasons.append("hidden_transform_or_content_capture_near_sensitive_log")
        if _contains_any(text, BENIGN_COMPLETION_TOKENS):
            reasons.append("possible_cancel_or_draft_completion")
        if not reasons:
            continue

        hint = normalize_path(log.get("file_path", "") or file_hint_from_log(log))
        windows.append(
            {
                "start_dt": dt - timedelta(seconds=pre_seconds),
                "end_dt": dt + timedelta(seconds=post_seconds),
                "anchor_files": [hint] if hint else [],
                "reasons": sorted(set(reasons)),
                "candidate_events": [
                    {
                        "timestamp": log.get("timestamp", ""),
                        "event_type": log.get("event_type", ""),
                        "app_name": _app_name(log),
                        "window_title": log.get("window_info", {}).get("window_title", ""),
                        "reason": ",".join(sorted(set(reasons))),
                    }
                ],
                "requires_completion_evidence": True,
            }
        )

    merged = _merge_windows(windows)
    for idx, window in enumerate(merged, 1):
        window["window_id"] = f"review_{idx}"
        window["start"] = format_timestamp(window.pop("start_dt"))
        window["end"] = format_timestamp(window.pop("end_dt"))
        window["candidate_events"] = window["candidate_events"][:12]
    return merged
