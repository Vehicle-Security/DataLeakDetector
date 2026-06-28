"""
Deterministic log-first upload detection.

This module is intentionally conservative: it creates upload alerts only when
the logs already contain enough evidence to connect a sensitive source file to
an upload event. VLM analysis can still be used as a fallback by callers.
"""

from __future__ import annotations

import os
from datetime import datetime
from typing import Any, Dict, List, Optional

from upload_detector_state import UploadEvent
from upload_detection_config import UploadDetectionConfig


SENSITIVE_KEYWORDS = [
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
]

OPEN_TYPES = {"opened", "file_open", "open"}
DERIVE_TYPES = {
    "created",
    "modified",
    "renamed",
    "copied",
    "moved",
    "compressed",
    "converted",
    "file_selected",
}
UPLOAD_TYPES = {"file_upload", "upload_detected"}


def normalize_path(path: str) -> str:
    normalized = str(path or "").strip().replace("\\", "/")
    while "//" in normalized:
        normalized = normalized.replace("//", "/")
    return normalized


def normalize_process(process_name: str) -> str:
    normalized = normalize_path(process_name)
    return normalized.rsplit("/", 1)[-1].lower() if normalized else "unknown"


def parse_timestamp_ms(timestamp: str) -> int:
    if not timestamp:
        return 0
    try:
        text = str(timestamp).replace("Z", "+00:00").replace(" ", "T")
        return int(datetime.fromisoformat(text).timestamp() * 1000)
    except Exception:
        return 0


def basename(path: str) -> str:
    return normalize_path(path).rsplit("/", 1)[-1]


def stem(path: str) -> str:
    return os.path.splitext(basename(path))[0].lower()


def file_key(path: str) -> str:
    return normalize_path(path).lower()


def is_sensitive_name(name: str) -> bool:
    return any(keyword in str(name or "") for keyword in SENSITIVE_KEYWORDS)


def is_upload_log(log: Dict[str, Any]) -> bool:
    event_type = str(log.get("event_type", ""))
    if event_type in UPLOAD_TYPES:
        return True
    if log.get("upload_detection", {}).get("is_upload"):
        return True
    window = str(log.get("window_info", {}).get("window_title", "")).lower()
    return any(token in window for token in ["upload", "\u4e0a\u4f20", "\u53d1\u9001", "\u9644\u4ef6"])


def process_name_from_log(log: Dict[str, Any]) -> str:
    return normalize_process(log.get("process_info", {}).get("process_name", "unknown"))


def app_name_from_log(log: Dict[str, Any]) -> str:
    return (
        log.get("app_name")
        or log.get("process_info", {}).get("process_name")
        or "unknown"
    )


class LogFirstDetector:
    def __init__(
        self,
        sensitive_files: List[str],
        blacklist_apps: List[str],
        whitelist_apps: List[str],
    ) -> None:
        self.config = UploadDetectionConfig()
        self.config.sensitive_files = list(sensitive_files)
        self.config.blacklist_apps = list(blacklist_apps)
        self.config.whitelist_apps = list(whitelist_apps)
        self.sensitive_file_keys = {file_key(path) for path in sensitive_files}
        self.sensitive_basenames = {basename(path).lower() for path in sensitive_files}

    def analyze(self, logs: List[Dict[str, Any]]) -> Dict[str, Any]:
        logs_by_time = sorted(logs, key=lambda item: parse_timestamp_ms(item.get("timestamp", "")))
        source_by_key: Dict[str, Dict[str, Any]] = {}
        mappings: Dict[str, str] = {}
        operation_records: List[Dict[str, Any]] = []
        upload_events: List[UploadEvent] = []
        seen_operations = set()
        seen_uploads = set()

        for log in logs_by_time:
            path = normalize_path(log.get("file_path", ""))
            if not path:
                continue

            if self._is_sensitive_path(path, log):
                root_path = mappings.get(file_key(path), path)
                source_by_key[file_key(path)] = {
                    "file_path": path,
                    "original_file": root_path,
                    "process_name": process_name_from_log(log),
                    "timestamp": log.get("timestamp", ""),
                    "log": log,
                }

                if log.get("event_type", "") in OPEN_TYPES:
                    self._append_operation(
                        operation_records,
                        seen_operations,
                        log,
                        path,
                        "log-open",
                        f"{process_name_from_log(log)} opened {basename(path)}",
                    )

            parent = self._find_parent_for_log(log, source_by_key, mappings)
            if parent:
                mappings[file_key(path)] = parent["original_file"]
                if file_key(path) not in source_by_key:
                    source_by_key[file_key(path)] = {
                        "file_path": path,
                        "original_file": parent["original_file"],
                        "process_name": process_name_from_log(log) or parent["process_name"],
                        "timestamp": log.get("timestamp", ""),
                        "log": log,
                    }

                if log.get("event_type", "") in DERIVE_TYPES:
                    self._append_operation(
                        operation_records,
                        seen_operations,
                        log,
                        path,
                        "log-transform",
                        f"{basename(parent['file_path'])} -> {basename(path)}",
                    )

            if is_upload_log(log):
                event = self._build_upload_event(log, source_by_key, mappings)
                if event:
                    dedup_key = (
                        file_key(event.upload_content or event.file_path),
                        normalize_process(event.app_name),
                        event.operation_type,
                        self._time_bucket(event.timestamp),
                    )
                    if dedup_key not in seen_uploads:
                        seen_uploads.add(dedup_key)
                        upload_events.append(event)
                        self._append_operation(
                            operation_records,
                            seen_operations,
                            log,
                            event.upload_content or event.file_path,
                            "log-upload",
                            event.description,
                            app_name=event.app_name,
                        )

        alert_events = [event for event in upload_events if event.should_alert]
        info_events = [event for event in upload_events if not event.should_alert]
        statistics = {
            "total_events_processed": len(logs_by_time),
            "upload_events_detected": len(upload_events),
            "blacklist_alerts": len(alert_events),
            "whitelist_uploads": sum(1 for event in upload_events if event.app_category == "whitelist"),
            "unknown_uploads": sum(1 for event in upload_events if event.app_category == "unknown"),
        }

        return {
            "alert_events": alert_events,
            "info_events": info_events,
            "upload_events": upload_events,
            "operation_records": operation_records,
            "statistics": statistics,
            "file_mappings": self._export_file_mappings(mappings, source_by_key),
            "log_first": {
                "used": True,
                "sensitive_events": len(source_by_key),
                "direct_mappings": len(mappings),
            },
        }

    def _is_sensitive_path(self, path: str, log: Dict[str, Any]) -> bool:
        path_key = file_key(path)
        name = log.get("file_name") or basename(path)
        return (
            path_key in self.sensitive_file_keys
            or basename(path).lower() in self.sensitive_basenames
            or is_sensitive_name(name)
        )

    def _find_parent_for_log(
        self,
        log: Dict[str, Any],
        source_by_key: Dict[str, Dict[str, Any]],
        mappings: Dict[str, str],
    ) -> Optional[Dict[str, Any]]:
        event_type = str(log.get("event_type", ""))
        path = normalize_path(log.get("file_path", ""))
        if not path or event_type not in DERIVE_TYPES | UPLOAD_TYPES:
            return None

        path_key = file_key(path)
        if path_key in mappings:
            original = mappings[path_key]
            return source_by_key.get(file_key(original))

        current_stem = stem(path)
        best_parent = None
        best_score = -1
        current_ts = parse_timestamp_ms(log.get("timestamp", ""))

        for candidate in source_by_key.values():
            parent_path = candidate["file_path"]
            parent_stem = stem(parent_path)
            if file_key(parent_path) == path_key:
                continue
            parent_ts = parse_timestamp_ms(candidate.get("timestamp", ""))
            if current_ts and parent_ts and parent_ts > current_ts:
                continue
            score = 0
            if current_stem.startswith(parent_stem) or parent_stem.startswith(current_stem):
                score += 4
            if is_sensitive_name(current_stem) and is_sensitive_name(parent_stem):
                score += 2
            if basename(parent_path).split(".")[0] in basename(path):
                score += 1
            if score > best_score:
                best_score = score
                best_parent = candidate

        return best_parent if best_score > 0 else None

    def _build_upload_event(
        self,
        log: Dict[str, Any],
        source_by_key: Dict[str, Dict[str, Any]],
        mappings: Dict[str, str],
    ) -> Optional[UploadEvent]:
        path = normalize_path(log.get("file_path", ""))
        if not path:
            return None

        path_key = file_key(path)
        original = mappings.get(path_key)
        source = source_by_key.get(path_key)
        if not source and original:
            source = source_by_key.get(file_key(original))
        if not source and self._is_sensitive_path(path, log):
            source = {
                "file_path": path,
                "original_file": original or path,
                "process_name": process_name_from_log(log),
                "timestamp": log.get("timestamp", ""),
                "log": log,
            }
        if not source:
            return None

        app_name = app_name_from_log(log)
        app_category = self.config.get_app_category(app_name)
        should_alert = app_category == "blacklist"
        alert_level = "critical" if should_alert else "info"
        window_title = log.get("window_info", {}).get("window_title", "")
        operation_type = "file_upload"
        behavior_category = "data_exfiltration"

        return UploadEvent(
            event_id=f"log_upload_{parse_timestamp_ms(log.get('timestamp', ''))}_{abs(hash(path_key))}",
            timestamp=log.get("timestamp", ""),
            file_path=path,
            file_name=basename(path),
            original_file=source.get("original_file") or path,
            upload_content=path,
            upload_content_mapping_link=self._mapping_chain(path, mappings),
            app_name=app_name,
            app_category=app_category,
            behavior_category=behavior_category,
            operation_type=operation_type,
            time_range=self._time_range(log.get("timestamp", "")),
            involved_timestamps=[log.get("timestamp", "")] if log.get("timestamp") else [],
            description=f"log evidence: {basename(path)} uploaded by {app_name}; window={window_title}",
            should_alert=should_alert,
            alert_level=alert_level,
            alert_reason=(
                f"blacklisted app {app_name} uploaded sensitive data"
                if should_alert
                else f"{app_name} uploaded sensitive data"
            ),
            extra_info={
                "source": "log_first",
                "confidence": 0.95,
                "log_event_type": log.get("event_type", ""),
                "window_title": window_title,
                "original_file": source.get("original_file") or path,
            },
        )

    def _append_operation(
        self,
        operation_records: List[Dict[str, Any]],
        seen_operations: set,
        log: Dict[str, Any],
        sensitive_file_path: str,
        operation: str,
        description: str,
        app_name: Optional[str] = None,
    ) -> None:
        record = {
            "operation_time": str(log.get("timestamp", "")).replace("T", " ").split(".")[0],
            "sensitive_file_path": normalize_path(sensitive_file_path),
            "app_name": app_name or app_name_from_log(log),
            "description": description,
            "operation": operation,
        }
        key = (
            record["operation_time"],
            file_key(record["sensitive_file_path"]),
            record["operation"],
            normalize_process(record["app_name"]),
        )
        if key in seen_operations:
            return
        seen_operations.add(key)
        operation_records.append(record)

    def _export_file_mappings(
        self,
        mappings: Dict[str, str],
        source_by_key: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Dict[str, str]]:
        direct = {
            source_by_key.get(child, {}).get("file_path", child): parent
            for child, parent in sorted(mappings.items())
            if child and parent and child != file_key(parent)
        }
        chains = {child: f"{normalize_path(parent)} -> {normalize_path(child)}" for child, parent in direct.items()}
        return {
            "direct_file_mappings": direct,
            "full_file_mapping_chains": chains,
        }

    def _mapping_chain(self, path: str, mappings: Dict[str, str]) -> str:
        current = file_key(path)
        parent = mappings.get(current)
        if not parent or file_key(parent) == current:
            return "none"
        return f"{normalize_path(parent)} -> {normalize_path(path)}"

    @staticmethod
    def _time_bucket(timestamp: str, seconds: int = 10) -> int:
        ts = parse_timestamp_ms(timestamp)
        return ts // (seconds * 1000) if ts else 0

    @staticmethod
    def _time_range(timestamp: str) -> str:
        text = str(timestamp or "").replace("T", " ").split(".")[0]
        return f"{text} - {text}" if text else ""
