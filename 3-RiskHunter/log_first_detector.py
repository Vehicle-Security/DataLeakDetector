"""
Deterministic log-first upload detection.

This module is intentionally conservative: it creates upload alerts only when
the logs already contain enough evidence to connect a sensitive source file to
an upload event. VLM analysis can still be used as a fallback by callers.
"""

from __future__ import annotations

import os
import re
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
}
SELECTION_TYPES = {"file_selected"}
NETWORK_UPLOAD_TYPES = {
    "http_post",
    "http_put",
    "network_upload",
    "web_upload",
    "cloud_upload",
    "api_upload",
}
UPLOAD_TYPES = {"file_upload", "upload_detected"} | NETWORK_UPLOAD_TYPES

CLOUD_SYNC_MARKERS = (
    "dropbox",
    "onedrive",
    "google drive",
    "googledrive",
    "icloud drive",
    "box sync",
    "baidunetdisk",
    "nutstore",
    "jianguoyun",
    "aliyundrive",
    "weiyun",
    "\u767e\u5ea6\u7f51\u76d8",
    "\u575a\u679c\u4e91",
    "\u963f\u91cc\u4e91\u76d8",
    "\u5fae\u4e91",
    "\u540c\u6b65",
    "\u4e91\u76d8",
)
REMOVABLE_MARKERS = (
    "removable",
    "usb",
    "thumb drive",
    "flash drive",
    "\u53ef\u79fb\u52a8",
    "\u79fb\u52a8\u78c1\u76d8",
    "u\u76d8",
)
NETWORK_UPLOAD_MARKERS = (
    "http://",
    "https://",
    " post ",
    " put ",
    "transfer.sh",
    "webdav",
)
EXPORT_CONTEXT_TOKENS = (
    "export",
    "save as",
    "saved as",
    "print to",
    "convert",
    "pdf",
    "\u5bfc\u51fa",
    "\u53e6\u5b58",
    "\u8f6c\u6362",
    "\u6253\u5370",
)
EXPORT_EXTENSIONS = {".pdf", ".csv", ".xlsx", ".xls", ".docx", ".doc", ".png", ".jpg", ".jpeg"}
FILE_PATH_PATTERN = re.compile(
    r"([A-Za-z]:[\\/][^\"'\r\n]+?\.[A-Za-z0-9]{1,8}|/[^\"'\r\n]+?\.[A-Za-z0-9]{1,8})"
)
SYSTEM_NOISE_PATH_MARKERS = (
    "/appdata/local/microsoft/edge/user data/",
    "/appdata/local/google/chrome/user data/",
    "/appdata/local/packages/microsoftwindows.client.cbs",
    "/appdata/local/microsoft/windows/",
    "/appdata/local/temp/",
    "/appdata/local/lenovo/slbrowser/",
    "/appdata/locallow/microsoft/cryptneturlcache/",
    "/appdata/roaming/microsoft/windows/recent/",
    "/appdata/roaming/tencent/",
    "/appdata/roaming/qqex/",
    "/appdata/roaming/qq/partitions/",
    "/appdata/roaming/baidu/",
    "/browserengine/users/",
    "/nt_db/",
    "/weblog/",
    "/log/radium/",
    "/windows/system32/",
    "/program files/",
    "/program files (x86)/",
)
SYSTEM_NOISE_BASENAMES = {
    "cookies",
    "cookies-journal",
    "quotamanager",
    "quotamanager-journal",
    "local state",
    "network persistent state",
    "preferences",
    "personalsetting.xml",
    "current",
    "lock",
    "log",
    "log.old",
}


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


def extension(path: str) -> str:
    return os.path.splitext(basename(path))[1].lower()


def flatten_log_text(log: Dict[str, Any]) -> str:
    parts = [
        log.get("event_type", ""),
        log.get("file_path", ""),
        log.get("file_name", ""),
        log.get("content_preview", ""),
        log.get("app_name", ""),
        log.get("process_info", {}).get("process_name", ""),
        log.get("window_info", {}).get("window_title", ""),
    ]
    upload_detection = log.get("upload_detection")
    if isinstance(upload_detection, dict):
        parts.extend(str(item) for item in upload_detection.values())
    return " ".join(str(part or "") for part in parts).lower()


def is_sensitive_name(name: str) -> bool:
    lowered = str(name or "").lower()
    return any(keyword in lowered for keyword in SENSITIVE_KEYWORDS)


def is_system_noise_path(path: str) -> bool:
    normalized = normalize_path(path).lower()
    if not normalized:
        return False
    base = basename(normalized)
    if base in SYSTEM_NOISE_BASENAMES:
        return True
    if any(marker in normalized for marker in SYSTEM_NOISE_PATH_MARKERS):
        return True
    return bool(re.search(r"/cache(_data)?/|/indexeddb/|/code cache/|/webstorage/|/network/", normalized))


def is_positive_upload_detection(upload_detection: Any) -> bool:
    if not isinstance(upload_detection, dict) or not upload_detection.get("is_upload"):
        return False
    upload_type = str(upload_detection.get("upload_type", "")).strip().lower()
    method = str(upload_detection.get("detection_method", "")).strip().lower()
    negative_markers = (
        "file access",
        "open dialog",
        "file picker",
        "read access",
        "download",
        "rename",
        "renamed",
        "modified",
        "local edit",
        "\u4e0b\u8f7d",
        "\u91cd\u547d\u540d",
        "\u4fee\u6539",
    )
    if any(marker in upload_type for marker in negative_markers):
        return False
    if any(marker in method for marker in negative_markers):
        return False
    return True


def _contains_external_marker(text: str, marker: str) -> bool:
    marker = marker.lower()
    if marker == "usb":
        return bool(re.search(r"(?<![a-z0-9])usb(?![a-z0-9])", text))
    return marker in text


def file_hint_from_log(log: Dict[str, Any]) -> str:
    for field in ("file_path", "file_name", "content_preview"):
        value = str(log.get(field, "") or "").strip().strip("\"'")
        if not value:
            continue
        if field == "file_path" and value:
            return normalize_path(value)
        match = FILE_PATH_PATTERN.search(value)
        if match:
            return normalize_path(match.group(1).strip().strip("\"'"))
        if field == "content_preview" and is_sensitive_name(value):
            return value
    return ""


def external_destination_reason(log: Dict[str, Any]) -> str:
    path = normalize_path(log.get("file_path", ""))
    text = flatten_log_text(log)
    if any(_contains_external_marker(text, marker) for marker in CLOUD_SYNC_MARKERS):
        return "cloud_sync"
    if any(_contains_external_marker(text, marker) for marker in REMOVABLE_MARKERS):
        return "removable_media"
    if re.match(r"^[e-z]:/", path.lower()):
        if process_name_from_log(log) in {"explorer.exe", "finder", "unknown"}:
            return "removable_media"
    return ""


def is_external_transfer_log(log: Dict[str, Any]) -> bool:
    event_type = str(log.get("event_type", "")).lower()
    write_like_types = DERIVE_TYPES | UPLOAD_TYPES | {"file_write", "file_create", "sync_upload"}
    return event_type in write_like_types and bool(external_destination_reason(log))


def is_network_upload_log(log: Dict[str, Any]) -> bool:
    event_type = str(log.get("event_type", "")).lower()
    if event_type in NETWORK_UPLOAD_TYPES:
        return True
    text = f" {flatten_log_text(log)} "
    return any(marker in text for marker in NETWORK_UPLOAD_MARKERS)


def is_upload_log(log: Dict[str, Any]) -> bool:
    event_type = str(log.get("event_type", "")).lower()
    if event_type in UPLOAD_TYPES:
        return True
    if is_positive_upload_detection(log.get("upload_detection")):
        return True
    return bool(is_network_upload_log(log) or is_external_transfer_log(log))


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
            hinted_path = file_hint_from_log(log)
            if not path and hinted_path and self._is_sensitive_path(hinted_path, {"file_name": basename(hinted_path)}):
                source_by_key[file_key(hinted_path)] = {
                    "file_path": hinted_path,
                    "original_file": hinted_path,
                    "process_name": process_name_from_log(log),
                    "timestamp": log.get("timestamp", ""),
                    "log": log,
                }
                self._append_operation(
                    operation_records,
                    seen_operations,
                    log,
                    hinted_path,
                    "log-sensitive-hint",
                    f"{process_name_from_log(log)} referenced {basename(hinted_path)}",
                )
            if not path:
                continue

            detected_original = self._upload_detection_original_file(log)
            if detected_original:
                mappings[file_key(path)] = detected_original
                source_by_key[file_key(detected_original)] = {
                    "file_path": detected_original,
                    "original_file": detected_original,
                    "process_name": process_name_from_log(log),
                    "timestamp": log.get("timestamp", ""),
                    "log": log,
                }
                source_by_key[file_key(path)] = {
                    "file_path": path,
                    "original_file": detected_original,
                    "process_name": process_name_from_log(log),
                    "timestamp": log.get("timestamp", ""),
                    "log": log,
                }

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
                "configured_sensitive_files": len(self.config.sensitive_files),
                "blacklist_apps": list(self.config.blacklist_apps),
                "whitelist_apps": list(self.config.whitelist_apps),
            },
        }

    def _is_sensitive_path(self, path: str, log: Dict[str, Any]) -> bool:
        path_key = file_key(path)
        name = log.get("file_name") or basename(path)
        if path_key in self.sensitive_file_keys or basename(path).lower() in self.sensitive_basenames:
            return True
        if is_system_noise_path(path):
            return False
        return is_sensitive_name(name)

    def _upload_detection_original_file(self, log: Dict[str, Any]) -> str:
        upload_detection = log.get("upload_detection")
        if not isinstance(upload_detection, dict):
            return ""
        original = normalize_path(upload_detection.get("original_file", ""))
        if not original:
            return ""
        if self._is_sensitive_path(original, {"file_name": basename(original)}):
            return original
        return ""

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
        if is_system_noise_path(path) and file_key(path) not in self.sensitive_file_keys:
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
            same_process = process_name_from_log(log) == candidate.get("process_name")
            close_in_time = current_ts and parent_ts and 0 <= current_ts - parent_ts <= 120_000
            if event_type in DERIVE_TYPES and same_process and close_in_time:
                score += 3
            export_window = current_ts and parent_ts and 0 <= current_ts - parent_ts <= 600_000
            if event_type in DERIVE_TYPES and same_process and export_window and self._is_export_like_log(log):
                score += 3
            if score > best_score:
                best_score = score
                best_parent = candidate

        return best_parent if best_score >= 3 else None

    @staticmethod
    def _is_export_like_log(log: Dict[str, Any]) -> bool:
        text = flatten_log_text(log)
        path = normalize_path(log.get("file_path", ""))
        return (
            any(token in text for token in EXPORT_CONTEXT_TOKENS)
            or extension(path) in EXPORT_EXTENSIONS
        )

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
        detected_original = self._upload_detection_original_file(log)
        if is_system_noise_path(path) and path_key not in self.sensitive_file_keys and not detected_original:
            return None

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
            if detected_original:
                source = {
                    "file_path": detected_original,
                    "original_file": detected_original,
                    "process_name": process_name_from_log(log),
                    "timestamp": log.get("timestamp", ""),
                    "log": log,
                }
                mappings[path_key] = detected_original
        if not source:
            return None

        app_name = app_name_from_log(log)
        app_category = self.config.get_app_category(app_name)
        operation_type = self._upload_operation_type(log)
        high_confidence_external = operation_type != "file_upload"
        should_alert = app_category == "blacklist" or (high_confidence_external and app_category != "whitelist")
        alert_level = "critical" if app_category == "blacklist" else ("warning" if should_alert else "info")
        window_title = log.get("window_info", {}).get("window_title", "")
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
                "external_channel": operation_type,
                "window_title": window_title,
                "original_file": source.get("original_file") or path,
            },
        )

    @staticmethod
    def _upload_operation_type(log: Dict[str, Any]) -> str:
        if is_external_transfer_log(log):
            external_reason = external_destination_reason(log)
            return external_reason
        if is_network_upload_log(log):
            return "network_upload"
        return "file_upload"

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
