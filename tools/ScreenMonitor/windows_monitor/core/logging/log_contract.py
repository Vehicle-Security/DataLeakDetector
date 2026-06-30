# -*- coding: utf-8 -*-
"""
log_contract.py - Windows 日志字段契约辅助函数

收口 keyevents/logs 中的公共字段语义，尤其是：
- file_path 只表示事件涉及的完整文件路径
- process_info.process_path 只表示应用程序路径
- 非文件事件的 file_path 始终为空字符串
"""

import copy
import ntpath
import os
import re
from datetime import datetime
from typing import Any, Dict, Optional, Tuple


NON_FILE_EVENT_TYPES = {
    "app_switch",
    "website_visit",
    "clipboard_text",
    "clipboard_image",
    "clipboard_files",
    "clipboard_copy",
    "clipboard_paste",
    "manual_note",
    "inferred_upload",
    "window_closed",
}

WINDOWS_DRIVE_PATTERN = re.compile(r"^[A-Za-z]:[\\/]")
UNC_PATH_PATTERN = re.compile(r"^\\\\[^\\]+\\[^\\]+")


def clean_text(value: Any) -> str:
    """将任意值安全转换为去首尾空白的字符串。"""
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return str(value).strip()


def is_non_file_event(event_type: str) -> bool:
    """判断事件是否不应携带 file_path。"""
    normalized = clean_text(event_type)
    return normalized in NON_FILE_EVENT_TYPES or normalized.startswith("clipboard_")


def looks_like_full_path(path: str) -> bool:
    """判断字符串是否像完整文件路径，而不是 basename 或应用路径别名。"""
    text = clean_text(path)
    if not text:
        return False

    if WINDOWS_DRIVE_PATTERN.match(text) or UNC_PATH_PATTERN.match(text):
        return True

    # 兼容测试或跨平台示例中的 Unix 风格绝对路径
    return text.startswith("/")


def same_path(left: str, right: str) -> bool:
    """宽松比较两个路径是否指向同一路径。"""
    left_text = clean_text(left)
    right_text = clean_text(right)
    if not left_text or not right_text:
        return False

    left_norm = _norm_path(left_text)
    right_norm = _norm_path(right_text)
    return left_norm == right_norm


def normalize_timestamp_text(timestamp: str) -> str:
    """统一时间戳文本为 ISO8601 毫秒格式。"""
    text = clean_text(timestamp)
    if not text:
        return ""

    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S"):
        try:
            parsed = datetime.strptime(text, fmt)
            return parsed.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3]
        except ValueError:
            continue

    normalized = text.replace(" ", "T")
    if "." not in normalized:
        normalized += ".000"
    return normalized


def normalize_app_name(process_name: str) -> str:
    """规范化常见应用名称。"""
    name = clean_text(process_name)
    if not name:
        return ""

    if name.lower().endswith(".exe"):
        name = name[:-4]

    app_name_map = {
        "chrome": "Chrome",
        "msedge": "Edge",
        "firefox": "Firefox",
        "opera": "Opera",
        "explorer": "Explorer",
        "notepad": "记事本",
        "code": "VS Code",
        "wechat": "微信",
        "qq": "QQ",
        "wps": "WPS",
        "wpsoffice": "WPS",
        "et": "WPS Excel",
        "wpp": "WPS PPT",
        "wpsclouddrive": "WPS云盘",
        "dingtalk": "钉钉",
        "feishu": "飞书",
        "lark": "飞书",
    }
    return app_name_map.get(name.lower(), name)


def _normalize_dict(source: Any, keys: Tuple[str, ...]) -> Dict[str, str]:
    result: Dict[str, str] = {}
    if not isinstance(source, dict):
        source = {}

    for key in keys:
        result[key] = clean_text(source.get(key, ""))
    return result


def _is_windows_style_path(path: str) -> bool:
    text = clean_text(path)
    return bool(WINDOWS_DRIVE_PATTERN.match(text) or UNC_PATH_PATTERN.match(text) or "\\" in text)


def _basename(path: str) -> str:
    text = clean_text(path)
    return ntpath.basename(text) if _is_windows_style_path(text) else os.path.basename(text)


def _splitext(path: str) -> Tuple[str, str]:
    text = clean_text(path)
    return ntpath.splitext(text) if _is_windows_style_path(text) else os.path.splitext(text)


def _norm_path(path: str) -> str:
    text = clean_text(path)
    if _is_windows_style_path(text):
        return ntpath.normcase(ntpath.normpath(text))
    return os.path.normcase(os.path.normpath(text))


def _parse_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _select_authoritative_file_path(entry: Dict[str, Any], process_path: str) -> str:
    candidates = [
        entry.get("file_path", ""),
    ]

    upload_detection = entry.get("upload_detection", {})
    if isinstance(upload_detection, dict):
        candidates.append(upload_detection.get("original_file", ""))

    extra = entry.get("extra", {})
    if isinstance(extra, dict):
        candidates.append(extra.get("original_file", ""))

    for candidate in candidates:
        path = clean_text(candidate)
        if not path:
            continue
        if same_path(path, process_path):
            continue
        if looks_like_full_path(path):
            return path

    return ""


def _normalize_destination_fields(entry: Dict[str, Any], process_path: str) -> None:
    destination_path = clean_text(entry.get("destination_path", ""))
    if same_path(destination_path, process_path) or not looks_like_full_path(destination_path):
        entry.pop("destination_path", None)
        entry.pop("destination_name", None)
        entry.pop("destination_extension", None)
        return

    entry["destination_path"] = destination_path
    entry["destination_name"] = _basename(destination_path.rstrip("\\/"))
    entry["destination_extension"] = _splitext(entry["destination_name"])[1]


def normalize_event_entry(
    entry: Dict[str, Any],
    *,
    drop_invalid_file_event: bool = False,
) -> Optional[Dict[str, Any]]:
    """
    归一化统一日志条目。

    Args:
        entry: 原始事件
        drop_invalid_file_event: 当文件事件无法修复 file_path 时返回 None
    """
    normalized = copy.deepcopy(entry or {})

    normalized["timestamp"] = clean_text(normalized.get("timestamp", ""))
    normalized["event_type"] = clean_text(normalized.get("event_type", "")) or "unknown"
    normalized["app_name"] = clean_text(normalized.get("app_name", ""))
    normalized["file_size"] = _parse_int(normalized.get("file_size", 0))

    normalized["process_info"] = _normalize_dict(
        normalized.get("process_info"),
        ("pid", "process_name", "process_path", "cmdline"),
    )
    normalized["window_info"] = _normalize_dict(
        normalized.get("window_info"),
        ("window_handle", "window_title", "window_class"),
    )
    normalized["user_info"] = _normalize_dict(
        normalized.get("user_info"),
        ("username", "hostname"),
    )
    normalized["disk_info"] = _normalize_dict(
        normalized.get("disk_info"),
        ("drive_letter", "disk_type"),
    )

    if not isinstance(normalized.get("extra"), dict):
        normalized["extra"] = {}

    if isinstance(normalized.get("upload_detection"), dict):
        normalized["upload_detection"] = copy.deepcopy(normalized["upload_detection"])

    process_path = normalized["process_info"].get("process_path", "")

    if is_non_file_event(normalized["event_type"]):
        normalized["file_path"] = ""
        normalized["file_name"] = ""
        normalized["file_extension"] = ""
        normalized["file_size"] = 0
        normalized.pop("destination_path", None)
        normalized.pop("destination_name", None)
        normalized.pop("destination_extension", None)
        return normalized

    authoritative_path = _select_authoritative_file_path(normalized, process_path)
    if authoritative_path:
        normalized["file_path"] = authoritative_path
        normalized["file_name"] = _basename(authoritative_path.rstrip("\\/"))
        normalized["file_extension"] = _splitext(normalized["file_name"])[1]
        if not normalized["disk_info"].get("drive_letter") and len(authoritative_path) >= 2 and authoritative_path[1] == ":":
            normalized["disk_info"]["drive_letter"] = authoritative_path[:2]

        upload_detection = normalized.get("upload_detection")
        if isinstance(upload_detection, dict):
            upload_detection["original_file"] = authoritative_path
    else:
        if drop_invalid_file_event:
            return None
        normalized["file_path"] = ""
        normalized["file_name"] = ""
        normalized["file_extension"] = ""
        normalized["file_size"] = 0

    _normalize_destination_fields(normalized, process_path)
    return normalized


def build_browser_file_access_event(
    *,
    raw_timestamp: str,
    process_name: str,
    pid: Any,
    file_path: str,
    username: str,
    hostname: str,
) -> Optional[Dict[str, Any]]:
    """构建统一形态的浏览器文件访问事件。"""
    path = clean_text(file_path)
    if not looks_like_full_path(path):
        return None

    event = {
        "timestamp": normalize_timestamp_text(raw_timestamp),
        "event_type": "created",
        "file_path": path,
        "file_name": _basename(path),
        "file_size": 0,
        "file_extension": _splitext(path)[1],
        "process_info": {
            "pid": clean_text(pid),
            "process_name": clean_text(process_name),
            "process_path": "",
            "cmdline": "",
        },
        "window_info": {
            "window_handle": "",
            "window_title": "",
            "window_class": "",
        },
        "user_info": {
            "username": clean_text(username),
            "hostname": clean_text(hostname),
        },
        "disk_info": {
            "drive_letter": path[:2] if len(path) >= 2 and path[1] == ":" else "",
            "disk_type": "Fixed",
        },
        "app_name": normalize_app_name(process_name),
        "extra": {
            "raw_operation": "browser_file_access",
            "category": "浏览器文件访问",
            "source": "etw_monitor",
        },
    }
    return normalize_event_entry(event, drop_invalid_file_event=True)
