"""Frontend application and visual-risk classification helpers.

The log-first layer should not decide final risk from a window title alone.
It can, however, label windows that deserve visual review when file-system logs
miss the sensitive artifact itself, such as screen sharing or VM copy flows.
"""

from __future__ import annotations

from typing import Any, Dict, List


AI_TOKENS = (
    "chatgpt",
    "chat.openai.com",
    "claude",
    "gemini",
    "deepseek",
    "kimi",
    "doubao",
    "tongyi",
    "yiyan",
    "yuanbao",
    "copilot",
    "llm",
    " ai ",
    "\u4eba\u5de5\u667a\u80fd",
    "\u5927\u6a21\u578b",
)

MEETING_TOKENS = (
    "teams",
    "zoom",
    "meeting",
    "webex",
    "tencent meeting",
    "voov",
    "\u4f1a\u8bae",
    "\u5f00\u4f1a",
    "\u901a\u8bdd",
    "\u817e\u8baf\u4f1a\u8bae",
)

SCREEN_SHARE_TOKENS = (
    "screen share",
    "share screen",
    "presenting",
    "presentation",
    "\u5171\u4eab\u5c4f\u5e55",
    "\u5c4f\u5e55\u5171\u4eab",
    "\u6b63\u5728\u5171\u4eab",
)

SCREEN_CAPTURE_TOKENS = (
    "screenshot",
    "screen capture",
    "snipping",
    "snipaste",
    "mspcmanager",
    "\u622a\u56fe",
    "\u622a\u5c4f",
    "\u5f55\u5c4f",
)

VM_TOKENS = (
    "vmware",
    "virtualbox",
    "hyper-v",
    "ubuntu - vmware",
    "openeuler",
    "virtual machine",
    "\u865a\u62df\u673a",
)

REMOTE_DESKTOP_TOKENS = (
    "remote desktop",
    "mstsc",
    "anydesk",
    "todesk",
    "sunlogin",
    "\u8fdc\u7a0b\u684c\u9762",
    "\u5411\u65e5\u8475",
)

EXTERNAL_APP_TOKENS = (
    "gmail",
    "outlook",
    "proton",
    "163",
    "qqmail",
    "mail",
    "dropbox",
    "onedrive",
    "google drive",
    "googledrive",
    "baidu",
    "weiyun",
    "github",
    "gitlab",
    "gitee",
    "bitbucket",
    "slack",
    "discord",
    "wechat",
    "weixin",
    "feishu",
    "lark",
    "dingtalk",
    "dingding",
    "qq",
    "\u90ae\u7bb1",
    "\u7f51\u76d8",
    "\u5fae\u4fe1",
    "\u98de\u4e66",
    "\u9489\u9489",
)


def _log_text(log: Dict[str, Any]) -> str:
    return " ".join(
        str(part or "")
        for part in (
            log.get("event_type", ""),
            log.get("app_name", ""),
            log.get("file_path", ""),
            log.get("file_name", ""),
            log.get("content_preview", ""),
            log.get("process_info", {}).get("process_name", ""),
            log.get("window_info", {}).get("window_title", ""),
        )
    ).lower()


def _contains_any(text: str, tokens: tuple[str, ...]) -> bool:
    padded = f" {text} "
    return any(token in padded for token in tokens)


def classify_log_context(log: Dict[str, Any]) -> Dict[str, Any]:
    text = _log_text(log)
    categories: List[str] = []

    checks = (
        ("ai_service", AI_TOKENS),
        ("meeting", MEETING_TOKENS),
        ("screen_share", SCREEN_SHARE_TOKENS),
        ("screen_capture", SCREEN_CAPTURE_TOKENS),
        ("virtual_machine", VM_TOKENS),
        ("remote_desktop", REMOTE_DESKTOP_TOKENS),
        ("external_app", EXTERNAL_APP_TOKENS),
    )
    for category, tokens in checks:
        if _contains_any(text, tokens):
            categories.append(category)

    event_type = str(log.get("event_type", "")).lower()
    if event_type in {"clipboard_image", "screenshot_capture", "screen_recording_started"}:
        if "screen_capture" not in categories:
            categories.append("screen_capture")

    visual_review = any(
        category in categories
        for category in ("meeting", "screen_share", "screen_capture", "virtual_machine", "remote_desktop")
    )

    return {
        "categories": categories,
        "visual_review": visual_review,
        "external_review": any(category in categories for category in ("external_app", "ai_service")),
    }


def is_no_anchor_visual_review_log(log: Dict[str, Any]) -> bool:
    return bool(classify_log_context(log).get("visual_review"))
