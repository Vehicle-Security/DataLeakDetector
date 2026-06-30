from __future__ import annotations

import re
from typing import Any


BROWSER_TOKENS = (
    "chrome",
    "msedge",
    "edge",
    "firefox",
    "safari",
    "browser",
    "chrome_widgetwin",
)

FRONTEND_CATEGORY_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "email",
        (
            "gmail",
            "outlook",
            "proton",
            "qqmail",
            "mail.qq.com",
            "mail.163.com",
            "163.com",
            "mail",
            "\u90ae\u7bb1",
            "\u90ae\u4ef6",
        ),
    ),
    (
        "ai_service",
        (
            "chatgpt",
            "chat.openai.com",
            "claude",
            "gemini",
            "deepseek",
            "kimi",
            "poe",
            "doubao",
            "tongyi",
            "yuanbao",
            "copilot",
            " ai ",
            "\u4eba\u5de5\u667a\u80fd",
            "\u5927\u6a21\u578b",
        ),
    ),
    (
        "cloud_storage",
        (
            "drive.google.com",
            "google drive",
            "onedrive",
            "dropbox",
            "icloud drive",
            "baidu",
            "weiyun",
            "quark",
            "kuake",
            "\u7f51\u76d8",
            "\u4e91\u76d8",
            "\u5fae\u4e91",
        ),
    ),
    (
        "code_repo",
        (
            "github",
            "gitlab",
            "gitee",
            "gitcode",
            "bitbucket",
        ),
    ),
    (
        "messaging",
        (
            "slack",
            "discord",
            "wechat",
            "weixin",
            "qq",
            "feishu",
            "lark",
            "dingtalk",
            "dingding",
            "\u5fae\u4fe1",
            "\u98de\u4e66",
            "\u9489\u9489",
            "\u804a\u5929",
        ),
    ),
    (
        "meeting",
        (
            "teams",
            "zoom",
            "meeting",
            "webex",
            "tencent meeting",
            "voov",
            "\u4f1a\u8bae",
            "\u817e\u8baf\u4f1a\u8bae",
        ),
    ),
    (
        "workplace",
        (
            "notion",
            "yuque",
            "google docs",
            "google sheets",
            "workspace",
            "\u8bed\u96c0",
            "\u6587\u6863",
        ),
    ),
)

EXTERNAL_FRONTEND_CATEGORIES = {
    "email",
    "ai_service",
    "cloud_storage",
    "code_repo",
    "messaging",
    "meeting",
    "workplace",
}

VISUAL_REVIEW_CATEGORIES = {"meeting"}

COMPLETION_TERMS = (
    "sent",
    "send successfully",
    "uploaded",
    "upload complete",
    "submitted",
    "\u5df2\u53d1\u9001",
    "\u53d1\u9001\u6210\u529f",
    "\u4e0a\u4f20\u5b8c\u6210",
    "\u63d0\u4ea4\u6210\u529f",
)


def _flatten(value: Any) -> str:
    if isinstance(value, dict):
        return " ".join(_flatten(item) for item in value.values())
    if isinstance(value, list):
        return " ".join(_flatten(item) for item in value)
    return str(value or "")


def _contains_token(text: str, token: str) -> bool:
    token = token.lower()
    if token.strip() == "ai":
        return bool(re.search(r"(?<![a-z0-9])ai(?![a-z0-9])", text))
    return token in text


def _contains_any(text: str, tokens: tuple[str, ...]) -> bool:
    return any(_contains_token(text, token) for token in tokens)


def classify_frontend_app(log: dict[str, Any]) -> dict[str, Any]:
    window_info = log.get("window_info", {}) or {}
    process_info = log.get("process_info", {}) or {}
    title = str(window_info.get("window_title", "") or "")
    url = str(log.get("url", "") or log.get("page_url", "") or "")
    process_name = str(process_info.get("process_name", "") or log.get("app_name", "") or "")
    text = f" {title} {url} {process_name} {_flatten(log.get('browser', {}))} ".lower()

    categories: list[str] = []
    for category, tokens in FRONTEND_CATEGORY_RULES:
        if _contains_any(text, tokens):
            categories.append(category)

    is_browser = _contains_any(text, BROWSER_TOKENS) or bool(url)
    primary_category = categories[0] if categories else ("browser" if is_browser else "desktop_app")
    display_name = primary_category
    if categories:
        display_name = f"{primary_category}:{_short_title(title) or process_name or 'unknown'}"
    elif process_name:
        display_name = process_name

    return {
        "category": primary_category,
        "categories": categories,
        "display_name": display_name,
        "window_title": title,
        "url": url,
        "is_browser": is_browser,
        "is_external": primary_category in EXTERNAL_FRONTEND_CATEGORIES,
        "visual_review": primary_category in VISUAL_REVIEW_CATEGORIES,
        "completion_hint": _contains_any(text, COMPLETION_TERMS),
    }


def _short_title(title: str) -> str:
    text = str(title or "").strip()
    if not text:
        return ""
    for separator in (" - ", " | ", " \u2014 "):
        if separator in text:
            text = text.split(separator, 1)[0]
            break
    return text[:80].strip()
