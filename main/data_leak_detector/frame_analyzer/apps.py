"""Frontend application recognition and unknown-risk helpers."""

from __future__ import annotations

from dataclasses import dataclass

from ..policy import SINK_TOKENS, contains_any

KNOWN_APP_HINTS = {
    "excel": "document_editor",
    "word": "document_editor",
    "wps": "document_editor",
    "notepad": "document_editor",
    "chrome": "browser",
    "edge": "browser",
    "firefox": "browser",
    "safari": "browser",
    "chatgpt": "ai_chat",
    "claude": "ai_chat",
    "gemini": "ai_chat",
    "kimi": "ai_chat",
    "deepseek": "ai_chat",
    "qwen": "ai_chat",
    "wechat": "chat",
    "qq": "chat",
    "feishu": "chat",
    "lark": "chat",
    "dingtalk": "chat",
    "gmail": "mail",
    "outlook": "mail",
    "163": "mail",
    "onedrive": "cloud_drive",
    "dropbox": "cloud_drive",
    "google drive": "cloud_drive",
    "baidu": "cloud_drive",
    "zoom": "meeting",
    "teams": "meeting",
}

RISKY_APP_CATEGORIES = {"browser", "ai_chat", "chat", "mail", "cloud_drive", "meeting"}


@dataclass(frozen=True)
class AppIdentity:
    app_name: str
    category: str
    known: bool
    risk_hint: str


def identify_frontend_app(app_name: str = "", window_title: str = "", ocr_text: str = "") -> AppIdentity:
    text = f"{app_name} {window_title} {ocr_text}".lower()
    for hint, category in KNOWN_APP_HINTS.items():
        if hint in text:
            risk = "external_capable" if category in RISKY_APP_CATEGORIES else "local_or_benign"
            return AppIdentity(app_name=app_name or hint, category=category, known=True, risk_hint=risk)
    if contains_any(text, SINK_TOKENS):
        return AppIdentity(app_name=app_name or "unknown", category="external_sink", known=False, risk_hint="unknown_external_sink")
    return AppIdentity(app_name=app_name or "unknown", category="unknown", known=False, risk_hint="unknown_app_near_sensitive_activity")
