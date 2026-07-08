"""前端应用识别和未知风险提示。"""

from __future__ import annotations

from dataclasses import dataclass

from ..policy import APP_CATEGORY_RULES, APP_HINTS, RISKY_APP_CATEGORIES, SINK_TOKENS, contains_any, normalize_text


@dataclass(frozen=True)
class AppIdentity:
    app_name: str
    category: str
    known: bool
    risk_hint: str


def identify_frontend_app(app_name: str = "", window_title: str = "", ocr_text: str = "") -> AppIdentity:
    text = normalize_text(f"{app_name} {window_title} {ocr_text}")
    for hint, category in APP_HINTS.items():
        if hint in text:
            risk = "external_capable" if category in RISKY_APP_CATEGORIES else "local_or_benign"
            return AppIdentity(app_name=app_name or hint, category=category, known=True, risk_hint=risk)

    for category, tokens in APP_CATEGORY_RULES:
        if contains_any(text, tokens):
            risk = "external_capable_inferred" if category in RISKY_APP_CATEGORIES else "local_or_benign_inferred"
            return AppIdentity(app_name=app_name or _label_from_text(window_title, ocr_text), category=category, known=False, risk_hint=risk)

    if contains_any(text, SINK_TOKENS):
        return AppIdentity(app_name=app_name or "unknown", category="external_sink", known=False, risk_hint="unknown_external_sink")
    return AppIdentity(app_name=app_name or "unknown", category="unknown", known=False, risk_hint="unknown_app_near_sensitive_activity")


def _label_from_text(window_title: str, ocr_text: str) -> str:
    for candidate in (window_title, ocr_text):
        text = str(candidate or "").strip()
        if text:
            return text[:80]
    return "unknown"
