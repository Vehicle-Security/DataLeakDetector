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


def identify_frontend_app(app_name: str = "", window_title: str = "", visual_text: str = "") -> AppIdentity:
    # Window/visual text identifies the actual frontend. Process names such as
    # Chrome and Edge are only wrappers and must not hide Outlook, ChatGPT, or
    # another sink named in the title.
    sources = tuple(
        text
        for source in (visual_text, window_title, app_name)
        if (text := normalize_text(source))
    )
    for hint, category in _ordered_app_hints(include_generic=False):
        if any(hint in text for text in sources):
            risk = "external_capable" if category in RISKY_APP_CATEGORIES else "local_or_benign"
            return AppIdentity(app_name=hint, category=category, known=True, risk_hint=risk)

    for text in sources:
        for category, tokens in APP_CATEGORY_RULES:
            if category in {"browser", "document_editor"}:
                continue
            if contains_any(text, tokens):
                risk = "external_capable_inferred" if category in RISKY_APP_CATEGORIES else "local_or_benign_inferred"
                return AppIdentity(
                    app_name=_label_from_text(window_title, visual_text) or app_name or "unknown",
                    category=category,
                    known=False,
                    risk_hint=risk,
                )

    for hint, category in _ordered_app_hints(include_generic=True):
        if any(hint in text for text in sources):
            risk = "external_capable" if category in RISKY_APP_CATEGORIES else "local_or_benign"
            return AppIdentity(app_name=hint, category=category, known=True, risk_hint=risk)

    for text in sources:
        for category, tokens in APP_CATEGORY_RULES:
            if contains_any(text, tokens):
                risk = "external_capable_inferred" if category in RISKY_APP_CATEGORIES else "local_or_benign_inferred"
                return AppIdentity(
                    app_name=_label_from_text(window_title, visual_text) or app_name or "unknown",
                    category=category,
                    known=False,
                    risk_hint=risk,
                )

    text = normalize_text(f"{visual_text} {window_title} {app_name}")

    if contains_any(text, SINK_TOKENS):
        return AppIdentity(app_name=app_name or "unknown", category="external_sink", known=False, risk_hint="unknown_external_sink")
    return AppIdentity(app_name=app_name or "unknown", category="unknown", known=False, risk_hint="unknown_app_near_sensitive_activity")


def _ordered_app_hints(*, include_generic: bool) -> tuple[tuple[str, str], ...]:
    """Prefer specific product hints over generic browser/editor wrappers."""

    generic = {"chrome", "edge", "firefox", "safari", "browser", "word", "wps", "notepad", "excel"}
    hints = (item for item in APP_HINTS.items() if (item[0] in generic) == include_generic)
    return tuple(sorted(hints, key=lambda item: (-len(item[0]), item[0])))


def _label_from_text(window_title: str, visual_text: str) -> str:
    for candidate in (window_title, visual_text):
        text = str(candidate or "").strip()
        if text:
            return text[:80]
    return "unknown"
