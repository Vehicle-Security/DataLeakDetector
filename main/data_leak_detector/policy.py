"""策略配置加载与文本语义归一化。

项目的主要判断证据来自日志、OCR 和 VLM/LLM 结构化输出。这里不把业务词表写死
成代码逻辑，而是从 `spec/config/policy.json` 加载可替换策略；代码只保留最小兜底，
负责统一大小写、全半角、空白和分类接口。
"""

from __future__ import annotations

import json
import os
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_POLICY_PATH = REPO_ROOT / "spec" / "config" / "policy.json"

FALLBACK_POLICY: dict[str, Any] = {
    "sensitive_tokens": ("confidential", "secret", "salary", "customer", "机密", "工资", "客户"),
    "transfer_tokens": ("copy", "paste", "export", "screenshot", "复制", "粘贴", "导出", "截图"),
    "sink_tokens": ("upload", "send", "share", "email", "chatgpt", "上传", "发送", "分享", "邮件"),
    "normal_activity_tokens": ("normal", "reading", "正常", "阅读", "浏览"),
    "unknown_risk_tokens": ("外发", "泄露", "隐藏", "截图", "录屏", "共享", "粘贴"),
    "sink_classification": [
        {"type": "mail_attachment", "tokens": ("mail", "email", "attachment", "attach", "邮件", "附件")},
        {"type": "cloud_sync", "tokens": ("cloud", "drive", "网盘", "云盘")},
        {"type": "chat_upload", "tokens": ("wechat", "qq", "chatgpt", "chat", "微信")},
        {"type": "removable_media", "tokens": ("usb", "removable", "可移动")},
        {"type": "screen_share", "tokens": ("screen", "share", "共享", "屏幕共享")},
    ],
    "risk_levels": [
        {"level": "completed", "tokens": ("complete", "sent", "success", "已发送", "完成", "成功")},
        {"level": "content_exposed", "tokens": ("visible", "paste", "content", "粘贴", "可见", "共享")},
        {"level": "selected_or_attached", "tokens": ("upload", "send", "attach", "选择", "附件", "上传", "发送")},
    ],
    "semantic_sensitive_aliases": {
        "薪资": ("薪酬", "工资", "月薪"),
        "工资": ("薪资", "薪酬", "月薪"),
        "预算": ("成本", "财务", "budget"),
        "账号": ("账户", "口令", "密码", "credential"),
    },
    "frontend_app_hints": {
        "excel": "document_editor",
        "word": "document_editor",
        "chrome": "browser",
        "edge": "browser",
        "chatgpt": "ai_chat",
        "wechat": "chat",
        "qq": "chat",
        "gmail": "mail",
        "outlook": "mail",
        "onedrive": "cloud_drive",
        "zoom": "meeting",
    },
    "frontend_category_rules": [
        {"category": "ai_chat", "tokens": ("prompt", "assistant", "chatbot", "大模型", "智能助手")},
        {"category": "mail", "tokens": ("mail", "email", "inbox", "compose", "attachment", "邮箱", "收件箱", "附件")},
        {"category": "cloud_drive", "tokens": ("cloud drive", "drive", "dropbox", "网盘", "云盘", "文件上传")},
        {"category": "chat", "tokens": ("chat", "message", "messenger", "聊天", "群聊", "会话")},
        {"category": "meeting", "tokens": ("meeting", "screen share", "屏幕共享", "会议")},
        {"category": "browser", "tokens": ("browser", "chrome_widgetwin", "网页", "浏览器")},
        {"category": "document_editor", "tokens": ("document", "spreadsheet", "office", "文档", "表格")},
    ],
    "risky_app_categories": ("browser", "ai_chat", "chat", "mail", "cloud_drive", "meeting"),
}


@dataclass(frozen=True)
class PolicyConfig:
    sensitive_tokens: tuple[str, ...]
    transfer_tokens: tuple[str, ...]
    sink_tokens: tuple[str, ...]
    normal_activity_tokens: tuple[str, ...]
    unknown_risk_tokens: tuple[str, ...]
    sink_classification: tuple[tuple[str, tuple[str, ...]], ...]
    risk_levels: tuple[tuple[str, tuple[str, ...]], ...]
    semantic_sensitive_aliases: dict[str, tuple[str, ...]]
    frontend_app_hints: dict[str, str]
    frontend_category_rules: tuple[tuple[str, tuple[str, ...]], ...]
    risky_app_categories: frozenset[str]


def load_policy_config(path: str | Path | None = None) -> PolicyConfig:
    """加载策略配置；配置缺失时使用最小兜底，避免流水线直接不可用。"""

    config_path = Path(path or os.getenv("DLD_POLICY_CONFIG") or DEFAULT_POLICY_PATH)
    raw = dict(FALLBACK_POLICY)
    if config_path.exists():
        raw.update(json.loads(config_path.read_text(encoding="utf-8")))

    sensitive = _tokens(raw, "sensitive_tokens", "DLD_SENSITIVE_TOKENS")
    transfer = _tokens(raw, "transfer_tokens", "DLD_TRANSFER_TOKENS")
    sink = _tokens(raw, "sink_tokens", "DLD_SINK_TOKENS")

    return PolicyConfig(
        sensitive_tokens=sensitive,
        transfer_tokens=transfer,
        sink_tokens=sink,
        normal_activity_tokens=_tuple(raw.get("normal_activity_tokens")),
        unknown_risk_tokens=_tuple(raw.get("unknown_risk_tokens")),
        sink_classification=_classification(raw.get("sink_classification")),
        risk_levels=_classification(raw.get("risk_levels"), label_key="level"),
        semantic_sensitive_aliases=_alias_map(raw.get("semantic_sensitive_aliases")),
        frontend_app_hints={normalize_text(key): str(value) for key, value in dict(raw.get("frontend_app_hints") or {}).items()},
        frontend_category_rules=_classification(raw.get("frontend_category_rules"), label_key="category"),
        risky_app_categories=frozenset(str(item) for item in _tuple(raw.get("risky_app_categories"))),
    )


CONFIRMED_RISK_LEVELS = {"content_exposed", "completed"}
RISK_LEVELS = {"selected_or_attached", "in_progress", "content_exposed", "completed"}


def normalize_text(value: object) -> str:
    """把文本统一为适合规则匹配的形式。"""

    text = unicodedata.normalize("NFKC", str(value or "")).casefold()
    return re.sub(r"\s+", " ", text).strip()


def contains_any(text: str, tokens: tuple[str, ...]) -> bool:
    normalized = f" {normalize_text(text)} "
    compact = re.sub(r"[\s_\-./\\:：,，。;；|]+", "", normalized)
    for token in tokens:
        normalized_token = normalize_text(token)
        if not normalized_token:
            continue
        compact_token = re.sub(r"[\s_\-./\\:：,，。;；|]+", "", normalized_token)
        if normalized_token in normalized or compact_token in compact:
            return True
    return False


def classify_sink(text: str) -> str:
    for sink_type, tokens in POLICY.sink_classification:
        if contains_any(text, tokens):
            return sink_type
    return "network_upload"


def risk_level_for_sink(text: str) -> str:
    for level, tokens in POLICY.risk_levels:
        if contains_any(text, tokens):
            return level
    return "in_progress"


def semantic_sensitive_match(keyword: str, text: str) -> bool:
    normalized_keyword = normalize_text(keyword)
    normalized_text = normalize_text(text)
    for token, aliases in POLICY.semantic_sensitive_aliases.items():
        if normalize_text(token) in normalized_keyword and contains_any(normalized_text, aliases):
            return True
    return False


def is_normal_only_context(text: str) -> bool:
    """判断一段文本是否只是普通浏览/阅读，而没有风险动作。"""

    return contains_any(text, NORMAL_ACTIVITY_TOKENS) and not contains_any(text, UNKNOWN_RISK_TOKENS + SINK_TOKENS + TRANSFER_TOKENS)


def _tokens(raw: dict[str, Any], key: str, env_name: str) -> tuple[str, ...]:
    tokens = list(_tuple(raw.get(key)))
    for token in os.getenv(env_name, "").split(","):
        token = token.strip()
        if token:
            tokens.append(token)
    return _dedupe(tokens)


def _tuple(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    if isinstance(value, list | tuple | set):
        return tuple(str(item).strip() for item in value if str(item).strip())
    return (str(value).strip(),) if str(value).strip() else ()


def _classification(value: Any, *, label_key: str = "type") -> tuple[tuple[str, tuple[str, ...]], ...]:
    items: list[tuple[str, tuple[str, ...]]] = []
    for item in value or ():
        if not isinstance(item, dict):
            continue
        label = str(item.get(label_key) or "").strip()
        tokens = _tuple(item.get("tokens"))
        if label and tokens:
            items.append((label, tokens))
    return tuple(items)


def _alias_map(value: Any) -> dict[str, tuple[str, ...]]:
    aliases: dict[str, tuple[str, ...]] = {}
    for key, items in dict(value or {}).items():
        aliases[str(key)] = _tuple(items)
    return aliases


def _dedupe(tokens: list[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    result: list[str] = []
    for token in tokens:
        key = normalize_text(token)
        if key and key not in seen:
            seen.add(key)
            result.append(token)
    return tuple(result)


POLICY = load_policy_config()

SENSITIVE_TOKENS = POLICY.sensitive_tokens
TRANSFER_TOKENS = POLICY.transfer_tokens
SINK_TOKENS = POLICY.sink_tokens
NORMAL_ACTIVITY_TOKENS = POLICY.normal_activity_tokens
UNKNOWN_RISK_TOKENS = POLICY.unknown_risk_tokens
APP_HINTS = POLICY.frontend_app_hints
APP_CATEGORY_RULES = POLICY.frontend_category_rules
RISKY_APP_CATEGORIES = POLICY.risky_app_categories
