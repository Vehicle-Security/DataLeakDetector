"""Load policy configuration and normalize text semantics."""

from __future__ import annotations

import json
import os
import re
import unicodedata
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_POLICY_PATH = REPO_ROOT / "spec" / "config" / "policy.json"
WHITESPACE_RE = re.compile(r"\s+")
COMPACT_SEPARATOR_RE = re.compile(r"[\s_\-./\\:：，。；\[\]]+")

FALLBACK_POLICY: dict[str, Any] = {
    "sensitive_tokens": (
        "confidential",
        "secret",
        "salary",
        "payroll",
        "customer",
        "contract",
        "finance",
        "password",
        "budget",
        "strategy",
        "机密",
        "绝密",
        "工资",
        "薪资",
        "薪酬",
        "客户",
        "合同",
        "财务",
        "密码",
        "账号",
        "账户",
        "预算",
        "成本",
        "战略",
        "内部",
    ),
    "transfer_tokens": (
        "copy",
        "copied",
        "paste",
        "pasted",
        "clipboard",
        "export",
        "screenshot",
        "screen recording",
        "recording",
        "convert",
        "compress",
        "rename",
        "复制",
        "粘贴",
        "剪贴板",
        "导出",
        "另存",
        "截图",
        "截屏",
        "录屏",
        "转换",
        "压缩",
        "重命名",
    ),
    "sink_tokens": (
        "upload",
        "send",
        "share",
        "mail",
        "email",
        "attach",
        "attachment",
        "cloud",
        "drive",
        "dropbox",
        "onedrive",
        "wechat",
        "qq",
        "feishu",
        "lark",
        "dingtalk",
        "teams",
        "zoom",
        "chatgpt",
        "claude",
        "gemini",
        "kimi",
        "usb",
        "removable",
        "上传",
        "发送",
        "分享",
        "外发",
        "泄露",
        "附件",
        "邮件",
        "邮箱",
        "网盘",
        "云盘",
        "共享",
        "屏幕共享",
        "可移动",
        "U盘",
        "移动磁盘",
        "移动硬盘",
        "微信",
        "飞书",
        "钉钉",
    ),
    "normal_activity_tokens": ("normal", "reading", "正常", "阅读", "浏览"),
    "unknown_risk_tokens": (
        "外发",
        "泄露",
        "隐藏",
        "截图",
        "截屏",
        "录屏",
        "共享",
        "粘贴",
        "发送",
        "导出",
        "另存",
    ),
    "sink_classification": [
        {
            "type": "ai_chat",
            "tokens": ("chatgpt", "gpt", "deepseek", "kimi", "claude", "gemini", "cherry studio", "大模型", "智能助手", "默认助手"),
        },
        {"type": "mail_attachment", "tokens": ("mail", "email", "attachment", "attach", "邮件", "邮箱", "附件")},
        {"type": "cloud_sync", "tokens": ("cloud", "drive", "dropbox", "onedrive", "网盘", "云盘")},
        {"type": "chat_upload", "tokens": ("wechat", "qq", "feishu", "lark", "dingtalk", "chat", "微信", "飞书", "钉钉")},
        {"type": "removable_media", "tokens": ("usb", "removable", "可移动", "U盘", "移动磁盘", "移动硬盘")},
        {"type": "screen_share", "tokens": ("screen", "share", "zoom", "teams", "共享", "屏幕共享")},
    ],
    "risk_levels": [
        {"level": "completed", "tokens": ("complete", "completed", "sent", "success", "已发送", "完成", "成功")},
        {"level": "content_exposed", "tokens": ("visible", "paste", "pasted", "content", "粘贴", "可见", "共享")},
        {"level": "selected_or_attached", "tokens": ("upload", "send", "attach", "selected", "选择", "附件", "上传", "发送")},
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
    },
    "frontend_category_rules": [
        {"category": "ai_chat", "tokens": ("prompt", "assistant", "chatbot", "llm", "大模型", "智能助手", "AI 对话", "AI聊天")},
        {"category": "mail", "tokens": ("mail", "email", "inbox", "compose", "attachment", "邮箱", "邮件", "收件箱", "写邮件", "附件")},
        {"category": "cloud_drive", "tokens": ("cloud drive", "drive", "dropbox", "netdisk", "网盘", "云盘", "文件上传", "上传文件")},
        {"category": "chat", "tokens": ("chat", "message", "messenger", "im", "聊天", "群聊", "会话", "发送消息")},
        {"category": "meeting", "tokens": ("meeting", "screen share", "share screen", "会议", "屏幕共享", "共享屏幕")},
        {"category": "browser", "tokens": ("browser", "chrome_widgetwin", "网页", "浏览器")},
        {"category": "document_editor", "tokens": ("document", "spreadsheet", "office", "文档", "表格", "演示文稿")},
        {"category": "removable_media", "tokens": ("bluetooth", "usb", "removable", "蓝牙", "可移动设备", "移动存储")},
    ],
    "risky_app_categories": ("browser", "ai_chat", "chat", "mail", "cloud_drive", "meeting", "removable_media"),
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
    """Load policy config, falling back to compact built-in defaults."""

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


CONFIRMED_RISK_LEVELS = {"selected_or_attached", "in_progress", "content_exposed", "completed"}
RISK_LEVELS = {"selected_or_attached", "in_progress", "content_exposed", "completed"}


def normalize_text(value: object) -> str:
    """Normalize text for rule matching."""

    text = unicodedata.normalize("NFKC", str(value or "")).casefold()
    return WHITESPACE_RE.sub(" ", text).strip()


def contains_any(text: str, tokens: tuple[str, ...]) -> bool:
    normalized = f" {normalize_text(text)} "
    compact = COMPACT_SEPARATOR_RE.sub("", normalized)
    for normalized_token, compact_token in _token_forms(tokens):
        if normalized_token in normalized or compact_token in compact:
            return True
    return False


@lru_cache(maxsize=128)
def _token_forms(tokens: tuple[str, ...]) -> tuple[tuple[str, str], ...]:
    forms: list[tuple[str, str]] = []
    for token in tokens:
        normalized_token = normalize_text(token)
        if not normalized_token:
            continue
        forms.append((normalized_token, COMPACT_SEPARATOR_RE.sub("", normalized_token)))
    return tuple(forms)


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
    """Return whether text only describes ordinary reading/browsing."""

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
