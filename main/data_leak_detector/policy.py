"""Policy vocabulary for sensitivity, transfer, and sink detection.

The defaults are intentionally small. Dataset-specific terms can be appended
without code changes through comma-separated environment variables:

- DLD_SENSITIVE_TOKENS
- DLD_TRANSFER_TOKENS
- DLD_SINK_TOKENS
"""

from __future__ import annotations

import os

DEFAULT_SENSITIVE_TOKENS = (
    "salary",
    "payroll",
    "confidential",
    "secret",
    "contract",
    "finance",
    "customer",
    "password",
    "budget",
    "strategy",
    "internal",
    "board",
    "credential",
    "\u5de5\u8d44",
    "\u85aa\u8d44",
    "\u85aa\u916c",
    "\u6708\u85aa",
    "\u673a\u5bc6",
    "\u7edd\u5bc6",
    "\u5408\u540c",
    "\u8d22\u52a1",
    "\u5ba2\u6237",
    "\u5bc6\u7801",
    "\u8d26\u53f7",
    "\u8d26\u6237",
    "\u9884\u7b97",
    "\u6210\u672c",
    "\u6218\u7565",
    "\u5185\u90e8",
    "\u8463\u4e8b\u4f1a",
)

DEFAULT_TRANSFER_TOKENS = (
    "copy",
    "copied",
    "paste",
    "pasted",
    "clipboard",
    "created",
    "modified",
    "rename",
    "renamed",
    "compress",
    "compressed",
    "convert",
    "export",
    "split",
    "screenshot",
    "screen recording",
    "recording",
    "\u590d\u5236",
    "\u7c98\u8d34",
    "\u526a\u8d34\u677f",
    "\u521b\u5efa",
    "\u4fee\u6539",
    "\u91cd\u547d\u540d",
    "\u538b\u7f29",
    "\u8f6c\u6362",
    "\u5bfc\u51fa",
    "\u53e6\u5b58",
    "\u622a\u56fe",
    "\u622a\u5c4f",
    "\u5f55\u5c4f",
    "\u6d3e\u751f",
)

DEFAULT_SINK_TOKENS = (
    "upload",
    "send",
    "share",
    "mail",
    "email",
    "attach",
    "attachment",
    "http_post",
    "post ",
    "network",
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
    "\u4e0a\u4f20",
    "\u53d1\u9001",
    "\u5206\u4eab",
    "\u5916\u53d1",
    "\u6cc4\u9732",
    "\u9644\u4ef6",
    "\u90ae\u4ef6",
    "\u7f51\u76d8",
    "\u4e91\u76d8",
    "\u5171\u4eab",
    "\u5c4f\u5e55\u5171\u4eab",
    "\u53ef\u79fb\u52a8",
    "\u5fae\u4fe1",
    "\u98de\u4e66",
    "\u9489\u9489",
)

LOCAL_APP_TOKENS = ("excel", "word", "wps", "explorer", "finder", "notepad")

CONFIRMED_RISK_LEVELS = {"content_exposed", "completed"}
RISK_LEVELS = {"selected_or_attached", "in_progress", "content_exposed", "completed"}

def _merge_env_tokens(defaults: tuple[str, ...], env_name: str) -> tuple[str, ...]:
    tokens = list(defaults)
    for token in os.getenv(env_name, "").split(","):
        token = token.strip()
        if token and token not in tokens:
            tokens.append(token)
    return tuple(tokens)


SENSITIVE_TOKENS = _merge_env_tokens(DEFAULT_SENSITIVE_TOKENS, "DLD_SENSITIVE_TOKENS")
TRANSFER_TOKENS = _merge_env_tokens(DEFAULT_TRANSFER_TOKENS, "DLD_TRANSFER_TOKENS")
SINK_TOKENS = _merge_env_tokens(DEFAULT_SINK_TOKENS, "DLD_SINK_TOKENS")


def contains_any(text: str, tokens: tuple[str, ...]) -> bool:
    normalized = f" {text.lower()} "
    compact = normalized.replace(" ", "")
    return any(token.lower() in normalized or token.lower().replace(" ", "") in compact for token in tokens)


def classify_sink(text: str) -> str:
    lowered = text.lower()
    if contains_any(lowered, ("mail", "email", "attachment", "attach", "\u90ae\u4ef6", "\u9644\u4ef6")):
        return "mail_attachment"
    if contains_any(lowered, ("cloud", "drive", "dropbox", "onedrive", "\u7f51\u76d8", "\u4e91\u76d8")):
        return "cloud_sync"
    if contains_any(lowered, ("wechat", "qq", "feishu", "lark", "dingtalk", "chat", "\u5fae\u4fe1", "\u98de\u4e66", "\u9489\u9489")):
        return "chat_upload"
    if contains_any(lowered, ("usb", "removable", "\u53ef\u79fb\u52a8")):
        return "removable_media"
    if contains_any(lowered, ("screen", "share", "zoom", "teams", "\u5171\u4eab", "\u5c4f\u5e55\u5171\u4eab")):
        return "screen_share"
    return "network_upload"


def risk_level_for_sink(text: str) -> str:
    lowered = text.lower()
    if contains_any(lowered, ("complete", "sent", "success", "\u5df2\u53d1\u9001", "\u5b8c\u6210", "\u6210\u529f")):
        return "completed"
    if contains_any(lowered, ("visible", "paste", "pasted", "content", "\u7c98\u8d34", "\u53ef\u89c1", "\u5171\u4eab")):
        return "content_exposed"
    if contains_any(lowered, ("upload", "send", "attach", "\u9009\u62e9", "\u9644\u4ef6", "\u4e0a\u4f20", "\u53d1\u9001")):
        return "selected_or_attached"
    return "in_progress"
