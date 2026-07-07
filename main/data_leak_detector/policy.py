"""敏感性、传输和汇聚点检测的核心策略词汇。

检测器刻意把关键字策略集中在一个文件里。这样一来，启发式规则更容易审计、调优，
或者在以后替换成学习型策略，而不会把产品假设散落到 FrameAnalyzer 和 EventCorrelator 各处。
"""

from __future__ import annotations

SENSITIVE_TOKENS = (
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
    "工资",
    "薪资",
    "薪酬",
    "机密",
    "绝密",
    "合同",
    "财务",
    "客户",
    "密码",
    "预算",
    "战略",
    "内部",
    "董事会",
)

TRANSFER_TOKENS = (
    "copy",
    "copied",
    "paste",
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
    "复制",
    "粘贴",
    "剪贴板",
    "创建",
    "修改",
    "重命名",
    "压缩",
    "转换",
    "导出",
)

SINK_TOKENS = (
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
    "上传",
    "发送",
    "分享",
    "外发",
    "附件",
    "邮件",
    "网盘",
    "云盘",
    "共享",
    "可移动",
    "微信",
    "飞书",
    "钉钉",
)

LOCAL_APP_TOKENS = ("excel", "word", "wps", "explorer", "finder", "notepad")

CONFIRMED_RISK_LEVELS = {"content_exposed", "completed"}
RISK_LEVELS = {"selected_or_attached", "in_progress", "content_exposed", "completed"}


def contains_any(text: str, tokens: tuple[str, ...]) -> bool:
    normalized = f" {text.lower()} "
    return any(token.lower() in normalized for token in tokens)


def classify_sink(text: str) -> str:
    lowered = text.lower()
    if contains_any(lowered, ("mail", "email", "attachment", "attach", "邮件", "附件")):
        return "mail_attachment"
    if contains_any(lowered, ("cloud", "drive", "dropbox", "onedrive", "网盘", "云盘")):
        return "cloud_sync"
    if contains_any(lowered, ("wechat", "qq", "feishu", "lark", "dingtalk", "chat", "微信", "飞书", "钉钉")):
        return "chat_upload"
    if contains_any(lowered, ("usb", "removable", "可移动")):
        return "removable_media"
    if contains_any(lowered, ("screen", "share", "zoom", "teams", "共享")):
        return "screen_share"
    return "network_upload"


def risk_level_for_sink(text: str) -> str:
    lowered = text.lower()
    if contains_any(lowered, ("complete", "sent", "success", "已发送", "完成", "成功")):
        return "completed"
    if contains_any(lowered, ("visible", "paste", "pasted", "content", "粘贴", "可见")):
        return "content_exposed"
    if contains_any(lowered, ("upload", "send", "attach", "选择", "附件", "上传")):
        return "selected_or_attached"
    return "in_progress"
