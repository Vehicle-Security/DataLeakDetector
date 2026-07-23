"""Parse VLM output into FrameObservation records."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..io import normalize_path, parse_timestamp_ms
from ..models import FrameObservation
from ..policy import SENSITIVE_TOKENS, SINK_TOKENS, TRANSFER_TOKENS, contains_any, is_normal_only_context, semantic_sensitive_match


_UNKNOWN_STRUCTURED_VALUES = {"", "unknown", "none", "null", "n/a", "na", "unspecified", "未知"}
_EXPLICIT_OUTBOUND_ACTIONS = {
    "attach_file",
    "ai_chat_upload",
    "ai_prompt_input",
    "ai_prompt_paste",
    "chat_paste",
    "chat_send",
    "cloud_sync",
    "cloud_upload",
    "commit",
    "copy_paste_to_ai",
    "copy_to_removable_media",
    "document_translation_upload",
    "email_send",
    "file_send",
    "file_upload",
    "http_post",
    "network_upload",
    "article_publish",
    "paste_exfiltration",
    "paste_to_ai",
    "paste_to_web",
    "publish",
    "post_question",
    "screen_share",
    "send",
    "send_click",
    "share_screen",
    "upload",
    "upload_complete",
    "upload_file_to_ai",
    "screenshot_paste_to_chat",
    "screenshot_to_chat",
    "web_form_composition",
    "web_upload",
    "folder_sync",
}


@dataclass(frozen=True)
class ParsedVisionEvent:
    start_ms: int
    end_ms: int
    app_name: str
    behavior_category: str
    operation_type: str
    original_resource: str
    modified_resource: str
    description: str
    confidence: float = 0.80
    evidence_frame_ids: tuple[str, ...] = ()
    sink_type: str = ""
    action_status: str = "unknown"


@dataclass(frozen=True)
class VlmParseResult:
    events: list[ParsedVisionEvent]
    raw_events: list[dict[str, Any]]
    dropped_events: list[dict[str, Any]]
    parse_errors: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "events": [_event_to_dict(item) for item in self.events],
            "raw_events": self.raw_events,
            "dropped_events": self.dropped_events,
            "parse_errors": self.parse_errors,
        }


def parse_vlm_response(response_text: str, *, keywords: list[str] | None = None) -> list[ParsedVisionEvent]:
    return parse_vlm_response_detailed(response_text, keywords=keywords).events


def parse_vlm_response_detailed(response_text: str, *, keywords: list[str] | None = None) -> VlmParseResult:
    try:
        payload = _extract_json(response_text)
    except Exception as exc:
        return VlmParseResult(events=[], raw_events=[], dropped_events=[], parse_errors=[f"{type(exc).__name__}: {exc}"])
    if isinstance(payload, dict):
        raw_events = payload.get("events", [])
    elif isinstance(payload, list):
        raw_events = payload
    else:
        return VlmParseResult(
            events=[],
            raw_events=[],
            dropped_events=[{"reason": "top_level_not_object_or_array", "event": payload}],
            parse_errors=[],
        )
    events: list[ParsedVisionEvent] = []
    raw_event_dicts: list[dict[str, Any]] = []
    dropped: list[dict[str, Any]] = []
    errors: list[str] = []
    for item in raw_events if isinstance(raw_events, list) else []:
        if not isinstance(item, dict):
            dropped.append({"reason": "not_object", "event": item})
            continue
        raw_event_dicts.append(item)
        try:
            event = _normalize_event(item)
        except Exception as exc:
            errors.append(f"event_parse_failed: {type(exc).__name__}: {exc}")
            dropped.append({"reason": "parse_failed", "event": item})
            continue
        if _is_relevant(event, keywords or []):
            events.append(event)
        else:
            dropped.append({"reason": "not_relevant", "event": item})
    return VlmParseResult(events=_dedupe(events), raw_events=raw_event_dicts, dropped_events=dropped, parse_errors=errors)


def vision_events_to_observations(
    events: list[ParsedVisionEvent],
    *,
    source: str = "vlm",
    start_index: int = 0,
) -> list[FrameObservation]:
    observations: list[FrameObservation] = []
    for index, event in enumerate(events, start_index):
        resource = normalize_path(event.modified_resource if event.modified_resource not in {"", "unknown", "未知"} else event.original_resource)
        related = tuple(
            item
            for item in (normalize_path(event.original_resource), normalize_path(event.modified_resource))
            if item and item.lower() not in {"unknown", "未知"}
        )
        observations.append(
            FrameObservation(
                observation_id=f"{source}_{index}",
                start_ms=event.start_ms,
                end_ms=event.end_ms,
                app_name=event.app_name,
                operation_type=_operation_to_pipeline(event),
                resource=resource,
                related_resources=related,
                description=_observation_description(event),
                confidence=event.confidence,
                source=source,
            )
        )
    return observations


def _extract_json(text: str) -> Any:
    stripped = text.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", stripped, flags=re.IGNORECASE | re.DOTALL)
    if fence:
        stripped = fence.group(1).strip()
    if not stripped.startswith(("{", "[")):
        start_candidates = [pos for pos in (stripped.find("{"), stripped.find("[")) if pos >= 0]
        if start_candidates:
            stripped = stripped[min(start_candidates) :]
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        end = max(stripped.rfind("}"), stripped.rfind("]"))
        if end >= 0:
            return json.loads(stripped[: end + 1])
        raise


def _normalize_event(item: dict[str, Any]) -> ParsedVisionEvent:
    start_ms, end_ms = _parse_time_range(str(item.get("time_range") or item.get("time") or ""))
    timestamp_ms = _parse_timestamp_ms_field(item.get("timestamp_ms") or item.get("frame_timestamp_ms"))
    if timestamp_ms:
        start_ms = timestamp_ms
        end_ms = start_ms
    elif not end_ms and start_ms:
        end_ms = start_ms
    original = _first_text(item, "original_filename", "original_file", "file_name", "filename", "resource")
    modified = _first_text(item, "modified_filename", "modified_file", "target_file", "derived_file")
    operation = _first_text(item, "operation_type", "operation", "action")
    app_name = _first_text(item, "app_name", "application", "frontend_app")
    behavior = _first_text(item, "behavior_category", "category", "risk_type")
    description = _first_text(item, "description", "reason", "evidence")
    evidence_frame_ids = _text_tuple(item.get("evidence_frame_ids") or item.get("frame_ids") or item.get("frame_id"))
    action_status = _action_status(
        _first_text(item, "action_status", "status", "transfer_status"),
        context=f"{operation} {behavior} {description}",
    )
    sink_type = _normalize_sink_type(
        _first_text(item, "sink_type", "sink", "channel"),
        app_name=app_name,
        operation=operation,
    )
    if _structured_value(behavior) == "direct_leak" and _is_local_virtual_print(operation, description):
        behavior = "hidden_transfer"
        operation = "virtual_print"
        sink_type = "unknown"
    if action_status == "selected" and _is_preparation_only_context(
        f"{operation} {app_name} {description}"
    ):
        action_status = "unknown"
        if _structured_value(behavior) == "direct_leak":
            behavior = "unknown_risk"
    if (
        action_status == "selected"
        and _structured_value(behavior) == "direct_leak"
        and _is_unconfirmed_chat_attachment(operation, description, original, modified)
    ):
        behavior = "unknown_risk"
        operation = "chat_attachment_preparation"
        sink_type = "unknown"
        action_status = "unknown"
    if (
        _structured_value(behavior) == "hidden_transfer"
        and action_status in {"submitted", "in_progress", "completed"}
        and _is_external_web_content_submission(app_name, operation, description)
    ):
        behavior = "direct_leak"
        operation = "paste_to_web"
        sink_type = "network_upload"
    if (
        _structured_value(behavior) == "direct_leak"
        and _structured_value(sink_type) == "cloud_sync"
        and _is_screenshot_only_action(operation, description)
        and not _has_explicit_cloud_transfer_evidence(description)
    ):
        behavior = "hidden_transfer"
        operation = "screenshot"
        sink_type = "unknown"
    if (
        _structured_value(behavior) == "hidden_transfer"
        and action_status in {"selected", "submitted", "in_progress", "completed"}
        and sink_type in {"unknown", ""}
        and _is_external_ocr_submission(app_name, operation, description)
    ):
        behavior = "direct_leak"
        operation = "network_upload"
        sink_type = "network_upload"
    if (
        _structured_value(behavior) == "hidden_transfer"
        and action_status in {"submitted", "in_progress", "completed"}
        and sink_type in {"unknown", ""}
        and _is_completed_integrated_ai_translation(app_name, operation, description)
    ):
        behavior = "direct_leak"
        operation = "ai_translation"
        sink_type = "ai_chat"
    if (
        _structured_value(behavior) == "direct_leak"
        and _structured_value(sink_type) == "removable_media"
        and _is_unconfirmed_removable_transfer(description)
    ):
        behavior = "unknown_risk"
        operation = "removable_media_observation"
        sink_type = "unknown"
        action_status = "unknown"
    monitoring_log_only_claim = _is_monitoring_log_only_claim(description)
    inferred_recording_attachment = _is_inferred_recording_attachment(description, status=action_status)
    if (
        _is_passive_local_preview(app_name, operation, description)
        or _is_inferred_toolbar_upload(description)
        or monitoring_log_only_claim
        or inferred_recording_attachment
    ):
        behavior = "unknown_risk"
        operation = (
            "monitoring_log_claim"
            if monitoring_log_only_claim
            else "recording_attachment_inference"
            if inferred_recording_attachment
            else "local_preview"
        )
        sink_type = "unknown"
        action_status = "unknown"
    return ParsedVisionEvent(
        start_ms=start_ms,
        end_ms=end_ms or start_ms,
        app_name=app_name,
        behavior_category=behavior or "unknown",
        operation_type=operation or "unknown",
        original_resource=original or "unknown",
        modified_resource=modified or "unknown",
        description=description,
        confidence=_confidence(item.get("confidence")),
        evidence_frame_ids=evidence_frame_ids,
        sink_type=sink_type,
        action_status=action_status,
    )


def _normalize_sink_type(sink_type: str, *, app_name: str, operation: str) -> str:
    sink = sink_type.strip().lower()
    app = app_name.strip().lower()
    action = operation.strip().lower().replace(" ", "_")
    meeting_document_actions = {"import_document", "import_local_document", "meeting_document_upload"}
    meeting_apps = ("tencent meeting", "wemeet", "腾讯会议")
    if sink == "screen_share" and action in meeting_document_actions and any(token in app for token in meeting_apps):
        return "chat_upload"
    if (
        sink == "cloud_sync"
        and any(token in app for token in ("github", "gitlab", "gitee", "jihulab", "codeberg"))
        and any(token in action for token in ("commit", "push", "upload", "paste"))
    ):
        return "network_upload"
    return sink_type


def _is_external_web_content_submission(app_name: str, operation: str, description: str) -> bool:
    text = f"{app_name} {operation} {description}".lower()
    submitted_content = bool(re.search(r"\bpasted\b.{0,100}\binto\b", text)) or any(
        marker in text
        for marker in (
            "pasted into",
            "pasted to",
            "pasted the content",
            "submitted to",
            "粘贴到",
            "提交到",
        )
    )
    web_destination = bool(
        re.search(r"\b[a-z0-9-]+\.(?:com|cn|net|org|io|ai)\b", text)
        or any(
            marker in text
            for marker in (
                "web-based",
                "online tool",
                "online service",
                "third-party tool",
                "browser",
                "在线工具",
                "在线服务",
                "第三方工具",
            )
        )
    )
    return submitted_content and web_destination


def _is_passive_local_preview(app_name: str, operation: str, description: str) -> bool:
    text = f"{app_name} {operation} {description}".lower()
    local_shell = any(marker in text for marker in ("file explorer", "windows explorer", "资源管理器"))
    preview = any(marker in text for marker in ("preview pane", "preview panel", "预览窗格", "预览栏"))
    ai_summary = any(marker in text for marker in ("ai summary", "ai public document", "ai 公文", "ai公文"))
    explicit_invocation = any(
        marker in text
        for marker in (
            "clicked ai",
            "opened ai",
            "submitted to ai",
            "uploaded to ai",
            "点击ai",
            "提交到ai",
            "上传到ai",
        )
    )
    return local_shell and preview and ai_summary and not explicit_invocation


def _is_inferred_toolbar_upload(description: str) -> bool:
    text = description.lower()
    passive_display = any(marker in text for marker in ("is displayed in", "is visible in", "显示在", "可见于"))
    toolbar_only = any(
        marker in text
        for marker in (
            "toolbar shows",
            "toolbar contains",
            "indicated by login",
            "indicated by login/vip",
            "indicated by vip",
            "ai processing toolbar",
            "工具栏显示",
            "工具栏包含",
        )
    )
    inferred_upload = any(
        marker in text
        for marker in (
            "implying the file was uploaded",
            "implying a network upload",
            "suggesting the file was uploaded",
            "therefore it was uploaded",
            "由此推断文件已上传",
            "暗示文件已上传",
        )
    )
    explicit_transfer = any(
        marker in text
        for marker in (
            "upload progress",
            "upload completed",
            "submitted to",
            "clicked upload",
            "service result",
            "上传进度",
            "上传完成",
            "点击上传",
        )
    )
    return toolbar_only and inferred_upload and not explicit_transfer and (
        passive_display or "loaded into" in text or "indicated by" in text
    )


def _is_monitoring_log_only_claim(description: str) -> bool:
    text = description.lower()
    monitoring = any(
        marker in text
        for marker in (
            "monitoring logs",
            "monitor log",
            "powershell log",
            "logs in powershell",
            "监控日志",
            "powershell 日志",
        )
    )
    inferred_from_log = any(
        marker in text
        for marker in (
            "log confirms",
            "logs explicitly record",
            "log explicitly records",
            "based on the log",
            "日志确认",
            "日志表明",
        )
    )
    direct_ui_evidence = any(
        marker in text
        for marker in (
            "attachment card is visible",
            "upload progress is visible",
            "send confirmation is visible",
            "generated answer is visible",
            "上传进度可见",
            "附件卡片可见",
            "发送确认可见",
        )
    )
    return monitoring and inferred_from_log and not direct_ui_evidence


def _is_inferred_recording_attachment(description: str, *, status: str) -> bool:
    if status != "selected":
        return False
    text = description.lower()
    recording = any(
        marker in text
        for marker in ("screen recording", "recording session", "screenshot", "recording mp4", "录屏", "截图")
    )
    attachment = any(
        marker in text for marker in ("attachment", "attached", "staged", "thumbnail", "附件", "待发送", "缩略图")
    )
    inferred = any(
        marker in text
        for marker in (
            "likely the screen recording",
            "likely a screen recording",
            "likely a screenshot",
            "likely of the sensitive document",
            "likely of the recording session",
            "given the dark thumbnail",
            "appears to contain the recording",
            "可能是录屏",
            "可能是截图",
        )
    )
    clearly_identified = any(
        marker in text
        for marker in (
            "attachment card is visible",
            "identified attachment card",
            "clearly identified screenshot",
            "明确的截图附件",
        )
    )
    completed = any(marker in text for marker in ("was sent", "sent successfully", "发送成功", "已发送"))
    return recording and attachment and inferred and not clearly_identified and not completed


def _parse_time_range(value: str) -> tuple[int, int]:
    if " - " in value:
        start, end = value.split(" - ", 1)
        return parse_timestamp_ms(start.strip()), parse_timestamp_ms(end.strip())
    parsed = parse_timestamp_ms(value)
    return parsed, parsed


def _first_text(item: dict[str, Any], *names: str) -> str:
    for name in names:
        value = item.get(name)
        if value is not None:
            return str(value).strip()
    return ""


def _parse_timestamp_ms_field(value: Any) -> int:
    if value is None or value == "":
        return 0
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return parse_timestamp_ms(value)


def _text_tuple(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value.strip(),) if value.strip() else ()
    if isinstance(value, list | tuple | set):
        return tuple(str(item).strip() for item in value if str(item).strip())
    return (str(value).strip(),) if str(value).strip() else ()


def _confidence(value: Any) -> float:
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        confidence = 0.80
    return max(0.0, min(confidence, 1.0))


def _action_status(value: str, *, context: str = "") -> str:
    status = value.strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "attached": "selected",
        "queued": "submitted",
        "uploading": "in_progress",
        "sending": "in_progress",
        "success": "completed",
        "succeeded": "completed",
        "complete": "completed",
        "error": "failed",
        "unsupported": "failed",
        "not_supported": "failed",
        "timed_out": "failed",
        "timeout": "failed",
        "rejected": "failed",
        "cancelled": "failed",
        "canceled": "failed",
    }
    status = aliases.get(status, status)
    allowed = {"selected", "submitted", "in_progress", "completed", "failed", "unknown"}
    if status in allowed and status != "unknown":
        return status
    context_text = context.lower()
    if re.search(r"\b(completed|complete|succeeded|success|sent)\b", context_text) and not re.search(
        r"\b(not supported|unsupported|failed|rejected|timed out|timeout)\b",
        context_text,
    ):
        return "completed"
    if any(
        marker in context_text
        for marker in (
            "not supported",
            "unsupported",
            "rejected",
            "cancelled",
            "canceled",
            "timed out",
            "timeout",
            "error occurred",
            "error:",
            "failed",
            "不支持",
            "失败",
            "错误",
            "取消",
        )
    ):
        return "failed"
    return "unknown"


def _is_preparation_only_context(value: str) -> bool:
    text = value.strip().lower()
    if not text:
        return False
    separate_upload_not_started = (
        any(marker in text for marker in ("upload button is visible", "上传按钮可见", "显示上传按钮"))
        and any(
            marker in text
            for marker in (
                "no upload progress",
                "upload has not started",
                "button was not clicked",
                "button is not clicked",
                "未开始上传",
                "未点击上传",
                "没有上传进度",
            )
        )
    )
    local_recording_preview = (
        any(marker in text for marker in ("screen recording preview", "recording preview", "录屏预览"))
        and any(marker in text for marker in ("send to", "发送到"))
        and any(marker in text for marker in ("button", "按钮", "staged", "待发送"))
        and not any(
            marker in text
            for marker in (
                "button was clicked",
                "clicked the send",
                "was sent",
                "sent successfully",
                "attached to the chat",
                "点击发送",
                "发送成功",
                "已发送",
                "已附加到聊天",
            )
        )
    )
    staged_web_form = (
        any(marker in text for marker in ("staged in the upload area", "ready for upload", "已进入上传区域", "等待上传"))
        and ("upload progress" not in text or "no upload progress" in text)
        and not any(marker in text for marker in ("upload completed", "upload succeeded", "was uploaded", "上传完成", "上传成功", "已上传"))
    )
    if separate_upload_not_started or local_recording_preview or staged_web_form:
        return True
    concrete_action_markers = (
        "file was selected",
        "selected file",
        "attached file",
        "transfer queue",
        "已选择文件",
        "已添加附件",
        "传输队列",
    )
    if any(marker in text for marker in concrete_action_markers):
        return False
    if (
        "upload progress is visible" in text
        and "no upload progress is visible" not in text
    ) or (
        "send progress is visible" in text
        and "no send progress is visible" not in text
    ):
        return False
    return any(
        marker in text
        for marker in (
            "indicating an intent",
            "suggesting potential",
            "no later action",
            "not confirmed",
            "merely visible",
            "menu was opened",
            "menu is visible",
            "capability is visible",
            "仅显示",
            "尚未确认",
            "未确认",
            "意图",
        )
    )


def _is_external_ocr_submission(app_name: str, operation: str, description: str) -> bool:
    text = f"{app_name} {operation} {description}".lower()
    ocr = any(marker in text for marker in ("ocr", "extract text", "text extraction", "提取文字", "文字识别"))
    ai_tool = any(marker in text for marker in ("ai image", "ai看图", "ai 看图", "ai tool", "图片工具"))
    invoked = any(
        marker in text
        for marker in (
            "initiates recognition",
            "start recognition",
            "selects the 'ai extract text",
            'selects the "ai extract text',
            "selects ai extract text",
            "uses the ai extract text",
            "initiating the process",
            "initiates the process",
            "开始识别",
            "启动识别",
            "点击识别",
            "选择ai提取文字",
            "选择 ai提取文字",
        )
    )
    passive_only = any(
        marker in text
        for marker in (
            "toolbar capability",
            "toolbar is visible",
            "feature is merely visible",
            "does not invoke",
            "not invoked",
            "仅显示工具栏",
            "未调用",
            "未开始识别",
        )
    )
    return ocr and ai_tool and (invoked or not passive_only)


def _is_local_virtual_print(operation: str, description: str) -> bool:
    text = f"{operation} {description}".lower()
    return (
        any(marker in text for marker in ("print", "打印"))
        and any(
            marker in text
            for marker in (
                "onenote (desktop)",
                "microsoft onenote 16 driver",
                "send to microsoft onenote",
                "onenote desktop",
            )
        )
        and not any(marker in text for marker in ("onenote web", "share link", "shared notebook", "上传完成"))
    )


def _is_completed_integrated_ai_translation(app_name: str, operation: str, description: str) -> bool:
    text = f"{app_name} {operation} {description}".lower()
    integrated_tool = any(marker in text for marker in ("wps", "built-in translation", "integrated ai"))
    translation = any(marker in text for marker in ("translation", "translate", "translated", "翻译"))
    result = any(
        marker in text
        for marker in (
            "copy success",
            "translated result",
            "translated text",
            "translation result",
            "translated the document content",
            "翻译结果",
            "翻译完成",
            "复制成功",
        )
    )
    return integrated_tool and translation and result


def _is_screenshot_only_action(operation: str, description: str) -> bool:
    text = f"{operation} {description}".lower()
    if any(
        marker in text
        for marker in (
            "paste to cloud",
            "paste into cloud",
            "pasted to cloud",
            "pasted into cloud",
            "paste_to_cloud",
            "粘贴到云文档",
            "粘贴至云文档",
            "上传到云文档",
        )
    ):
        return False
    return any(marker in text for marker in ("screenshot", "screen capture", "截图", "屏幕截图"))


def _is_unconfirmed_removable_transfer(description: str) -> bool:
    text = description.lower()
    return (
        any(marker in text for marker in ("protecting drive", "正在保护", "保护驱动器"))
        and any(marker in text for marker in ("likely", "可能", "推测", "疑似"))
    )


def _is_unconfirmed_chat_attachment(
    operation: str,
    description: str,
    original: str,
    modified: str,
) -> bool:
    text = f"{operation} {description}".lower()
    chat_action = any(marker in text for marker in ("chat_upload", "screenshot_chat_upload", "附件", "attached"))
    staged = any(marker in text for marker in ("send button visible", "send button is visible", "发送按钮可见", "待发送"))
    sent = any(marker in text for marker in ("clicked send", "sent successfully", "was sent", "点击发送", "发送成功", "已发送"))
    if not (chat_action and staged and not sent):
        return False
    # A visibly inserted screenshot preview is itself the derived sensitive
    # carrier, even when the PNG filename is not shown in the composer.
    if (
        "screenshot" in text
        and any(
            marker in text
            for marker in (
                "staged in the qq chat composer",
                "staged in the composer",
                "visible in the composer",
                "chat composer",
                "preview is visible",
                "thumbnail is visible",
                "编辑区可见",
                "编辑区中",
            )
        )
    ):
        return False
    source_name = Path(normalize_path(original)).name.lower()
    target_name = Path(normalize_path(modified)).name.lower()
    def has_visible_card(name: str) -> bool:
        if not name or name == "unknown":
            return False
        return any(
            marker in text
            for marker in (
                f"attached file '{name}'",
                f'attachment card shows {name}',
                f"file '{name}' is attached",
                f"attachment '{name}'",
                f"附件卡显示{name}",
                f"附件为{name}",
            )
        )

    has_named_source = has_visible_card(source_name)
    has_named_target = has_visible_card(target_name)
    return not (has_named_source or has_named_target)


def _has_explicit_cloud_transfer_evidence(description: str) -> bool:
    text = description.lower()
    return any(
        marker in text
        for marker in (
            "upload progress",
            "sync progress",
            "upload completed",
            "sync completed",
            "上传进度",
            "同步进度",
            "上传完成",
            "同步完成",
        )
    )


def _is_relevant(event: ParsedVisionEvent, keywords: list[str]) -> bool:
    text = " ".join(
        [
            event.app_name,
            event.behavior_category,
            event.operation_type,
            event.original_resource,
            event.modified_resource,
            event.description,
            event.sink_type,
            " ".join(event.evidence_frame_ids),
        ]
    ).lower()
    sensitive_match = any(_mentions_sensitive_keyword(text, keyword) for keyword in keywords)
    if sensitive_match and event.action_status in {"selected", "submitted", "in_progress", "completed", "failed"}:
        return True
    if _is_normal_only(text):
        return False
    if contains_any(text, SINK_TOKENS) or contains_any(text, TRANSFER_TOKENS):
        return True
    if contains_any(text, SENSITIVE_TOKENS):
        return True
    if sensitive_match:
        return True
    compact = re.sub(r"\s+", "", text)
    for keyword in keywords:
        key = re.sub(r"\s+", "", keyword.lower())
        if key and (key in compact or semantic_sensitive_match(key, compact)):
            return True
    return "unknown" in event.behavior_category.lower() or "未知" in event.behavior_category


def _mentions_sensitive_keyword(text: str, keyword: str) -> bool:
    normalized = normalize_path(keyword).strip().lower()
    if not normalized:
        return False
    name = Path(normalized).name
    stem = Path(name).stem
    compact = re.sub(r"\s+", "", normalize_path(text).lower())
    return any(
        len(alias) >= 3 and re.sub(r"\s+", "", alias) in compact
        for alias in {normalized, name, stem}
        if alias
    )


def _is_normal_only(text: str) -> bool:
    return is_normal_only_context(text)


def _operation_to_pipeline(event: ParsedVisionEvent) -> str:
    behavior = _structured_value(event.behavior_category)
    if _is_confirmed_external_submission(event):
        return "external_sink_interaction"
    if (
        behavior == "direct_leak"
        and event.action_status in {"submitted", "in_progress", "completed"}
        and _structured_value(event.sink_type) == "ai_chat"
    ):
        return "external_sink_interaction"
    if behavior in {"hidden_transfer", "unknown_risk"}:
        return "file_or_content_transfer"
    text = f"{event.behavior_category} {event.operation_type} {event.description} {event.sink_type}".lower()
    if behavior == "direct_leak" and (
        _is_explicit_outbound_action(event.operation_type)
        or (_has_explicit_sink(event.sink_type) and not _is_suspicious_content_transform(event, text))
    ):
        return "external_sink_interaction"
    if _is_suspicious_content_transform(event, text):
        return "file_or_content_transfer"
    if contains_any(text, SINK_TOKENS):
        return "external_sink_interaction"
    if contains_any(text, TRANSFER_TOKENS):
        return "file_or_content_transfer"
    return event.operation_type or "visual_review"


def _structured_value(value: str) -> str:
    return re.sub(r"[\s-]+", "_", (value or "").strip().lower())


def _has_explicit_sink(sink_type: str) -> bool:
    return _structured_value(sink_type) not in _UNKNOWN_STRUCTURED_VALUES


def _is_explicit_outbound_action(operation_type: str) -> bool:
    action = _structured_value(operation_type)
    if action in _EXPLICIT_OUTBOUND_ACTIONS:
        return True
    if not action or "processing" in action or "transform" in action:
        return False
    return action.startswith(("publish_", "article_publish_", "folder_sync_"))


def _is_confirmed_external_submission(event: ParsedVisionEvent) -> bool:
    if event.action_status not in {"submitted", "in_progress", "completed"}:
        return False
    if not _has_explicit_sink(event.sink_type):
        return False
    if (
        event.behavior_category == "direct_leak"
        and event.action_status in {"submitted", "in_progress", "completed"}
        and _structured_value(event.sink_type) == "network_upload"
        and event.operation_type.lower() in {"translate", "translation", "rewrite", "改写", "翻译"}
    ):
        return True
    if _is_explicit_outbound_action(event.operation_type):
        return True
    text = f"{event.operation_type} {event.description}".lower()
    return any(
        marker in text
        for marker in (
            "pasted into",
            "pasted to",
            "sends it to",
            "sent it to",
            "submitted to",
            "uploaded to",
            "sent to",
            "transmitted to",
            "synced to",
            "transferring sensitive data to",
            "粘贴到",
            "提交到",
            "上传到",
            "发送到",
            "同步到",
        )
    )


def _is_suspicious_content_transform(event: ParsedVisionEvent, text: str) -> bool:
    operation = (event.operation_type or "").lower()
    if operation in {"cloud_sync_access", "cloud sync access"}:
        return True
    return any(
        token in text
        for token in (
            "translation",
            "translate",
            "translating",
            "rewrite",
            "rewriting",
            "paraphrase",
            "base64",
            "encode",
            "encoding",
            "decode",
            "decoding",
            "翻译",
            "全文翻译",
            "改写",
            "润色",
            "编码",
            "解码",
            "转码",
        )
    )


def _observation_description(event: ParsedVisionEvent) -> str:
    parts = [f"{event.behavior_category}: {event.operation_type}."]
    if event.evidence_frame_ids:
        parts.append("evidence_frame_ids=" + "|".join(event.evidence_frame_ids) + ".")
    if event.sink_type:
        parts.append(f"sink_type={event.sink_type}.")
    parts.append(f"action_status={event.action_status}.")
    if event.description:
        parts.append(event.description)
    return " ".join(parts)


def _event_to_dict(event: ParsedVisionEvent) -> dict[str, Any]:
    return {
        "start_ms": event.start_ms,
        "end_ms": event.end_ms,
        "app_name": event.app_name,
        "behavior_category": event.behavior_category,
        "operation_type": event.operation_type,
        "original_resource": event.original_resource,
        "modified_resource": event.modified_resource,
        "description": event.description,
        "confidence": event.confidence,
        "evidence_frame_ids": list(event.evidence_frame_ids),
        "sink_type": event.sink_type,
        "action_status": event.action_status,
    }


def _dedupe(events: list[ParsedVisionEvent]) -> list[ParsedVisionEvent]:
    local_recordings = [event for event in events if _is_local_screen_recording(event)]
    seen: set[tuple[object, ...]] = set()
    result: list[ParsedVisionEvent] = []
    for event in events:
        if _is_screen_share_conflicted_by_local_recording(event, local_recordings):
            continue
        key = (
            event.start_ms,
            event.app_name,
            event.operation_type,
            event.original_resource,
            event.modified_resource,
            event.sink_type,
            event.action_status,
            event.description,
        )
        if key in seen:
            continue
        seen.add(key)
        result.append(event)
    return result


def _is_local_screen_recording(event: ParsedVisionEvent) -> bool:
    action = _structured_value(event.operation_type)
    text = f"{event.modified_resource} {event.description}".lower()
    return action in {"screen_recording", "screen_record", "record_screen"} and any(
        marker in text for marker in ("mp4", "screen recording", "录屏", "屏幕录制")
    )


def _is_screen_share_conflicted_by_local_recording(
    event: ParsedVisionEvent,
    local_recordings: list[ParsedVisionEvent],
) -> bool:
    if _structured_value(event.operation_type) not in {"screen_share", "share_screen"}:
        return False
    text = f"{event.modified_resource} {event.description}".lower()
    if not any(marker in text for marker in ("mp4", "screen recording", "录屏", "屏幕录制")):
        return False
    if _has_independent_screen_share_evidence(text):
        return False
    event_name = Path(normalize_path(event.original_resource)).stem.lower()
    return any(
        abs(recording.start_ms - event.start_ms) <= 10_000
        and (
            not event_name
            or event_name in {"unknown", "未知"}
            or Path(normalize_path(recording.original_resource)).stem.lower() == event_name
        )
        for recording in local_recordings
    )


def _has_independent_screen_share_evidence(text: str) -> bool:
    return any(
        marker in text
        for marker in (
            "sharing toolbar",
            "share toolbar",
            "sharing banner",
            "share banner",
            "active share indicator",
            "remote participant",
            "共享工具栏",
            "共享横幅",
            "正在共享",
            "远端参会者",
        )
    )
