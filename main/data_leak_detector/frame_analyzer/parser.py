"""把 OCR/VLM 输出解析为 FrameObservation。"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from ..io import normalize_path, parse_timestamp_ms
from ..models import FrameObservation
from ..policy import SENSITIVE_TOKENS, SINK_TOKENS, TRANSFER_TOKENS, contains_any, is_normal_only_context, semantic_sensitive_match


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


def parse_vlm_response(response_text: str, *, keywords: list[str] | None = None) -> list[ParsedVisionEvent]:
    payload = _extract_json(response_text)
    raw_events = payload.get("events", payload if isinstance(payload, list) else [])
    events: list[ParsedVisionEvent] = []
    for item in raw_events if isinstance(raw_events, list) else []:
        if not isinstance(item, dict):
            continue
        event = _normalize_event(item)
        if _is_relevant(event, keywords or []):
            events.append(event)
    return _dedupe(events)


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
                description=f"{event.behavior_category}: {event.operation_type}. {event.description}",
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
    original = _first_text(item, "original_filename", "original_file", "file_name", "filename", "resource")
    modified = _first_text(item, "modified_filename", "modified_file", "target_file", "derived_file")
    operation = _first_text(item, "operation_type", "operation", "action")
    behavior = _first_text(item, "behavior_category", "category", "risk_type")
    description = _first_text(item, "description", "reason", "evidence")
    return ParsedVisionEvent(
        start_ms=start_ms,
        end_ms=end_ms or start_ms,
        app_name=_first_text(item, "app_name", "application", "frontend_app"),
        behavior_category=behavior or "unknown",
        operation_type=operation or "unknown",
        original_resource=original or "unknown",
        modified_resource=modified or "unknown",
        description=description,
        confidence=float(item.get("confidence") or 0.80),
    )


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


def _is_relevant(event: ParsedVisionEvent, keywords: list[str]) -> bool:
    text = " ".join(
        [
            event.app_name,
            event.behavior_category,
            event.operation_type,
            event.original_resource,
            event.modified_resource,
            event.description,
        ]
    ).lower()
    if _is_normal_only(text):
        return False
    if contains_any(text, SINK_TOKENS) or contains_any(text, TRANSFER_TOKENS):
        return True
    if contains_any(text, SENSITIVE_TOKENS):
        return True
    compact = re.sub(r"\s+", "", text)
    for keyword in keywords:
        key = re.sub(r"\s+", "", keyword.lower())
        if key and (key in compact or semantic_sensitive_match(key, compact)):
            return True
    return "unknown" in event.behavior_category.lower() or "未知" in event.behavior_category


def _is_normal_only(text: str) -> bool:
    return is_normal_only_context(text)


def _operation_to_pipeline(event: ParsedVisionEvent) -> str:
    text = f"{event.behavior_category} {event.operation_type} {event.description}".lower()
    if contains_any(text, SINK_TOKENS):
        return "external_sink_interaction"
    if contains_any(text, TRANSFER_TOKENS):
        return "file_or_content_transfer"
    return event.operation_type or "visual_review"


def _dedupe(events: list[ParsedVisionEvent]) -> list[ParsedVisionEvent]:
    seen: set[tuple[int, str, str, str]] = set()
    result: list[ParsedVisionEvent] = []
    for event in events:
        key = (event.start_ms, event.app_name, event.operation_type, event.description)
        if key in seen:
            continue
        seen.add(key)
        result.append(event)
    return result
