from __future__ import annotations

from typing import Any


UNKNOWN_RESOURCE_MARKERS = {"", "未知", "??", "unknown", "unk", "n/a"}


def _split_legacy_resources(original_filename: str, modified_filename: str) -> tuple[str, list[str]]:
    original = str(original_filename or "").strip()
    modified = str(modified_filename or "").strip()

    if modified and modified not in UNKNOWN_RESOURCE_MARKERS:
        related = [
            item.strip()
            for item in modified.replace("；", ";").replace("，", ",").replace(",", ";").split(";")
            if item.strip()
        ]
        return original or (related[0] if related else ""), related

    if original:
        related = [
            item.strip()
            for item in original.replace("；", ";").replace("，", ",").replace(",", ";").split(";")
            if item.strip()
        ]
        primary = related[0] if related else original
        others = related[1:] if len(related) > 1 else []
        return primary, others

    return "", []


def _normalize_unknown_resource(value: str) -> str:
    text = str(value or "").strip()
    if not text:
        return ""

    lowered = text.lower()
    if lowered in UNKNOWN_RESOURCE_MARKERS or text in UNKNOWN_RESOURCE_MARKERS:
        return ""
    return text


def adapt_legacy_frame_result(legacy_result: dict[str, Any]) -> dict[str, Any]:
    legacy_events = legacy_result.get("events", []) or []
    segments = []

    for index, event in enumerate(legacy_events):
        primary_resource, related_resources = _split_legacy_resources(
            event.get("original_filename", ""),
            event.get("modified_filename", ""),
        )
        primary_resource = _normalize_unknown_resource(primary_resource)
        related_resources = [
            _normalize_unknown_resource(item)
            for item in related_resources
            if _normalize_unknown_resource(item)
        ]
        description = str(event.get("description", "") or "").strip()
        supporting_timestamps = [
            item
            for item in list(event.get("involved_timestamps", []) or [])
            if str(item or "").strip()
        ]
        segment = {
            "segment_id": f"segment_{index}",
            "time_range": str(event.get("time_range", "") or "").strip(),
            "app_name": str(event.get("app_name", "") or "").strip(),
            "operation_type": str(event.get("operation_type", "") or "").strip(),
            "primary_resource": primary_resource,
            "related_resources": related_resources,
            "action_description": description,
            "visible_evidence": [item for item in [primary_resource, *related_resources] if item],
            "supporting_timestamps": supporting_timestamps,
            "confidence": float(event.get("confidence", 0.8) or 0.8),
            "analysis_backend": "legacy_adapter",
        }
        segments.append(segment)

    all_resources = []
    all_apps = []
    all_operations = []
    for segment in segments:
        all_apps.append(segment["app_name"])
        all_operations.append(segment["operation_type"])
        all_resources.extend([segment["primary_resource"], *segment["related_resources"]])

    return {
        "time_window": legacy_result.get("search_range", {}) or {},
        "ocr_hit": bool(segments),
        "segments": segments,
        "summary": {
            "apps": sorted({item for item in all_apps if item}),
            "operations": sorted({item for item in all_operations if item}),
            "resources": sorted({item for item in all_resources if item}),
        },
        "status": "success" if segments else "no_match",
    }


class FrameAnalyzerAdapter:
    def analyze_with_legacy_backend(
        self,
        rec_start_time_str: str,
        search_start_time_str: str,
        search_end_time_str: str,
        target_keywords: list[str],
        video_path: str,
    ) -> dict[str, Any]:
        from .legacy_relavance_frame import analyze_video_behavior

        legacy_result = analyze_video_behavior(
            rec_start_time_str=rec_start_time_str,
            search_start_time_str=search_start_time_str,
            search_end_time_str=search_end_time_str,
            target_keywords=target_keywords,
            video_path=video_path,
        )
        return adapt_legacy_frame_result(legacy_result)
