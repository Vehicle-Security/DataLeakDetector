"""
模块3工具函数
- 模块3专用辅助函数
- 复用模块2共享工具函数
"""

import os
import re
import sys
from typing import Any, Dict, List

sys.path.append(os.path.join(os.path.dirname(__file__), "../2-FileTracker"))

from behavior_analysis_tools import (
    build_sensitive_operation_dedup_key,
    build_sensitive_operation_record,
    normalize_file_path,
    normalize_timestamp_display,
    read_recording_start_time,
    resolve_full_path,
    split_output_filenames,
)


def extract_hidden_transformed_paths(module2_result: Dict[str, Any]) -> List[str]:
    """
    直接从模块2结果提取转换后文件路径列表，按模块2输出顺序返回。
    """
    new_events = module2_result.get("new_events", []) if isinstance(module2_result, dict) else []
    transformed_paths = []

    for new_event in new_events:
        if isinstance(new_event, dict):
            path = new_event.get("current_file", "")
        else:
            path = getattr(new_event, "current_file", "")

        if path:
            transformed_paths.append(path)

    return transformed_paths


def append_operation_record_with_dedup(state: Dict[str, Any], operation_record: Dict[str, Any]) -> bool:
    """
    添加操作记录并去重。

    去重键：同时间 + 同文件 + 同操作。
    Returns:
        True: 新增成功
        False: 重复记录，已忽略
    """
    dedup_key = build_sensitive_operation_dedup_key(operation_record)

    keys = state.get("_operation_record_keys")
    if not isinstance(keys, set):
        keys = set()
        state["_operation_record_keys"] = keys

    if dedup_key in keys:
        return False

    keys.add(dedup_key)
    state["operation_records"].append(operation_record)
    return True


def _get_upload_bucket_name(upload_event: Any) -> str:
    return "alert_events" if getattr(upload_event, "should_alert", False) else "info_events"


def _remove_upload_event_from_bucket(bucket: List[Any], upload_event: Any) -> None:
    for idx, existing in enumerate(bucket):
        if existing is upload_event:
            del bucket[idx]
            return


def _normalize_upload_content(upload_content: str) -> str:
    items = split_output_filenames(upload_content)
    if not items:
        return normalize_file_path(upload_content).lower()

    normalized_items = []
    seen = set()
    for item in items:
        normalized = normalize_file_path(item).lower()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        normalized_items.append(normalized)

    return "|".join(sorted(normalized_items))


def _extract_upload_event_time_bucket(upload_event: Any) -> str:
    time_range = str(getattr(upload_event, "time_range", "") or "").strip()
    if time_range:
        start_time = re.split(r"\s+-\s+", time_range, maxsplit=1)[0]
    else:
        start_time = getattr(upload_event, "timestamp", "")

    normalized = normalize_timestamp_display(start_time)
    return normalized[:16] if normalized else ""


def build_upload_event_dedup_key(upload_event: Any) -> str:
    """
    生成上传事件去重键。

    同一文件、同一应用、同一分钟内的上传结果视为同一候选事实；
    如果后续出现更完整的证据，则使用更高质量结果覆盖旧事件。
    """
    return "|".join(
        [
            normalize_file_path(getattr(upload_event, "file_path", "")).lower(),
            str(getattr(upload_event, "app_name", "") or "").strip().lower(),
            _extract_upload_event_time_bucket(upload_event),
        ]
    )


def _score_upload_event(upload_event: Any) -> int:
    score = 0
    alert_level = str(getattr(upload_event, "alert_level", "") or "").strip().lower()
    upload_content = _normalize_upload_content(getattr(upload_event, "upload_content", ""))
    file_path = normalize_file_path(getattr(upload_event, "file_path", "")).lower()
    mapping_link = str(getattr(upload_event, "upload_content_mapping_link", "") or "").strip()
    description = str(getattr(upload_event, "description", "") or "").strip()
    involved_timestamps = getattr(upload_event, "involved_timestamps", []) or []

    if getattr(upload_event, "should_alert", False):
        score += 8

    if alert_level == "critical":
        score += 4
    elif alert_level == "warning":
        score += 2
    elif alert_level == "info":
        score += 1

    if mapping_link and "->" in mapping_link:
        score += 3

    if upload_content and upload_content != file_path:
        score += 2

    if getattr(upload_event, "time_range", ""):
        score += 1

    if description:
        score += 1

    score += min(len(involved_timestamps), 5)
    return score


def _replace_upload_event(existing_event: Any, new_event: Any) -> None:
    for field_name, value in vars(new_event).items():
        setattr(existing_event, field_name, value)


def append_upload_event_with_dedup(state: Dict[str, Any], upload_event: Any) -> bool:
    """
    添加上传事件并去重。

    Returns:
        True: 新增事件
        False: 事件已存在，本次仅忽略或用更高质量证据覆盖旧事件
    """
    event_index = state.get("_upload_event_index")
    if not isinstance(event_index, dict):
        event_index = {}
        state["_upload_event_index"] = event_index

    dedup_key = build_upload_event_dedup_key(upload_event)
    existing_event = event_index.get(dedup_key)

    if existing_event is None:
        event_index[dedup_key] = upload_event
        state["upload_events"].append(upload_event)
        state[_get_upload_bucket_name(upload_event)].append(upload_event)
        return True

    if _score_upload_event(upload_event) > _score_upload_event(existing_event):
        old_bucket_name = _get_upload_bucket_name(existing_event)
        new_bucket_name = _get_upload_bucket_name(upload_event)

        if old_bucket_name != new_bucket_name:
            _remove_upload_event_from_bucket(state.get(old_bucket_name, []), existing_event)
            state[new_bucket_name].append(existing_event)

        _replace_upload_event(existing_event, upload_event)

    return False


def refresh_upload_statistics(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    基于去重后的上传结果重算统计，避免统计值与最终输出不一致。
    """
    statistics = state.setdefault("statistics", {})
    upload_events = state.get("upload_events", [])
    alert_events = state.get("alert_events", [])
    info_events = state.get("info_events", [])

    statistics["upload_events_detected"] = len(upload_events)
    statistics["blacklist_alerts"] = sum(
        1
        for event in alert_events
        if getattr(event, "app_category", "") == "blacklist" and getattr(event, "should_alert", False)
    )
    statistics["whitelist_uploads"] = sum(
        1 for event in info_events if getattr(event, "app_category", "") == "whitelist"
    )
    statistics["unknown_uploads"] = sum(
        1 for event in info_events if getattr(event, "app_category", "") == "unknown"
    )

    return statistics
