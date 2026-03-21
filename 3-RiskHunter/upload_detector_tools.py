# upload_detector_tools.py
"""
模块3工具函数
- 模块3专用辅助函数
- 复用模块2共享工具函数
"""

import os
import sys
from typing import Dict, Any, List

sys.path.append(os.path.join(os.path.dirname(__file__), "../2-FileTracker"))

from behavior_analysis_tools import (
    resolve_full_path,
    read_recording_start_time,
    normalize_timestamp_display,
    build_sensitive_operation_record,
    build_sensitive_operation_dedup_key,
)


def extract_hidden_transformed_paths(module2_result: Dict[str, Any]) -> List[str]:
    """
    直接从模块2结果提取变换后文件路径列表（按模块2输出顺序）
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
    去重键：同时间 + 同文件 + 同操作

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
