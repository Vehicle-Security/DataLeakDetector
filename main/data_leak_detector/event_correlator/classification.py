"""用于将原始事件绑定到行为的分类辅助函数。

关联器会调用这些小函数来判断应用类型、源元数据、操作标签以及可能的源文件猜测。把它们
放在这里，可以避免工作流对象变成一堆互不相关的字符串规则。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..io import normalize_path
from ..policy import SINK_TOKENS, TRANSFER_TOKENS, contains_any


def classify_frontend_app(app_name: str, window_title: str = "") -> str:
    text = f"{app_name} {window_title}"
    if contains_any(text, SINK_TOKENS):
        return "external_sink"
    return "local_app"


def original_file_from_metadata(record: dict[str, Any]) -> str:
    upload = record.get("upload_detection") if isinstance(record.get("upload_detection"), dict) else {}
    extra = record.get("extra") if isinstance(record.get("extra"), dict) else {}
    return normalize_path(
        record.get("source_file")
        or record.get("original_file")
        or record.get("src_path")
        or record.get("source_path")
        or record.get("from_path")
        or extra.get("source_file")
        or extra.get("original_file")
        or extra.get("src_path")
        or extra.get("source_path")
        or upload.get("original_file")
        or ""
    )


def target_file_from_metadata(record: dict[str, Any]) -> str:
    upload = record.get("upload_detection") if isinstance(record.get("upload_detection"), dict) else {}
    extra = record.get("extra") if isinstance(record.get("extra"), dict) else {}
    return normalize_path(
        record.get("destination_path")
        or record.get("target_file")
        or record.get("derived_file")
        or record.get("dst_path")
        or record.get("target_path")
        or record.get("to_path")
        or record.get("output_file")
        or record.get("output_path")
        or extra.get("destination_path")
        or extra.get("target_file")
        or extra.get("derived_file")
        or extra.get("dst_path")
        or extra.get("target_path")
        or extra.get("output_file")
        or extra.get("output_path")
        or upload.get("temp_file")
        or record.get("file_path")
        or ""
    )


def behavior_category(text: str) -> str:
    if "external_sink_interaction" in text:
        return "data_exfiltration_candidate"
    if "file_or_content_transfer" in text:
        return "hidden_transformation_candidate"
    if contains_any(text, SINK_TOKENS):
        return "data_exfiltration_candidate"
    if contains_any(text, TRANSFER_TOKENS):
        return "hidden_transformation_candidate"
    return "sensitive_access"


def operation_from_text(text: str, fallback: str) -> str:
    if "external_sink_interaction" in text:
        return "external_sink_interaction"
    if "file_or_content_transfer" in text:
        return "file_or_content_transfer"
    if contains_any(text, SINK_TOKENS):
        return "external_sink_interaction"
    if contains_any(text, TRANSFER_TOKENS):
        return "file_or_content_transfer"
    return fallback or "sensitive_access"


def guess_source_by_stem(file_path: str, known_files: list[str]) -> str:
    stem = Path(normalize_path(file_path)).stem.lower()
    if not stem:
        return ""
    for known in known_files:
        known_stem = Path(normalize_path(known)).stem.lower()
        if known_stem and (stem.startswith(known_stem) or known_stem.startswith(stem)):
            return known
    return ""
