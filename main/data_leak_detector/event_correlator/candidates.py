"""上传和外部汇聚点候选项提取。

已关联的事件仍然只是通用证据。本模块会把它们收窄为可能的外发泄露动作，并附上汇聚点类型、
风险等级和置信度，供推理器和最终报告使用。
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from ..io import normalize_path
from ..models import CorrelatedEvent, UploadCandidate
from ..policy import SINK_TOKENS, classify_sink, contains_any, risk_level_for_sink


def build_upload_candidates(
    correlated: list[CorrelatedEvent],
    *,
    default_confidence: float,
) -> list[UploadCandidate]:
    """从已关联的敏感事件中创建外部汇聚点候选项。"""

    uploads: list[UploadCandidate] = []
    for event in correlated:
        text = f"{event.event_type} {event.operation_type} {event.app_name} {event.behavior_category}"
        current_file = event.current_file or event.original_file
        if not _is_explicit_upload_event(event, text):
            continue
        if _is_noise_or_placeholder_path(current_file):
            continue
        uploads.append(
            UploadCandidate(
                candidate_id=f"upload_{len(uploads)}",
                timestamp=event.timestamp,
                app_name=event.app_name,
                original_file=event.original_file,
                current_file=current_file,
                sink_type=classify_sink(text),
                risk_level=risk_level_for_sink(text),
                confidence=max(event.confidence, default_confidence),
                evidence_refs=event.evidence_refs,
            )
        )
    return _dedupe_upload_candidates(uploads)


def _dedupe_upload_candidates(candidates: list[UploadCandidate]) -> list[UploadCandidate]:
    merged: dict[tuple[str, str], UploadCandidate] = {}
    for candidate in candidates:
        key = (normalize_path(candidate.original_file).lower(), normalize_path(candidate.current_file).lower())
        previous = merged.get(key)
        if previous is None:
            merged[key] = candidate
            continue
        winner, loser = _prefer_upload_candidate(previous, candidate)
        refs = tuple(dict.fromkeys([*winner.evidence_refs, *loser.evidence_refs]))
        merged[key] = replace(winner, confidence=max(winner.confidence, loser.confidence), evidence_refs=refs)
    return [replace(candidate, candidate_id=f"upload_{index}") for index, candidate in enumerate(merged.values())]


def _prefer_upload_candidate(left: UploadCandidate, right: UploadCandidate) -> tuple[UploadCandidate, UploadCandidate]:
    left_has_log = any(ref.startswith("log:") for ref in left.evidence_refs)
    right_has_log = any(ref.startswith("log:") for ref in right.evidence_refs)
    if left_has_log != right_has_log:
        return (left, right) if left_has_log else (right, left)
    if right.confidence > left.confidence:
        return right, left
    return left, right


def _is_explicit_upload_event(event: CorrelatedEvent, text: str) -> bool:
    if event.operation_type == "file_or_content_transfer":
        return False
    if event.operation_type != "external_sink_interaction" and not contains_any(text, SINK_TOKENS):
        return False
    reasons = set(event.join_reasons)
    if {"explicit_sink_log", "ocr_sink_context", "visual_only"} & reasons:
        return True
    return any(ref.startswith("frame:vlm") for ref in event.evidence_refs) and event.operation_type == "external_sink_interaction"


def _is_noise_or_placeholder_path(value: str) -> bool:
    path = normalize_path(value).strip().strip("\"'")
    lowered = path.lower()
    if not lowered or lowered in {"n/a", "na", "none", "null", "unknown", "-", "无", "空"} or lowered.startswith("n/a "):
        return True
    markers = (
        "/appdata/local/google/chrome/user data/",
        "/appdata/local/microsoft/edge/user data/",
        "/appdata/roaming/cursor/",
        "/appdata/roaming/code/",
        "/appdata/roaming/larkshell/",
        "/tencent files/",
        "/nt_qq/",
        "/driverstore/",
        "/screenmonitor/",
        "/recordings/session_",
        "/logs/",
        "/cache/",
        "/cache_data/",
        "/cacheddata/",
    )
    if any(marker in lowered for marker in markers):
        return True
    suffix = Path(lowered).suffix
    if suffix in {".tmp", ".temp", ".log", ".xlog", ".db", ".db-journal", ".db-wal", ".db-shm"}:
        return True
    return False
