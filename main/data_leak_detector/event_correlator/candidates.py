"""上传和外部汇聚点候选项提取。

已关联的事件仍然只是通用证据。本模块会把它们收窄为可能的外发泄露动作，并附上汇聚点类型、
风险等级和置信度，供推理器和最终报告使用。
"""

from __future__ import annotations

from dataclasses import replace

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
        if (
            event.behavior_category != "data_exfiltration_candidate"
            and event.operation_type != "external_sink_interaction"
            and not contains_any(text, SINK_TOKENS)
        ):
            continue
        uploads.append(
            UploadCandidate(
                candidate_id=f"upload_{len(uploads)}",
                timestamp=event.timestamp,
                app_name=event.app_name,
                original_file=event.original_file,
                current_file=event.current_file or event.original_file,
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
