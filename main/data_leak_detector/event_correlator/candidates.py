"""上传和外部汇聚点候选项提取。

已关联的事件仍然只是通用证据。本模块会把它们收窄为可能的外发泄露动作，并附上汇聚点类型、
风险等级和置信度，供推理器和最终报告使用。
"""

from __future__ import annotations

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
        if event.behavior_category != "data_exfiltration_candidate" and not contains_any(text, SINK_TOKENS):
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
    return uploads
