"""将已关联证据转换为符号化 LeakReasoner 事实。

本模块是事件关联与污点传播之间的桥梁。它把关系生成保持为确定性且可审阅的逻辑，
而不是把它隐藏到推理器或图写入器里。
"""

from __future__ import annotations

from ..io import same_file
from ..models import CorrelatedEvent, DatalogFact, UploadCandidate
from .lineage import Lineage


def build_datalog_facts(
    correlated: list[CorrelatedEvent],
    uploads: list[UploadCandidate],
    lineage: Lineage,
) -> list[DatalogFact]:
    """将绑定后的证据转换为 LeakReasoner 的符号事实。"""

    facts: list[DatalogFact] = []
    opened: set[tuple[str, str]] = set()
    transfers: set[tuple[str, str, str]] = set()

    for event in correlated:
        proc = event.app_name or "unknown"
        if _is_suspicious_event(event):
            facts.append(
                DatalogFact(
                    "SuspiciousBehavior",
                    (
                        f"{event.event_id}:suspicious",
                        proc,
                        event.original_file,
                        event.current_file,
                        event.operation_type,
                        event.behavior_category,
                        0,
                    ),
                )
            )

        open_key = (proc, event.original_file)
        if open_key not in opened:
            facts.append(DatalogFact("OpenFile", (f"{event.event_id}:open", proc, event.original_file, 0)))
            opened.add(open_key)

        if not same_file(event.original_file, event.current_file):
            key = (proc, event.original_file, event.current_file)
            if key not in transfers:
                facts.append(DatalogFact("TransferFile", (f"{event.event_id}:transfer", proc, event.original_file, event.current_file, 0)))
                transfers.add(key)

    for derived, source in lineage.direct.items():
        key = ("system", source, derived)
        if key not in transfers:
            facts.append(DatalogFact("TransferFile", (f"lineage:{len(facts)}", "system", source, derived, 0)))
            transfers.add(key)

    for upload in uploads:
        proc = upload.app_name or "unknown"
        leaked_file = upload.original_file or upload.current_file
        if not same_file(upload.original_file, upload.current_file):
            facts.append(DatalogFact("TransferFile", (f"{upload.candidate_id}:upload_bind", proc, upload.original_file, upload.current_file, 0)))
        facts.append(DatalogFact("LeakFile", (f"{upload.candidate_id}:leak", proc, leaked_file, upload.sink_type, 0)))
    return facts


def _is_suspicious_event(event: CorrelatedEvent) -> bool:
    text = f"{event.operation_type} {event.behavior_category}".lower()
    return (
        event.behavior_category in {"hidden_transformation_candidate", "unknown_risk"}
        or event.operation_type == "file_or_content_transfer"
        or "hidden" in text
        or "unknown_risk" in text
    )
