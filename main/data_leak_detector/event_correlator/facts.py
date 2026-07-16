"""将已关联证据转换为符号化 LeakReasoner 事实。

本模块是事件关联与污点传播之间的桥梁。它把关系生成保持为确定性且可审阅的逻辑，
而不是把它隐藏到推理器或图写入器里。
"""

from __future__ import annotations

from ..io import normalize_path
from ..models import CorrelatedEvent, DatalogFact, UploadCandidate
from ..evidence_semantics import is_confirmed_risk_level
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
        if event.original_file and open_key not in opened:
            facts.append(DatalogFact("OpenFile", (f"{event.event_id}:open", proc, event.original_file, 0)))
            opened.add(open_key)

        _add_transfer_chain(
            facts,
            transfers,
            proc=proc,
            original=event.original_file,
            current=event.current_file,
            lineage=lineage,
            fact_prefix=event.event_id,
        )

    for derived, source in lineage.direct.items():
        key = ("system", source, derived)
        if key not in transfers:
            facts.append(DatalogFact("TransferFile", (f"lineage:{len(facts)}", "system", source, derived, 0)))
            transfers.add(key)

    for upload in uploads:
        proc = upload.app_name or "unknown"
        # Leak the object that actually reached the sink. Keeping the derived
        # object here forces Datalog to traverse original -> derived -> sink
        # instead of collapsing a lineage chain into a direct source leak.
        leaked_file = upload.current_file or upload.original_file
        _add_transfer_chain(
            facts,
            transfers,
            proc=proc,
            original=upload.original_file,
            current=upload.current_file,
            lineage=lineage,
            fact_prefix=f"{upload.candidate_id}:upload_bind",
        )
        if is_confirmed_risk_level(upload.risk_level):
            facts.append(DatalogFact("LeakFile", (f"{upload.candidate_id}:leak", proc, leaked_file, upload.sink_type, 0)))
        else:
            facts.append(
                DatalogFact(
                    "SuspiciousBehavior",
                    (
                        f"{upload.candidate_id}:upload_candidate",
                        proc,
                        upload.original_file,
                        upload.current_file,
                        upload.sink_type,
                        upload.risk_level,
                        0,
                    ),
                )
            )
    return facts


def _add_transfer_chain(
    facts: list[DatalogFact],
    transfers: set[tuple[str, str, str]],
    *,
    proc: str,
    original: str,
    current: str,
    lineage: Lineage,
    fact_prefix: str,
) -> None:
    if not original or not current or _same_artifact_path(original, current):
        return
    reverse_chain = list(reversed(lineage.chain(current)))
    if not reverse_chain or not _same_artifact_path(reverse_chain[0], original):
        reverse_chain = [original, current]
    for index, (source, derived) in enumerate(zip(reverse_chain, reverse_chain[1:], strict=False)):
        key = (proc, source, derived)
        if key in transfers:
            continue
        facts.append(DatalogFact("TransferFile", (f"{fact_prefix}:transfer:{index}", proc, source, derived, 0)))
        transfers.add(key)


def _same_artifact_path(left: str, right: str) -> bool:
    return normalize_path(left).lower() == normalize_path(right).lower()


def _is_suspicious_event(event: CorrelatedEvent) -> bool:
    text = f"{event.operation_type} {event.behavior_category}".lower()
    return (
        event.behavior_category
        in {
            "hidden_transformation_candidate",
            "unknown_risk",
            "failed_external_attempt",
            "selected_external_attempt",
        }
        or event.operation_type == "file_or_content_transfer"
        or "hidden" in text
        or "unknown_risk" in text
    )
