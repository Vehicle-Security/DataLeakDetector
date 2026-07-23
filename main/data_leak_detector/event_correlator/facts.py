"""将已关联证据转换为符号化 LeakReasoner 事实。

本模块是事件关联与污点传播之间的桥梁。它把关系生成保持为确定性且可审阅的逻辑，
而不是把它隐藏到推理器或图写入器里。
"""

from __future__ import annotations

from ..io import normalize_path, parse_timestamp_ms
from ..models import CorrelatedEvent, DatalogFact, UploadCandidate
from ..evidence_semantics import is_confirmed_risk_level
from .lineage import Lineage

_LINEAGE_PROCESS = "case_lineage"


def build_datalog_facts(
    correlated: list[CorrelatedEvent],
    uploads: list[UploadCandidate],
    lineage: Lineage,
    *,
    case_id: str = "",
) -> list[DatalogFact]:
    """将绑定后的证据转换为 LeakReasoner 的符号事实。"""

    facts: list[DatalogFact] = []
    opened: set[tuple[str, str]] = set()
    transfers: set[tuple[str, str, str]] = set()
    lineage_events: dict[str, tuple[int, int, CorrelatedEvent]] = {}
    source_events: dict[str, tuple[int, int, CorrelatedEvent]] = {}

    for index, event in enumerate(correlated):
        timestamp = parse_timestamp_ms(event.timestamp)
        source_key = normalize_path(event.original_file).lower()
        if source_key:
            candidate = (timestamp, index, event)
            current = source_events.get(source_key)
            if current is None or _event_rank(candidate) < _event_rank(current):
                source_events[source_key] = candidate
        current_key = normalize_path(event.current_file).lower()
        if (
            current_key
            and not _same_artifact_path(event.original_file, event.current_file)
            and (
                _is_lineage_transfer_event(event)
                or "cloud_sync_directory_transfer" in event.join_reasons
            )
        ):
            candidate = (timestamp, index, event)
            current = lineage_events.get(current_key)
            if current is None or _event_rank(candidate) < _event_rank(current):
                lineage_events[current_key] = candidate

    # File lineage spans applications and recording sessions. The canonical
    # process carries that artifact history until an upload accesses the file.
    for _, _, event in source_events.values():
        timestamp = parse_timestamp_ms(event.timestamp)
        if _is_late_identity_for_visual_upload(event, uploads):
            # A confirmed visual upload/share is the authoritative sink
            # evidence; a later monitor event only supplies exact file
            # identity. Do not let its timestamp reverse the leak path.
            timestamp = 0
        facts.append(
            DatalogFact(
                "OpenFile",
                (f"{event.event_id}:open", _LINEAGE_PROCESS, event.original_file, timestamp),
            )
        )

    for event in correlated:
        proc = event.app_name or "unknown"
        timestamp = parse_timestamp_ms(event.timestamp)
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
                        timestamp,
                    ),
                )
            )

        open_key = (proc, event.original_file)
        if event.original_file and open_key not in opened and not _is_external_sink_event(event):
            facts.append(DatalogFact("OpenFile", (f"{event.event_id}:open:app", proc, event.original_file, timestamp)))
            opened.add(open_key)

        _add_transfer_chain(
            facts,
            transfers,
            proc=proc,
            original=event.original_file,
            current=event.current_file,
            lineage=lineage,
            fact_prefix=event.event_id,
            timestamp=timestamp,
        )

    for derived, source in lineage.direct.items():
        key = (_LINEAGE_PROCESS, source, derived)
        if key not in transfers:
            lineage_event = lineage_events.get(normalize_path(derived).lower())
            event_id = lineage_event[2].event_id if lineage_event else f"lineage:{len(facts)}"
            # A VLM timestamp identifies an evidence frame, not necessarily
            # the precise filesystem transition. Retain its lineage edge but
            # leave ordering unknown so a nearby authoritative log edge is
            # not rejected solely because the frame was sampled early.
            lineage_timestamp = (
                0
                if lineage_event is not None and lineage_event[2].event_type == "visual_observation"
                else (lineage_event[0] if lineage_event else 0)
            )
            facts.append(
                DatalogFact(
                    "TransferFile",
                    (
                        f"{event_id}:transfer:lineage",
                        _LINEAGE_PROCESS,
                        source,
                        derived,
                        lineage_timestamp,
                    ),
                )
            )
            transfers.add(key)

    for upload in uploads:
        proc = upload.app_name or "unknown"
        timestamp = parse_timestamp_ms(upload.timestamp)
        if not normalize_path(upload.original_file):
            facts.append(
                DatalogFact(
                    "SuspiciousBehavior",
                    (
                        f"{upload.candidate_id}:unbound_upload_candidate",
                        proc,
                        "",
                        upload.current_file,
                        upload.sink_type,
                        f"unbound_{upload.risk_level}",
                        timestamp,
                    ),
                )
            )
            continue
        # Leak the object that actually reached the sink. Keeping the derived
        # object here forces Datalog to traverse original -> derived -> sink
        # instead of collapsing a lineage chain into a direct source leak.
        leaked_file = lineage.resolve_artifact(upload.current_file or upload.original_file)
        _add_transfer_chain(
            facts,
            transfers,
            proc=_LINEAGE_PROCESS,
            original=upload.original_file,
            current=leaked_file,
            lineage=lineage,
            fact_prefix=f"{upload.candidate_id}:upload_bind",
            timestamp=timestamp,
        )
        facts.append(
            DatalogFact(
                "UploadBinding",
                (
                    f"{upload.candidate_id}:binding",
                    f"{upload.candidate_id}:leak",
                    upload.original_file,
                    leaked_file,
                    timestamp,
                ),
            )
        )
        facts.append(
            DatalogFact(
                "CrossProcessTransfer",
                (f"{upload.candidate_id}:access", _LINEAGE_PROCESS, proc, leaked_file, timestamp),
            )
        )
        if is_confirmed_risk_level(upload.risk_level):
            share_start = (
                parse_timestamp_ms(upload.active_start_timestamp)
                if upload.sink_type == "screen_share"
                else 0
            )
            facts.append(
                DatalogFact(
                    "LeakFile",
                    (f"{upload.candidate_id}:leak", proc, leaked_file, upload.sink_type, timestamp, share_start),
                )
            )
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
                        timestamp,
                    ),
                )
            )
    return [DatalogFact(item.relation, item.args, case_id=case_id) for item in facts]


def _add_transfer_chain(
    facts: list[DatalogFact],
    transfers: set[tuple[str, str, str]],
    *,
    proc: str,
    original: str,
    current: str,
    lineage: Lineage,
    fact_prefix: str,
    timestamp: int,
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
        facts.append(
            DatalogFact("TransferFile", (f"{fact_prefix}:transfer:{index}", proc, source, derived, timestamp))
        )
        transfers.add(key)


def _same_artifact_path(left: str, right: str) -> bool:
    return normalize_path(left).lower() == normalize_path(right).lower()


def _event_rank(item: tuple[int, int, CorrelatedEvent]) -> tuple[int, int]:
    timestamp, index, _ = item
    event = item[2]
    # VLM observations are timestamped by the selected keyframe. A nearby
    # filesystem log is a more reliable ordering signal for canonical lineage,
    # even when the frame happened to be sampled a few seconds earlier.
    evidence_rank = 1 if event.event_type == "visual_observation" else 0
    return evidence_rank, (timestamp if timestamp > 0 else 2**63 - 1), index


def _is_lineage_transfer_event(event: CorrelatedEvent) -> bool:
    return not _is_external_sink_event(event) and (
        event.operation_type == "file_or_content_transfer"
        or event.behavior_category == "hidden_transformation_candidate"
    )


def _is_external_sink_event(event: CorrelatedEvent) -> bool:
    return event.operation_type == "external_sink_interaction" or event.behavior_category == "data_exfiltration_candidate"


def _is_late_identity_for_visual_upload(
    event: CorrelatedEvent,
    uploads: list[UploadCandidate],
) -> bool:
    event_time = parse_timestamp_ms(event.timestamp)
    if not event_time or not event.original_file:
        return False
    source_key = normalize_path(event.original_file).lower()
    for upload in uploads:
        upload_time = parse_timestamp_ms(upload.timestamp)
        if (
            not is_confirmed_risk_level(upload.risk_level)
            or not any(ref.startswith("frame:") for ref in upload.evidence_refs)
            or normalize_path(upload.original_file).lower() != source_key
            or not (0 < event_time - upload_time <= 15_000)
        ):
            continue
        return True
    return False


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
