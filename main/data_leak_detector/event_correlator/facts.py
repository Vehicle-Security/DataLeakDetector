"""Conversion from correlated evidence to symbolic LeakReasoner facts.

This module is the bridge between event correlation and taint propagation. It
keeps relation generation deterministic and reviewable, instead of hiding it in
the reasoner or graph writer.
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
    """Convert bound evidence into symbolic facts for LeakReasoner."""

    facts: list[DatalogFact] = []
    opened: set[tuple[str, str]] = set()
    transfers: set[tuple[str, str, str]] = set()

    for event in correlated:
        proc = event.app_name or "unknown"
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
        if not same_file(upload.original_file, upload.current_file):
            facts.append(DatalogFact("CrossProcessTransfer", (f"{upload.candidate_id}:bind", "system", proc, upload.current_file, 0)))
        facts.append(DatalogFact("LeakFile", (f"{upload.candidate_id}:leak", proc, upload.current_file, upload.sink_type, 0)))
    return facts
