"""Upload and external-sink candidate extraction.

Correlated events are still generic evidence. This module narrows them into
possible outbound leak actions, attaching sink type, risk level, and confidence
for the reasoner and final report.
"""

from __future__ import annotations

from ..models import CorrelatedEvent, UploadCandidate
from ..policy import SINK_TOKENS, classify_sink, contains_any, risk_level_for_sink


def build_upload_candidates(
    correlated: list[CorrelatedEvent],
    *,
    default_confidence: float,
) -> list[UploadCandidate]:
    """Create external sink candidates from correlated sensitive events."""

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
