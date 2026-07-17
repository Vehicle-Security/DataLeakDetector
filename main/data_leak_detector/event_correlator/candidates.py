"""Extract upload and external-sink candidates from correlated events."""

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
    """Build external-sink candidates from correlated sensitive events."""

    uploads: list[UploadCandidate] = []
    for event in correlated:
        text = (
            f"{event.event_type} {event.operation_type} {event.app_name} "
            f"{event.behavior_category} {' '.join(event.join_reasons)}"
        )
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
                sink_type=_upload_sink_type(event, current_file, text),
                risk_level=_upload_risk_level(event, text),
                confidence=max(event.confidence, default_confidence),
                evidence_refs=event.evidence_refs,
            )
        )
    return _dedupe_upload_candidates(uploads)


def _dedupe_upload_candidates(candidates: list[UploadCandidate]) -> list[UploadCandidate]:
    merged: dict[tuple[str, str, str], UploadCandidate] = {}
    for candidate in candidates:
        key = (
            normalize_path(candidate.original_file).lower(),
            normalize_path(candidate.current_file).lower(),
            candidate.sink_type,
        )
        previous = merged.get(key)
        if previous is None:
            merged[key] = candidate
            continue
        winner, loser = _prefer_upload_candidate(previous, candidate)
        refs = tuple(dict.fromkeys([*winner.evidence_refs, *loser.evidence_refs]))
        merged[key] = replace(winner, confidence=max(winner.confidence, loser.confidence), evidence_refs=refs)
    return [replace(candidate, candidate_id=f"upload_{index}") for index, candidate in enumerate(merged.values())]


def _prefer_upload_candidate(left: UploadCandidate, right: UploadCandidate) -> tuple[UploadCandidate, UploadCandidate]:
    risk_rank = {
        "selected_or_attached": 1,
        "in_progress": 2,
        "content_exposed": 3,
        "completed": 4,
    }
    left_rank = risk_rank.get(left.risk_level, 0)
    right_rank = risk_rank.get(right.risk_level, 0)
    if left_rank != right_rank:
        return (left, right) if left_rank > right_rank else (right, left)
    if right.confidence != left.confidence:
        return (left, right) if left.confidence > right.confidence else (right, left)
    left_has_log = any(ref.startswith("log:") for ref in left.evidence_refs)
    right_has_log = any(ref.startswith("log:") for ref in right.evidence_refs)
    if left_has_log != right_has_log:
        return (left, right) if left_has_log else (right, left)
    return left, right


def _is_explicit_upload_event(event: CorrelatedEvent, text: str) -> bool:
    if event.operation_type == "file_or_content_transfer":
        return False
    if event.operation_type != "external_sink_interaction" and not contains_any(text, SINK_TOKENS):
        return False
    reasons = set(event.join_reasons)
    if "removable_media_sink" in reasons:
        return True
    if {"explicit_sink_log", "visual_sink_context", "visual_only"} & reasons:
        return True
    return any(ref.startswith("frame:vlm") for ref in event.evidence_refs) and event.operation_type == "external_sink_interaction"


def _upload_risk_level(event: CorrelatedEvent, text: str) -> str:
    if "action_status:failed" in text or "failed_external_attempt" in text:
        return "selected_or_attached"
    if "action_status:selected" in text or "action_status:submitted" in text:
        return "selected_or_attached"
    if "action_status:in_progress" in text:
        return "in_progress"
    if "action_status:completed" in text:
        return "completed"
    if any(ref.startswith("frame:vlm") for ref in event.evidence_refs):
        return "content_exposed"
    if "removable_media_sink" in event.join_reasons:
        return "completed"
    if event.event_type in {"file_upload", "upload", "uploaded", "upload_complete", "send_click"}:
        return "completed"
    if event.event_type == "file_selected":
        return "selected_or_attached"
    return risk_level_for_sink(text)


def _upload_sink_type(event: CorrelatedEvent, current_file: str, text: str) -> str:
    combined = f"{text} {event.app_name} {event.current_file} {current_file} {' '.join(event.join_reasons)}"
    if contains_any(
        combined,
        (
            "usb",
            "removable",
            "removable media",
            "removable drive",
            "flash drive",
            "thumb drive",
            "u disk",
            "udisk",
            "external drive",
            "可移动",
            "可移动存储",
            "可移动磁盘",
            "移动磁盘",
            "移动硬盘",
            "u盘",
        ),
    ):
        return "removable_media"
    return classify_sink(combined)


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


