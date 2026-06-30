from __future__ import annotations

from typing import Any

from .utils import minute_bucket, normalize_app_name, normalize_file_path


def _score_correlated_event(event: dict[str, Any]) -> float:
    score = float(event.get("correlation_score", 0.0) or 0.0)
    score += float(event.get("confidence", 0.0) or 0.0)
    score += min(len(event.get("evidence_refs", []) or []), 5) * 0.05
    event_type = str(event.get("event_type", "") or "").strip().lower()
    if event_type in {"file_upload", "file_selected", "upload_detected"}:
        score += 0.35
    if event_type in {"created", "modified"}:
        score -= 0.15
    if event.get("status") == "ambiguous":
        score -= 0.1
    return score


def build_correlated_event_dedup_key(event: dict[str, Any]) -> str:
    return "|".join(
        [
            normalize_file_path(str(event.get("current_file", "") or "")).lower(),
            normalize_app_name(str(event.get("app_name", "") or "")).lower(),
            minute_bucket(str(event.get("timestamp", "") or "")),
            str(event.get("operation_type", "") or "").strip().lower(),
        ]
    )


def deduplicate_correlated_events(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deduped: dict[str, dict[str, Any]] = {}

    for event in events:
        key = build_correlated_event_dedup_key(event)
        existing = deduped.get(key)
        if existing is None:
            deduped[key] = event
            continue

        if _score_correlated_event(event) > _score_correlated_event(existing):
            merged_refs = list(
                dict.fromkeys((existing.get("evidence_refs", []) or []) + (event.get("evidence_refs", []) or []))
            )
            event["evidence_refs"] = merged_refs
            deduped[key] = event
            continue

        existing["evidence_refs"] = list(
            dict.fromkeys((existing.get("evidence_refs", []) or []) + (event.get("evidence_refs", []) or []))
        )

    return list(deduped.values())


def _score_upload_candidate(candidate: dict[str, Any]) -> float:
    score = float(candidate.get("confidence", 0.0) or 0.0)
    score += min(len(candidate.get("mapping_links", []) or []), 5) * 0.1
    score += min(len(candidate.get("evidence_refs", []) or []), 5) * 0.05
    return score


def build_upload_candidate_dedup_key(candidate: dict[str, Any]) -> str:
    current_files = [
        normalize_file_path(str(item or "")).lower()
        for item in candidate.get("current_files", []) or []
        if normalize_file_path(str(item or ""))
    ]
    current_files_key = ",".join(sorted(dict.fromkeys(current_files)))
    return "|".join(
        [
            normalize_app_name(str(candidate.get("app_name", "") or "")).lower(),
            normalize_file_path(str(candidate.get("original_file", "") or "")).lower(),
            str(candidate.get("sink_type", "") or "").strip().lower(),
            current_files_key,
            minute_bucket(str(candidate.get("timestamp", "") or "")),
        ]
    )


def deduplicate_upload_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deduped: dict[str, dict[str, Any]] = {}

    for candidate in candidates:
        key = build_upload_candidate_dedup_key(candidate)
        existing = deduped.get(key)
        if existing is None:
            deduped[key] = candidate
            continue

        if _score_upload_candidate(candidate) > _score_upload_candidate(existing):
            candidate["evidence_refs"] = list(
                dict.fromkeys((existing.get("evidence_refs", []) or []) + (candidate.get("evidence_refs", []) or []))
            )
            candidate["mapping_links"] = list(
                dict.fromkeys((existing.get("mapping_links", []) or []) + (candidate.get("mapping_links", []) or []))
            )
            deduped[key] = candidate
            continue

        existing["evidence_refs"] = list(
            dict.fromkeys((existing.get("evidence_refs", []) or []) + (candidate.get("evidence_refs", []) or []))
        )
        existing["mapping_links"] = list(
            dict.fromkeys((existing.get("mapping_links", []) or []) + (candidate.get("mapping_links", []) or []))
        )
        existing["current_files"] = list(
            dict.fromkeys((existing.get("current_files", []) or []) + (candidate.get("current_files", []) or []))
        )

    return list(deduped.values())
