"""Deterministic FrameAnalyzer implementation.

For now, the analyzer turns logs and optional precomputed visual observations
into review windows. This keeps the pipeline runnable in local tests while
preserving a clean insertion point for future OCR/VLM frame analysis.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..io import flatten_text, looks_sensitive, normalize_path
from ..models import FrameObservation, LogEvent
from ..policy import SINK_TOKENS, TRANSFER_TOKENS, contains_any


def analyze_video_behavior(
    video_path: str | Path = "",
    *,
    logs: list[LogEvent] | None = None,
    sensitive_files: list[str] | None = None,
    observations_file: str | Path | None = None,
    **_: Any,
) -> dict[str, Any]:
    """
    Produce frame-level behavior observations.

    The default implementation is deterministic: it accepts optional precomputed
    OCR/VLM observations, then augments them with log-anchored review windows
    around sensitive files, transfers, and external sinks.
    """

    observations: list[FrameObservation] = []
    if observations_file:
        observations.extend(load_observations(observations_file))

    sensitive_files = tuple(normalize_path(item) for item in sensitive_files or [])
    for event in logs or []:
        text = flatten_text(event.raw)
        if not should_review(event, text, sensitive_files):
            continue

        observations.append(
            FrameObservation(
                observation_id=f"obs_{len(observations)}",
                start_ms=max(event.timestamp_ms - 2000, 0),
                end_ms=event.timestamp_ms + 2000 if event.timestamp_ms else 0,
                app_name=event.app_name or event.process_name,
                operation_type=infer_operation(text, event.event_type),
                resource=event.file_path,
                related_resources=(event.file_path,) if event.file_path else (),
                description=f"Review window around {event.event_type}",
                confidence=0.65 if event.file_path else 0.55,
                source="log_anchored",
            )
        )

    return {
        "video_file": str(video_path or ""),
        "observations": [item.to_dict() for item in observations],
        "statistics": {
            "mode": "deterministic_log_anchored",
            "observations": len(observations),
        },
    }


def should_review(event: LogEvent, text: str, sensitive_files: tuple[str, ...]) -> bool:
    normalized_file = normalize_path(event.file_path).lower()
    explicit_sensitive = any(item and item.lower() in normalized_file for item in sensitive_files)
    sensitive_context = explicit_sensitive or looks_sensitive(event.file_path) or looks_sensitive(text)
    activity_context = contains_any(text, TRANSFER_TOKENS) or contains_any(text, SINK_TOKENS)
    return sensitive_context or activity_context


def infer_operation(text: str, fallback: str) -> str:
    lowered = text.lower()
    if contains_any(lowered, SINK_TOKENS):
        return "external_sink_interaction"
    if contains_any(lowered, TRANSFER_TOKENS):
        return "file_or_content_transfer"
    return fallback or "visual_review"


def load_observations(path: str | Path) -> list[FrameObservation]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, dict):
        data = data.get("observations", [])

    observations: list[FrameObservation] = []
    for index, item in enumerate(data if isinstance(data, list) else []):
        if not isinstance(item, dict):
            continue
        observations.append(
            FrameObservation(
                observation_id=str(item.get("observation_id") or f"file_obs_{index}"),
                start_ms=int(item.get("start_ms") or 0),
                end_ms=int(item.get("end_ms") or item.get("start_ms") or 0),
                app_name=str(item.get("app_name") or ""),
                operation_type=str(item.get("operation_type") or item.get("operation") or ""),
                resource=normalize_path(item.get("resource") or item.get("file_path") or ""),
                related_resources=tuple(normalize_path(value) for value in item.get("related_resources") or ()),
                description=str(item.get("description") or ""),
                confidence=float(item.get("confidence") or 0.0),
                source=str(item.get("source") or "observation_file"),
            )
        )
    return observations
