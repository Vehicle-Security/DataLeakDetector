from __future__ import annotations

from pathlib import Path
from typing import Any

from ..io import flatten_text, looks_sensitive, normalize_path
from ..models import FrameObservation, LogEvent


VISUAL_ACTION_TOKENS = (
    "upload",
    "send",
    "share",
    "attach",
    "clipboard",
    "copy",
    "paste",
    "screenshot",
    "screen",
    "meeting",
    "上传",
    "发送",
    "分享",
    "附件",
    "剪贴板",
    "复制",
    "粘贴",
    "截图",
    "屏幕",
    "会议",
)


def analyze_video_behavior(
    video_path: str | Path = "",
    *,
    logs: list[LogEvent] | None = None,
    sensitive_files: list[str] | None = None,
    observations_file: str | Path | None = None,
    **_: Any,
) -> dict[str, Any]:
    """
    Produce frame observations.

    The function is intentionally an adapter-shaped boundary: real OCR/VLM can
    be plugged in later, while the current implementation provides deterministic
    observations from nearby log context so the rest of the system remains
    testable and readable.
    """

    observations: list[FrameObservation] = []
    if observations_file:
        observations.extend(_load_observations(observations_file))

    sensitive_files = [normalize_path(item) for item in sensitive_files or []]
    for event in logs or []:
        text = flatten_text(event.raw).lower()
        if not _is_visual_candidate(text, event.file_path, sensitive_files):
            continue
        observations.append(
            FrameObservation(
                observation_id=f"visual_{len(observations)}",
                start_ms=max(event.timestamp_ms - 2000, 0),
                end_ms=event.timestamp_ms + 2000 if event.timestamp_ms else 0,
                app_name=event.app_name or event.process_name,
                operation_type=_infer_operation(text, event.event_type),
                resource=event.file_path,
                related_resources=[event.file_path] if event.file_path else [],
                description=f"Log-anchored visual review candidate near {event.event_type}",
                confidence=0.65,
                source="log_anchored_frame_fallback",
            )
        )

    return {
        "video_file": str(video_path or ""),
        "observations": [item.to_dict() for item in observations],
        "statistics": {
            "observations": len(observations),
            "mode": "deterministic_adapter",
        },
    }


def _is_visual_candidate(text: str, file_path: str, sensitive_files: list[str]) -> bool:
    has_action = any(token.lower() in text for token in VISUAL_ACTION_TOKENS)
    has_sensitive = looks_sensitive(file_path) or any(item and item.lower() in normalize_path(file_path).lower() for item in sensitive_files)
    return has_action or has_sensitive


def _infer_operation(text: str, fallback: str) -> str:
    for token in VISUAL_ACTION_TOKENS:
        if token.lower() in text:
            return token
    return fallback or "visual_observation"


def _load_observations(path: str | Path) -> list[FrameObservation]:
    import json

    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, dict):
        data = data.get("observations", [])
    observations: list[FrameObservation] = []
    for index, item in enumerate(data if isinstance(data, list) else []):
        if not isinstance(item, dict):
            continue
        observations.append(
            FrameObservation(
                observation_id=str(item.get("observation_id") or f"obs_{index}"),
                start_ms=int(item.get("start_ms") or 0),
                end_ms=int(item.get("end_ms") or 0),
                app_name=str(item.get("app_name") or ""),
                operation_type=str(item.get("operation_type") or ""),
                resource=normalize_path(item.get("resource") or ""),
                related_resources=[normalize_path(value) for value in item.get("related_resources") or []],
                description=str(item.get("description") or ""),
                confidence=float(item.get("confidence") or 0.0),
                source=str(item.get("source") or "observation_file"),
            )
        )
    return observations


__all__ = ["analyze_video_behavior"]
