"""FrameAnalyzer with deterministic, OCR, and VLM-assisted evidence paths."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..io import flatten_text, looks_sensitive, normalize_path
from ..models import FrameObservation, LogEvent
from ..policy import SINK_TOKENS, TRANSFER_TOKENS, contains_any
from .apps import identify_frontend_app
from .config import VisionConfig
from .frames import build_analysis_windows, select_keyframes
from .ocr import OcrResult, run_ocr
from .parser import parse_vlm_response, vision_events_to_observations
from .vlm import OpenAICompatibleVlmClient, choose_vlm_frames


def analyze_video_behavior(
    video_path: str | Path = "",
    *,
    logs: list[LogEvent] | None = None,
    sensitive_files: list[str] | None = None,
    observations_file: str | Path | None = None,
    vision_enabled: bool | None = None,
    vision_mode: str | None = None,
    max_vlm_frames: int | None = None,
    **_: Any,
) -> dict[str, Any]:
    """Produce frame-level behavior observations for downstream correlation.

    The analyzer always keeps the deterministic log-anchored path. When vision
    is enabled, it adds a non-uniform keyframe pipeline:

    1. mine suspicious windows from logs;
    2. select visually changed keyframes instead of uniform sampling;
    3. run OCR over all selected keyframes;
    4. send only low-confidence or suspicious OCR frames to VLM;
    5. normalize OCR/VLM output into FrameObservation records.
    """

    config = VisionConfig.from_env().with_overrides(
        enabled=vision_enabled,
        mode=vision_mode,
        max_vlm_frames=max_vlm_frames,
    )
    logs = logs or []
    sensitive_files = [normalize_path(item) for item in sensitive_files or []]
    observations: list[FrameObservation] = []
    warnings: list[str] = []
    errors: list[str] = []

    if observations_file:
        observations.extend(load_observations(observations_file))

    observations.extend(_log_anchored_observations(logs, sensitive_files, start_index=len(observations)))

    vision_stats = {
        "enabled": config.enabled,
        "mode": config.mode,
        "analysis_windows": 0,
        "keyframes": 0,
        "ocr_frames": 0,
        "vlm_frames": 0,
        "vlm_events": 0,
    }

    if config.enabled:
        vision_observations, vision_stats, vision_warnings, vision_errors = _run_vision_pipeline(
            video_path=video_path,
            logs=logs,
            sensitive_files=sensitive_files,
            config=config,
            start_index=len(observations),
        )
        observations.extend(vision_observations)
        warnings.extend(vision_warnings)
        errors.extend(vision_errors)

    return {
        "video_file": str(video_path or ""),
        "observations": [item.to_dict() for item in observations],
        "statistics": {
            "mode": "hybrid_log_ocr_vlm" if config.enabled else "deterministic_log_anchored",
            "observations": len(observations),
            "vision": vision_stats,
        },
        "warnings": warnings,
        "errors": errors,
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


def _log_anchored_observations(
    logs: list[LogEvent],
    sensitive_files: list[str],
    *,
    start_index: int,
) -> list[FrameObservation]:
    observations: list[FrameObservation] = []
    sensitive = tuple(sensitive_files)
    for event in logs:
        text = flatten_text(event.raw)
        if not should_review(event, text, sensitive):
            continue
        app_identity = identify_frontend_app(event.app_name or event.process_name, event.window_title, text)
        observations.append(
            FrameObservation(
                observation_id=f"obs_{start_index + len(observations)}",
                start_ms=max(event.timestamp_ms - 2000, 0),
                end_ms=event.timestamp_ms + 2000 if event.timestamp_ms else 0,
                app_name=event.app_name or event.process_name or app_identity.app_name,
                operation_type=infer_operation(text, event.event_type),
                resource=event.file_path,
                related_resources=(event.file_path,) if event.file_path else (),
                description=f"Review window around {event.event_type}; app_category={app_identity.category}; risk_hint={app_identity.risk_hint}",
                confidence=0.65 if event.file_path else 0.55,
                source="log_anchored",
            )
        )
    return observations


def _run_vision_pipeline(
    *,
    video_path: str | Path,
    logs: list[LogEvent],
    sensitive_files: list[str],
    config: VisionConfig,
    start_index: int,
) -> tuple[list[FrameObservation], dict[str, Any], list[str], list[str]]:
    observations: list[FrameObservation] = []
    warnings: list[str] = []
    errors: list[str] = []

    windows = build_analysis_windows(logs, sensitive_files, config)
    keyframes, frame_warnings = select_keyframes(video_path, windows, config)
    warnings.extend(frame_warnings)

    ocr_results = run_ocr(keyframes, config) if keyframes else []
    observations.extend(_ocr_observations(ocr_results, config, start_index=start_index))

    vlm_frames = choose_vlm_frames(
        ocr_results,
        min_confidence=config.ocr_min_confidence,
        max_frames=config.max_vlm_frames,
    )
    vlm_events = 0
    if vlm_frames and config.mode.lower() in {"hybrid", "vlm"}:
        try:
            active_apps = sorted({app for window in windows for app in window.active_apps})
            response = OpenAICompatibleVlmClient(config).analyze(
                vlm_frames,
                sensitive_files=sensitive_files,
                active_apps=active_apps,
            )
            events = parse_vlm_response(response.text, keywords=sensitive_files)
            vlm_events = len(events)
            observations.extend(
                vision_events_to_observations(events, source="vlm", start_index=start_index + len(observations))
            )
        except Exception as exc:
            errors.append(f"vlm_failed: {type(exc).__name__}: {exc}")

    stats = {
        "enabled": config.enabled,
        "mode": config.mode,
        "analysis_windows": len(windows),
        "keyframes": len(keyframes),
        "ocr_frames": len(ocr_results),
        "vlm_frames": len(vlm_frames),
        "vlm_events": vlm_events,
    }
    return observations, stats, warnings, errors


def _ocr_observations(
    results: list[OcrResult],
    config: VisionConfig,
    *,
    start_index: int,
) -> list[FrameObservation]:
    observations: list[FrameObservation] = []
    for result in results:
        if result.confidence < config.ocr_min_confidence or not result.text.strip():
            continue
        app = identify_frontend_app(ocr_text=result.text)
        operation = "external_sink_interaction" if app.risk_hint == "external_capable" and contains_any(result.text, SINK_TOKENS) else "visual_text_observed"
        observations.append(
            FrameObservation(
                observation_id=f"ocr_{start_index + len(observations)}",
                start_ms=result.frame.timestamp_ms,
                end_ms=result.frame.timestamp_ms,
                app_name=app.app_name,
                operation_type=operation,
                resource="",
                related_resources=(),
                description=f"OCR text: {result.text[:500]}; app_category={app.category}; risk_hint={app.risk_hint}",
                confidence=result.confidence,
                source=result.provider,
            )
        )
    return observations
