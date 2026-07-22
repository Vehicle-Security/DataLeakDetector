"""Frame analyzer entry point for log evidence and direct-keyframe VLM."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from ..io import flatten_text, looks_sensitive, normalize_path
from ..log_mining import build_analysis_windows
from ..models import FrameObservation, LogEvent
from ..policy import SINK_TOKENS, TRANSFER_TOKENS, contains_any
from .apps import identify_frontend_app
from .artifacts import (
    export_vision_artifacts,
    load_vision_precompute,
    prepare_vlm_request_frames,
    write_json_artifact,
    write_vision_precompute,
)
from .config import VisionConfig
from .frames import AnalysisWindow, KeyFrameSelection, augment_with_video_coverage, select_keyframes_detailed
from .parser import vision_events_to_observations
from .vlm_client import choose_keyframes_for_vlm
from .vlm_dispatch import (
    build_vlm_clients,
    combine_vlm_request_metrics,
    effective_vlm_parallelism,
    run_vlm_batches,
    vlm_batch_request_summary,
    vlm_frame_batches,
    vlm_parse_artifact_payload,
    vlm_request_artifact_payload,
    vlm_response_artifact_payload,
)


def analyze_video_behavior(
    video_path: str | Path = "",
    *,
    logs: list[LogEvent] | None = None,
    sensitive_files: list[str] | None = None,
    vlm_sensitive_files: list[str] | None = None,
    observations_file: str | Path | None = None,
    vision_enabled: bool | None = None,
    max_vlm_frames: int | None = None,
    vision_precompute_file: str | Path | None = None,
    artifact_dir: str | Path | None = None,
    analysis_windows: list[AnalysisWindow] | None = None,
    log_mining: dict[str, Any] | None = None,
    debug_artifacts: bool = True,
    **_: Any,
) -> dict[str, Any]:
    """Produce frame-level behavior observations for downstream correlation."""

    config = VisionConfig.from_env().with_overrides(
        enabled=vision_enabled,
        max_vlm_frames=max_vlm_frames,
    )
    logs = logs or []
    sensitive_files = [normalize_path(item) for item in sensitive_files or []]
    vlm_sensitive_files = [normalize_path(item) for item in vlm_sensitive_files or []]
    observations = load_observations(observations_file) if observations_file else []
    warnings: list[str] = []
    errors: list[str] = []

    if not config.enabled:
        observations.extend(_log_anchored_observations(logs, sensitive_files, start_index=len(observations)))

    vision_stats = _empty_vision_stats(config.enabled, log_mining)
    if config.enabled:
        vision_observations, vision_stats, vision_warnings, vision_errors = _run_vision_pipeline(
            video_path=video_path,
            logs=logs,
            sensitive_files=sensitive_files,
            vlm_sensitive_files=vlm_sensitive_files,
            config=config,
            start_index=len(observations),
            artifact_dir=artifact_dir,
            analysis_windows=analysis_windows,
            log_mining=log_mining,
            vision_precompute_file=vision_precompute_file,
            debug_artifacts=debug_artifacts,
        )
        observations.extend(vision_observations)
        warnings.extend(vision_warnings)
        errors.extend(vision_errors)

    return {
        "video_file": str(video_path or ""),
        "observations": [item.to_dict() for item in observations],
        "statistics": {
            "mode": "direct_keyframe_vlm" if config.enabled else "deterministic_log_anchored",
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


def _empty_vision_stats(enabled: bool, log_mining: dict[str, Any] | None) -> dict[str, Any]:
    return {
        "enabled": enabled,
        "analysis_windows": 0,
        "keyframes": 0,
        "vlm_frames": 0,
        "vlm_events": 0,
        "window_source": str((log_mining or {}).get("source") or "in_memory"),
        "log_mining": dict(log_mining or {}),
    }


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
                start_ms=max(event.video_time_ms - 2000, 0) if event.video_time_ms >= 0 else 0,
                end_ms=event.video_time_ms + 2000 if event.video_time_ms >= 0 else 0,
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
    vlm_sensitive_files: list[str],
    config: VisionConfig,
    start_index: int,
    artifact_dir: str | Path | None,
    analysis_windows: list[AnalysisWindow] | None,
    log_mining: dict[str, Any] | None,
    vision_precompute_file: str | Path | None,
    debug_artifacts: bool,
) -> tuple[list[FrameObservation], dict[str, Any], list[str], list[str]]:
    started = time.perf_counter()
    warnings: list[str] = []
    errors: list[str] = []

    windows, selection, windows_seconds, keyframes_seconds, cache_reused = _load_or_select_keyframes(
        video_path=video_path,
        logs=logs,
        sensitive_files=sensitive_files,
        config=config,
        analysis_windows=analysis_windows,
        vision_precompute_file=vision_precompute_file,
    )
    warnings.extend(selection.warnings)

    send_vlm_frames = config.max_vlm_frames != 0
    selected_frames = choose_keyframes_for_vlm(selection.keyframes, max_frames=config.max_vlm_frames) if send_vlm_frames else []
    manifest = export_vision_artifacts(
        artifact_dir=artifact_dir,
        keyframes=selection.keyframes,
        raw_all_keyframes=selection.raw_keyframes,
        duplicate_keyframes=selection.duplicates,
    )
    if not cache_reused:
        write_vision_precompute(manifest, windows=windows, selection=selection)

    request_frames = prepare_vlm_request_frames(
        selected_frames,
        max_image_side=config.vlm_max_image_side,
        grid_size=config.vlm_grid_size,
        grid_layout=config.vlm_grid_layout,
        artifact_dir=artifact_dir,
        manifest=manifest,
    )
    vlm_result = _run_vlm_if_needed(
        request_frames,
        windows=windows,
        sensitive_files=vlm_sensitive_files,
        config=config,
        artifact_dir=artifact_dir,
        manifest=manifest,
        debug_artifacts=debug_artifacts,
    )

    warnings.extend(vlm_result["warnings"])
    errors.extend(vlm_result["errors"])
    observations = vision_events_to_observations(vlm_result["events"], source="vlm", start_index=start_index)
    stats = _vision_stats(
        config=config,
        windows=windows,
        selection=selection,
        selected_frame_count=len(selected_frames),
        request_frame_count=len(request_frames),
        send_vlm_frames=send_vlm_frames,
        cache_reused=cache_reused,
        vision_precompute_file=vision_precompute_file,
        manifest=manifest,
        log_mining=log_mining,
        analysis_windows=analysis_windows,
        debug_artifacts=debug_artifacts,
        windows_seconds=windows_seconds,
        keyframes_seconds=keyframes_seconds,
        vlm_result=vlm_result,
        total_seconds=time.perf_counter() - started,
    )
    return observations, stats, warnings, errors


def _load_or_select_keyframes(
    *,
    video_path: str | Path,
    logs: list[LogEvent],
    sensitive_files: list[str],
    config: VisionConfig,
    analysis_windows: list[AnalysisWindow] | None,
    vision_precompute_file: str | Path | None,
) -> tuple[list[AnalysisWindow], KeyFrameSelection, float, float, bool]:
    if vision_precompute_file:
        cached = load_vision_precompute(vision_precompute_file)
        return cached["windows"], cached["selection"], 0.0, 0.0, True

    windows_started = time.perf_counter()
    windows = analysis_windows if analysis_windows is not None else build_analysis_windows(logs, sensitive_files, config)
    windows = augment_with_video_coverage(video_path, windows, config)
    windows_seconds = time.perf_counter() - windows_started

    keyframes_started = time.perf_counter()
    selection = select_keyframes_detailed(video_path, windows, config)
    keyframes_seconds = time.perf_counter() - keyframes_started
    return windows, selection, windows_seconds, keyframes_seconds, False


def _run_vlm_if_needed(
    request_frames: list[Any],
    *,
    windows: list[AnalysisWindow],
    sensitive_files: list[str],
    config: VisionConfig,
    artifact_dir: str | Path | None,
    manifest: dict[str, Any],
    debug_artifacts: bool,
) -> dict[str, Any]:
    empty = {
        "events": [],
        "errors": [],
        "warnings": [],
        "seconds": 0.0,
        "parse_errors": 0,
        "request_metrics": {},
        "usage": {},
        "dispatch": {},
        "api_key_count": len(build_vlm_clients(config)),
        "parallelism": effective_vlm_parallelism(config),
        "batch_count": 0,
    }
    if not request_frames:
        return empty

    started = time.perf_counter()
    try:
        active_apps = sorted({app for window in windows for app in window.active_apps})
        clients = build_vlm_clients(config)
        parallelism = effective_vlm_parallelism(config)
        batches = vlm_frame_batches(request_frames, parallelism)
        summaries = [
            vlm_batch_request_summary(
                clients[0],
                batch,
                batch_index=index,
                batch_count=len(batches),
                workers=parallelism,
                sensitive_files=sensitive_files,
                active_apps=active_apps,
            )
            for index, batch in enumerate(batches)
        ]
        if debug_artifacts:
            write_json_artifact(
                artifact_dir,
                "vlm_request.json",
                vlm_request_artifact_payload(
                    summaries,
                    workers=parallelism,
                    workers_per_key=config.vlm_workers,
                    fast_dispatch=config.vlm_fast_dispatch,
                    api_key_count=len(clients),
                ),
                manifest,
                "vlm_request_file",
            )
        results = run_vlm_batches(
            clients,
            batches,
            sensitive_files=sensitive_files,
            active_apps=active_apps,
            workers_per_key=config.vlm_workers,
            retry_attempts=config.vlm_retry_attempts,
            retry_backoff_seconds=config.vlm_retry_backoff_seconds,
        )
        if debug_artifacts:
            write_json_artifact(artifact_dir, "vlm_response.json", vlm_response_artifact_payload(results), manifest, "vlm_response_file")
            write_json_artifact(artifact_dir, "vlm_parse_result.json", vlm_parse_artifact_payload(results), manifest, "vlm_parse_result_file")
        return {
            "events": list(results.get("events") or []),
            "errors": [str(item) for item in results.get("errors", [])],
            "warnings": [str(item) for item in results.get("retry_warnings", [])],
            "seconds": time.perf_counter() - started,
            "parse_errors": len(results.get("parse_errors") or []),
            "request_metrics": combine_vlm_request_metrics(summaries),
            "usage": dict(results.get("usage") or {}),
            "dispatch": dict(results.get("dispatch") or {}),
            "api_key_count": len(clients),
            "parallelism": parallelism,
            "batch_count": len(batches),
        }
    except Exception as exc:
        return {
            **empty,
            "seconds": time.perf_counter() - started,
            "errors": [f"vlm_failed: {type(exc).__name__}: {exc}"],
        }


def _vision_stats(
    *,
    config: VisionConfig,
    windows: list[AnalysisWindow],
    selection: KeyFrameSelection,
    selected_frame_count: int,
    request_frame_count: int,
    send_vlm_frames: bool,
    cache_reused: bool,
    vision_precompute_file: str | Path | None,
    manifest: dict[str, Any],
    log_mining: dict[str, Any] | None,
    analysis_windows: list[AnalysisWindow] | None,
    debug_artifacts: bool,
    windows_seconds: float,
    keyframes_seconds: float,
    vlm_result: dict[str, Any],
    total_seconds: float,
) -> dict[str, Any]:
    return {
        "enabled": config.enabled,
        "analysis_windows": len(windows),
        "window_source": str((log_mining or {}).get("source") or ("provided" if analysis_windows is not None else "in_memory")),
        "log_mining": dict(log_mining or {}),
        "keyframes": len(selection.keyframes),
        "keyframes_raw_all": len(selection.raw_keyframes),
        "keyframe_duplicates": len(selection.duplicates),
        "vlm_frames": request_frame_count,
        "vlm_source_frames": selected_frame_count,
        "vlm_events": len(vlm_result["events"]),
        "vlm_dry_run": config.vlm_dry_run,
        "vlm_frame_source": "direct_keyframes",
        "vlm_grid_size": config.vlm_grid_size,
        "vlm_grid_layout": config.vlm_grid_layout,
        "vlm_workers": config.vlm_workers,
        "vlm_fast_dispatch": config.vlm_fast_dispatch,
        "vlm_api_key_count": vlm_result["api_key_count"],
        "vlm_parallelism": vlm_result["parallelism"],
        "vlm_max_image_side": config.vlm_max_image_side,
        "vlm_batches": vlm_result["batch_count"],
        "vlm_enabled_for_run": send_vlm_frames,
        "vision_debug_artifacts": debug_artifacts,
        "vlm_parse_errors": vlm_result["parse_errors"],
        "vlm_request_metrics": vlm_result["request_metrics"],
        "vlm_usage": vlm_result["usage"],
        "vlm_dispatch": vlm_result["dispatch"],
        "vision_precompute_reused": cache_reused,
        "vision_precompute_file": str(vision_precompute_file or manifest.get("vision_precompute_file", "")),
        "timing_seconds": {
            "windows": round(windows_seconds, 3),
            "keyframes": round(keyframes_seconds, 3),
            "vlm": round(vlm_result["seconds"], 3),
            "total": round(total_seconds, 3),
        },
        "artifacts": manifest,
    }
