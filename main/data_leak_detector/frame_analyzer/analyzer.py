"""FrameAnalyzer with deterministic, OCR, and VLM-assisted evidence paths."""

from __future__ import annotations

import json
import re
import shutil
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from ..io import basename, flatten_text, looks_sensitive, normalize_path
from ..models import FrameObservation, LogEvent
from ..policy import SINK_TOKENS, TRANSFER_TOKENS, contains_any
from ..log_mining import build_analysis_windows
from .apps import identify_frontend_app
from .config import VisionConfig
from .frames import AnalysisWindow, KeyFrameDuplicate, select_keyframes_detailed
from .ocr import OcrResult, run_ocr
from .parser import parse_vlm_response_detailed, vision_events_to_observations
from .roi import OcrFrameCandidate, prepare_ocr_candidates
from .vlm import OpenAICompatibleVlmClient, build_vlm_frame_grids, choose_keyframes_for_vlm, choose_vlm_frames, prepare_vlm_frame_images


_VLM_ENDPOINT_LOCK_GUARD = threading.Lock()
_VLM_ENDPOINT_LOCKS: dict[tuple[str, str, int], threading.BoundedSemaphore] = {}


def analyze_video_behavior(
    video_path: str | Path = "",
    *,
    logs: list[LogEvent] | None = None,
    sensitive_files: list[str] | None = None,
    observations_file: str | Path | None = None,
    vision_enabled: bool | None = None,
    vision_mode: str | None = None,
    max_vlm_frames: int | None = None,
    artifact_dir: str | Path | None = None,
    analysis_windows: list[AnalysisWindow] | None = None,
    log_mining: dict[str, Any] | None = None,
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
        "window_source": str((log_mining or {}).get("source") or "in_memory"),
        "log_mining": dict(log_mining or {}),
    }

    if config.enabled:
        vision_observations, vision_stats, vision_warnings, vision_errors = _run_vision_pipeline(
            video_path=video_path,
            logs=logs,
            sensitive_files=sensitive_files,
            config=config,
            start_index=len(observations),
            artifact_dir=artifact_dir,
            analysis_windows=analysis_windows,
            log_mining=log_mining,
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
    config: VisionConfig,
    start_index: int,
    artifact_dir: str | Path | None,
    analysis_windows: list[AnalysisWindow] | None,
    log_mining: dict[str, Any] | None,
) -> tuple[list[FrameObservation], dict[str, Any], list[str], list[str]]:
    observations: list[FrameObservation] = []
    warnings: list[str] = []
    errors: list[str] = []

    vision_started = time.perf_counter()
    windows_started = time.perf_counter()
    windows = analysis_windows if analysis_windows is not None else build_analysis_windows(logs, sensitive_files, config)
    windows_seconds = time.perf_counter() - windows_started

    keyframes_started = time.perf_counter()
    keyframe_selection = select_keyframes_detailed(video_path, windows, config)
    keyframes_seconds = time.perf_counter() - keyframes_started
    keyframes = keyframe_selection.keyframes
    warnings.extend(keyframe_selection.warnings)

    direct_keyframes_to_vlm = _vlm_uses_direct_keyframes(config)
    if direct_keyframes_to_vlm:
        ocr_frames = []
        ocr_candidates = []
        ocr_candidate_frames = []
        raw_ocr_results = []
        ocr_results = []
        ocr_prepare_seconds = 0.0
        ocr_seconds = 0.0
        ocr_postprocess_seconds = 0.0
        # direct_keyframes means VLM sees the selected keyframes themselves.
        # Keep only the explicit global VLM budget; do not apply OCR/window triage caps.
        vlm_frames = choose_keyframes_for_vlm(
            keyframes,
            max_frames=config.max_vlm_frames,
        )
    else:
        ocr_prepare_started = time.perf_counter()
        ocr_frames = _select_ocr_frames_for_ocr(keyframes, config)
        if config.ocr_roi_enabled:
            ocr_candidates = prepare_ocr_candidates(ocr_frames, config)
            ocr_candidate_frames = [candidate.frame for candidate in ocr_candidates if candidate.selected_for_ocr]
        else:
            ocr_candidates = []
            ocr_candidate_frames = ocr_frames
        ocr_prepare_seconds = time.perf_counter() - ocr_prepare_started

        ocr_started = time.perf_counter()
        raw_roi_ocr_results = run_ocr(ocr_candidate_frames, config) if ocr_candidate_frames else []
        ocr_seconds = time.perf_counter() - ocr_started

        ocr_postprocess_started = time.perf_counter()
        raw_ocr_results = _merge_roi_ocr_results(raw_roi_ocr_results, ocr_candidates) if config.ocr_roi_enabled else raw_roi_ocr_results
        ocr_results = _dedupe_ocr_results(raw_ocr_results, config)
        observations.extend(_ocr_observations(ocr_results, config, sensitive_files=sensitive_files, start_index=start_index))
        ocr_postprocess_seconds = time.perf_counter() - ocr_postprocess_started

        vlm_frames = choose_vlm_frames(
            ocr_results,
            min_confidence=config.ocr_min_confidence,
            max_frames=config.max_vlm_frames,
            strategy=config.vlm_frame_strategy,
            include_empty_ocr_strong_frames=config.vlm_include_empty_ocr_strong_frames,
            max_frames_per_window=_vlm_frames_per_window_limit(config),
        )
    artifact_manifest = _export_vision_artifacts(
        artifact_dir=artifact_dir,
        keyframes=keyframes,
        raw_all_keyframes=keyframe_selection.raw_keyframes,
        duplicate_keyframes=keyframe_selection.duplicates,
        ocr_candidates=ocr_candidates,
        ocr_selected_frames=[item.frame for item in vlm_frames],
        ocr_results=ocr_results,
    )
    vlm_request_frames = _prepare_vlm_request_frames(
        vlm_frames,
        config=config,
        artifact_dir=artifact_dir,
        manifest=artifact_manifest,
    )
    vlm_events = 0
    vlm_seconds = 0.0
    vlm_parse_errors = 0
    vlm_request_metrics: dict[str, Any] = {}
    vlm_usage: dict[str, Any] = {}
    if vlm_request_frames and config.mode.lower() in {"hybrid", "vlm"}:
        try:
            vlm_started = time.perf_counter()
            active_apps = sorted({app for window in windows for app in window.active_apps})
            vlm_clients = _build_vlm_clients(config)
            vlm_parallelism = _effective_vlm_parallelism(config, key_count=len(vlm_clients))
            vlm_batches = _vlm_frame_batches(vlm_request_frames, vlm_parallelism)
            request_summaries = [
                _vlm_batch_request_summary(
                    vlm_clients[0],
                    batch,
                    batch_index=index,
                    batch_count=len(vlm_batches),
                    workers=vlm_parallelism,
                    sensitive_files=sensitive_files,
                    active_apps=active_apps,
                )
                for index, batch in enumerate(vlm_batches)
            ]
            vlm_request_metrics = _combine_vlm_request_metrics(request_summaries)
            _write_vlm_request_artifact(
                artifact_dir,
                _vlm_request_artifact_payload(
                    request_summaries,
                    workers=vlm_parallelism,
                    workers_per_key=config.vlm_workers,
                    fast_dispatch=config.vlm_fast_dispatch,
                    api_key_count=len(vlm_clients),
                ),
                artifact_manifest,
            )
            vlm_results = _run_vlm_batches(
                vlm_clients,
                vlm_batches,
                sensitive_files=sensitive_files,
                active_apps=active_apps,
                workers_per_key=config.vlm_workers,
            )
            errors.extend(str(item) for item in vlm_results.get("errors", []))
            warnings.extend(str(item) for item in vlm_results.get("retry_warnings", []))
            vlm_usage = dict(vlm_results.get("usage") or {})
            _write_vlm_response_payload_artifact(
                artifact_dir,
                _vlm_response_artifact_payload(vlm_results),
                artifact_manifest,
            )
            _write_vlm_parse_artifact(
                artifact_dir,
                _vlm_parse_artifact_payload(vlm_results),
                artifact_manifest,
            )
            events = list(vlm_results.get("events") or [])
            vlm_events = len(events)
            vlm_parse_errors = len(vlm_results.get("parse_errors") or [])
            observations.extend(
                vision_events_to_observations(events, source="vlm", start_index=start_index + len(observations))
            )
            vlm_seconds = time.perf_counter() - vlm_started
        except Exception as exc:
            vlm_seconds = time.perf_counter() - vlm_started if "vlm_started" in locals() else 0.0
            errors.append(f"vlm_failed: {type(exc).__name__}: {exc}")

    stats = {
        "enabled": config.enabled,
        "mode": config.mode,
        "analysis_windows": len(windows),
        "window_source": str((log_mining or {}).get("source") or ("provided" if analysis_windows is not None else "in_memory")),
        "log_mining": dict(log_mining or {}),
        "keyframes": len(keyframes),
        "keyframes_raw_all": len(keyframe_selection.raw_keyframes),
        "keyframe_duplicates": len(keyframe_selection.duplicates),
        "ocr_input_keyframes": len(ocr_frames),
        "ocr_roi_candidates": len(ocr_candidates),
        "ocr_roi_selected": len(ocr_candidate_frames) if config.ocr_roi_enabled else 0,
        "ocr_frames": len(ocr_results),
        "ocr_raw_frames": len(raw_ocr_results),
        "vlm_frames": len(vlm_request_frames),
        "vlm_source_frames": len(vlm_frames),
        "vlm_events": vlm_events,
        "vlm_dry_run": config.vlm_dry_run,
        "vlm_frame_strategy": config.vlm_frame_strategy,
        "vlm_grid_size": config.vlm_grid_size,
        "vlm_workers": config.vlm_workers,
        "vlm_fast_dispatch": config.vlm_fast_dispatch,
        "vlm_api_key_count": len(_build_vlm_clients(config)),
        "vlm_parallelism": _effective_vlm_parallelism(config),
        "vlm_max_image_side": config.vlm_max_image_side,
        "vlm_batches": len(_vlm_frame_batches(vlm_request_frames, _effective_vlm_parallelism(config))) if vlm_request_frames else 0,
        "ocr_skipped_for_direct_vlm": direct_keyframes_to_vlm,
        "vlm_parse_errors": vlm_parse_errors,
        "vlm_request_metrics": vlm_request_metrics,
        "vlm_usage": vlm_usage,
        "timing_seconds": {
            "windows": round(windows_seconds, 3),
            "keyframes": round(keyframes_seconds, 3),
            "ocr_prepare": round(ocr_prepare_seconds, 3),
            "ocr": round(ocr_seconds, 3),
            "ocr_postprocess": round(ocr_postprocess_seconds, 3),
            "vlm": round(vlm_seconds, 3),
            "total": round(time.perf_counter() - vision_started, 3),
        },
        "artifacts": artifact_manifest,
    }
    return observations, stats, warnings, errors


def _vlm_frame_batches(frames: list[Any], workers: int) -> list[list[Any]]:
    if not frames:
        return []
    batch_count = min(max(1, workers), len(frames))
    batch_size = (len(frames) + batch_count - 1) // batch_count
    return [frames[index : index + batch_size] for index in range(0, len(frames), batch_size)]


def _build_vlm_clients(config: VisionConfig) -> list[OpenAICompatibleVlmClient]:
    endpoints = config.effective_vlm_endpoints()
    if not endpoints:
        return [OpenAICompatibleVlmClient(config)]
    if not config.vlm_fast_dispatch:
        endpoints = endpoints[:1]
    return [
        OpenAICompatibleVlmClient(
            replace(config, vlm_base_url=endpoint.base_url, vlm_chat_url=endpoint.chat_url, vlm_api_key=endpoint.api_key, vlm_api_keys=())
        )
        for endpoint in endpoints
    ]


def _effective_vlm_parallelism(config: VisionConfig, *, key_count: int | None = None) -> int:
    if not config.vlm_fast_dispatch:
        return config.vlm_workers
    count = len(config.effective_vlm_endpoints()) if key_count is None else key_count
    return config.vlm_workers * max(1, count)


def _vlm_batch_request_summary(
    client: OpenAICompatibleVlmClient,
    frames: list[Any],
    *,
    batch_index: int,
    batch_count: int,
    workers: int,
    sensitive_files: list[str],
    active_apps: list[str],
) -> dict[str, Any]:
    summary = client.request_summary(frames, sensitive_files=sensitive_files, active_apps=active_apps)
    if batch_count > 1:
        summary["batch_index"] = batch_index
        summary["batch_count"] = batch_count
        summary["workers"] = workers
    return summary


def _vlm_request_artifact_payload(
    request_summaries: list[dict[str, Any]],
    *,
    workers: int,
    workers_per_key: int | None = None,
    fast_dispatch: bool = False,
    api_key_count: int = 1,
) -> dict[str, Any]:
    dispatch = {
        "fast_dispatch": fast_dispatch,
        "api_key_count": api_key_count,
        "workers_per_key": workers if workers_per_key is None else workers_per_key,
        "parallelism": workers,
    }
    if len(request_summaries) == 1:
        payload = dict(request_summaries[0])
        payload["dispatch"] = dispatch
        return payload
    first = request_summaries[0] if request_summaries else {}
    return {
        "provider": first.get("provider", ""),
        "model": first.get("model", ""),
        "chat_url": first.get("chat_url", ""),
        "dry_run": first.get("dry_run", False),
        "frame_strategy": first.get("frame_strategy", ""),
        "grid_size": first.get("grid_size", 1),
        "workers": workers,
        "dispatch": dispatch,
        "batch_count": len(request_summaries),
        "request_metrics": _combine_vlm_request_metrics(request_summaries),
        "batches": request_summaries,
    }


def _combine_vlm_request_metrics(request_summaries: list[dict[str, Any]]) -> dict[str, Any]:
    metrics = [item.get("request_metrics") for item in request_summaries if isinstance(item.get("request_metrics"), dict)]
    if not metrics:
        return {}
    if len(metrics) == 1:
        return dict(metrics[0])
    combined: dict[str, Any] = {"batches": [dict(item) for item in metrics]}
    for item in metrics:
        for key, value in item.items():
            if key == "image_sizes" and isinstance(value, list):
                combined.setdefault(key, []).extend(value)
            elif isinstance(value, int | float):
                combined[key] = combined.get(key, 0) + value
    if "image_megapixels" in combined:
        combined["image_megapixels"] = round(float(combined["image_megapixels"]), 3)
    return combined


def _run_vlm_batches(
    clients: list[OpenAICompatibleVlmClient],
    batches: list[list[Any]],
    *,
    sensitive_files: list[str],
    active_apps: list[str],
    workers_per_key: int,
) -> dict[str, Any]:
    if not clients:
        return {"batches": [], "errors": ["vlm_client_pool_empty"], "events": [], "parse_errors": [], "usage": {}}

    client_locks = _shared_vlm_endpoint_locks(clients, workers_per_key=workers_per_key)

    def run_one(batch_index: int, frames: list[Any]) -> dict[str, Any]:
        start = batch_index % len(clients)
        ordered_clients = clients[start:] + clients[:start]
        retry_warnings: list[str] = []
        response = None
        for attempt, client in enumerate(ordered_clients):
            lock = client_locks[id(client)]
            try:
                with lock:
                    response = client.analyze(frames, sensitive_files=sensitive_files, active_apps=active_apps)
                break
            except Exception as exc:
                if attempt + 1 == len(ordered_clients):
                    raise
                retry_warnings.append(f"vlm_key_retry[{batch_index}]: {type(exc).__name__}: {exc}")
        if response is None:
            raise RuntimeError("vlm_response_unavailable")
        parse_result = parse_vlm_response_detailed(response.text, keywords=sensitive_files)
        return {
            "batch_index": batch_index,
            "frame_count": len(frames),
            "response": response,
            "parse_result": parse_result,
            "retry_warnings": retry_warnings,
        }

    results: list[dict[str, Any]] = []
    errors: list[str] = []
    max_workers = min(max(1, len(clients) * max(1, workers_per_key)), len(batches) or 1)
    if max_workers <= 1:
        for index, batch in enumerate(batches):
            try:
                results.append(run_one(index, batch))
            except Exception as exc:
                errors.append(f"vlm_batch_failed[{index}]: {type(exc).__name__}: {exc}")
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_index = {executor.submit(run_one, index, batch): index for index, batch in enumerate(batches)}
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                try:
                    results.append(future.result())
                except Exception as exc:
                    errors.append(f"vlm_batch_failed[{index}]: {type(exc).__name__}: {exc}")

    results.sort(key=lambda item: int(item.get("batch_index", 0)))
    events: list[Any] = []
    parse_errors: list[str] = []
    usages: list[dict[str, Any]] = []
    retry_warnings: list[str] = []
    for result in results:
        parse_result = result["parse_result"]
        events.extend(parse_result.events)
        parse_errors.extend(parse_result.parse_errors)
        retry_warnings.extend(str(item) for item in result.get("retry_warnings", []))
        usage = result["response"].usage
        if isinstance(usage, dict):
            usages.append(usage)
    return {
        "batches": results,
        "errors": errors,
        "events": events,
        "parse_errors": parse_errors,
        "retry_warnings": retry_warnings,
        "usage": _combine_vlm_usage(usages),
    }


def _shared_vlm_endpoint_locks(
    clients: list[OpenAICompatibleVlmClient],
    *,
    workers_per_key: int,
) -> dict[int, threading.BoundedSemaphore]:
    """Share plan quotas across concurrently analyzed cases in this process."""

    limit = max(1, workers_per_key)
    locks: dict[int, threading.BoundedSemaphore] = {}
    with _VLM_ENDPOINT_LOCK_GUARD:
        for client in clients:
            identity = (client.config.vlm_base_url.rstrip("/"), client.config.vlm_api_key, limit)
            lock = _VLM_ENDPOINT_LOCKS.get(identity)
            if lock is None:
                lock = threading.BoundedSemaphore(limit)
                _VLM_ENDPOINT_LOCKS[identity] = lock
            locks[id(client)] = lock
    return locks


def _combine_vlm_usage(usages: list[dict[str, Any]]) -> dict[str, Any]:
    if not usages:
        return {}
    if len(usages) == 1:
        return dict(usages[0])
    combined: dict[str, Any] = {"batches": [dict(item) for item in usages]}
    for usage in usages:
        for key, value in usage.items():
            if isinstance(value, int | float):
                combined[key] = combined.get(key, 0) + value
    return combined


def _vlm_response_artifact_payload(vlm_results: dict[str, Any]) -> dict[str, Any]:
    batch_results = list(vlm_results.get("batches") or [])
    errors = list(vlm_results.get("errors") or [])
    if len(batch_results) == 1 and not errors:
        return _vlm_response_to_dict(batch_results[0]["response"])
    first_response = batch_results[0]["response"] if batch_results else None
    return {
        "provider": getattr(first_response, "provider", ""),
        "model": getattr(first_response, "model", ""),
        "dry_run": bool(getattr(first_response, "dry_run", False)) if first_response is not None else False,
        "usage": vlm_results.get("usage") or {},
        "errors": errors,
        "batch_count": len(batch_results) + len(errors),
        "responses": [
            {
                "batch_index": result.get("batch_index"),
                "frame_count": result.get("frame_count"),
                **_vlm_response_to_dict(result["response"]),
            }
            for result in batch_results
        ],
    }


def _vlm_response_to_dict(response: Any) -> dict[str, Any]:
    return {
        "provider": response.provider,
        "model": response.model,
        "dry_run": response.dry_run,
        "usage": response.usage,
        "text": response.text,
        "raw_payload": response.raw_payload,
    }


def _vlm_parse_artifact_payload(vlm_results: dict[str, Any]) -> dict[str, Any]:
    batch_results = list(vlm_results.get("batches") or [])
    errors = list(vlm_results.get("errors") or [])
    if len(batch_results) == 1 and not errors:
        return batch_results[0]["parse_result"].to_dict()

    events: list[dict[str, Any]] = []
    raw_events: list[dict[str, Any]] = []
    dropped_events: list[dict[str, Any]] = []
    parse_errors: list[str] = []
    batches: list[dict[str, Any]] = []
    for result in batch_results:
        parse_payload = result["parse_result"].to_dict()
        events.extend(parse_payload.get("events", []))
        raw_events.extend(parse_payload.get("raw_events", []))
        dropped_events.extend(parse_payload.get("dropped_events", []))
        parse_errors.extend(parse_payload.get("parse_errors", []))
        batches.append(
            {
                "batch_index": result.get("batch_index"),
                "frame_count": result.get("frame_count"),
                "parse_result": parse_payload,
            }
        )
    return {
        "events": events,
        "raw_events": raw_events,
        "dropped_events": dropped_events,
        "parse_errors": parse_errors,
        "errors": errors,
        "batches": batches,
    }


def _write_vlm_request_artifact(
    artifact_dir: str | Path | None,
    request_summary: dict[str, Any],
    manifest: dict[str, Any],
) -> None:
    if artifact_dir is None:
        return
    path = Path(artifact_dir) / "vlm_request.json"
    path.write_text(json.dumps(request_summary, ensure_ascii=False, indent=2), encoding="utf-8")
    manifest["vlm_request_file"] = str(path)


def _write_vlm_response_artifact(
    artifact_dir: str | Path | None,
    response: Any,
    manifest: dict[str, Any],
) -> None:
    if artifact_dir is None:
        return
    path = Path(artifact_dir) / "vlm_response.json"
    payload = {
        "provider": response.provider,
        "model": response.model,
        "dry_run": response.dry_run,
        "usage": response.usage,
        "text": response.text,
        "raw_payload": response.raw_payload,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    manifest["vlm_response_file"] = str(path)


def _write_vlm_response_payload_artifact(
    artifact_dir: str | Path | None,
    payload: dict[str, Any],
    manifest: dict[str, Any],
) -> None:
    if artifact_dir is None:
        return
    path = Path(artifact_dir) / "vlm_response.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    manifest["vlm_response_file"] = str(path)


def _write_vlm_parse_artifact(
    artifact_dir: str | Path | None,
    parse_payload: dict[str, Any],
    manifest: dict[str, Any],
) -> None:
    if artifact_dir is None:
        return
    path = Path(artifact_dir) / "vlm_parse_result.json"
    path.write_text(json.dumps(parse_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    manifest["vlm_parse_result_file"] = str(path)


def _prepare_vlm_request_frames(
    frames: list[Any],
    *,
    config: VisionConfig,
    artifact_dir: str | Path | None,
    manifest: dict[str, Any],
) -> list[Any]:
    input_dir = Path(artifact_dir) / "keyframes_vlm_input" if artifact_dir is not None and config.vlm_max_image_side > 0 else None
    if input_dir is not None:
        if input_dir.exists():
            shutil.rmtree(input_dir)
        input_dir.mkdir(parents=True, exist_ok=True)
    prepared_frames = prepare_vlm_frame_images(frames, max_image_side=config.vlm_max_image_side, output_dir=input_dir)
    if input_dir is not None:
        input_files = [item.frame.image_path for item in prepared_frames if Path(item.frame.image_path).parent == input_dir]
        if input_files:
            manifest["keyframes_vlm_input_dir"] = str(input_dir)
            manifest["keyframes_vlm_input_files"] = input_files
            counts = manifest.setdefault("counts", {})
            if isinstance(counts, dict):
                counts["keyframes_vlm_input_files"] = len(input_files)
            _update_artifact_manifest_file(manifest, {"keyframes_vlm_input_files": input_files})
    if config.vlm_grid_size <= 1:
        return prepared_frames
    grid_dir = Path(artifact_dir) / "keyframes_vlm_grid" if artifact_dir is not None else None
    if grid_dir is not None:
        if grid_dir.exists():
            shutil.rmtree(grid_dir)
        grid_dir.mkdir(parents=True, exist_ok=True)
    grid_frames = build_vlm_frame_grids(prepared_frames, grid_size=config.vlm_grid_size, output_dir=grid_dir)
    if grid_dir is not None:
        grid_files = [item.frame.image_path for item in grid_frames]
        manifest["keyframes_vlm_grid_dir"] = str(grid_dir)
        manifest["keyframes_vlm_grid_files"] = grid_files
        counts = manifest.setdefault("counts", {})
        if isinstance(counts, dict):
            counts["keyframes_vlm_grid_files"] = len(grid_files)
        _update_artifact_manifest_file(manifest, {"keyframes_vlm_grid_files": grid_files})
    return grid_frames


def _update_artifact_manifest_file(manifest: dict[str, Any], updates: dict[str, Any]) -> None:
    manifest_file = str(manifest.get("artifact_manifest_file") or "")
    if not manifest_file:
        return
    path = Path(manifest_file)
    if path.exists():
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            payload = {}
    else:
        payload = {}
    if not isinstance(payload, dict):
        payload = {}
    payload.update(updates)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _vlm_uses_direct_keyframes(config: VisionConfig) -> bool:
    strategy = config.vlm_frame_strategy.strip().lower().replace("-", "_")
    direct_aliases = {"direct", "direct_keyframes", "all_keyframes", "keyframes"}
    return (
        strategy in direct_aliases
        and config.mode.lower() in {"hybrid", "vlm"}
        and config.max_vlm_frames != 0
    )


def _vlm_frames_per_window_limit(config: VisionConfig) -> int | None:
    if config.vlm_max_frames_per_window <= 0:
        return None
    return config.vlm_max_frames_per_window


def _export_vision_artifacts(
    *,
    artifact_dir: str | Path | None,
    keyframes: list[Any],
    ocr_selected_frames: list[Any],
    ocr_results: list[OcrResult],
    raw_all_keyframes: list[Any] | None = None,
    duplicate_keyframes: list[KeyFrameDuplicate] | None = None,
    ocr_candidates: list[OcrFrameCandidate] | None = None,
) -> dict[str, Any]:
    if artifact_dir is None:
        return {}

    root = Path(artifact_dir)
    raw_all_dir = root / "keyframes_raw_all"
    raw_dir = root / "keyframes_raw"
    roi_dir = root / "keyframes_ocr_roi"
    selected_dir = root / "keyframes_ocr_selected"
    for directory in (raw_all_dir, raw_dir, roi_dir, selected_dir):
        if directory.exists():
            shutil.rmtree(directory)
        directory.mkdir(parents=True, exist_ok=True)

    raw_all_files = _copy_frame_images(raw_all_keyframes if raw_all_keyframes is not None else keyframes, raw_all_dir)
    raw_files = _copy_frame_images(keyframes, raw_dir)
    roi_files = _copy_frame_images([item.frame for item in ocr_candidates or [] if item.selected_for_ocr], roi_dir)
    selected_files = _copy_frame_images(ocr_selected_frames, selected_dir)
    ocr_file = root / "ocr_results.json"
    ocr_file.write_text(json.dumps([_ocr_result_to_dict(item) for item in ocr_results], ensure_ascii=False, indent=2), encoding="utf-8")
    roi_file = root / "ocr_roi_regions.json"
    roi_file.write_text(json.dumps([_ocr_candidate_to_dict(item) for item in ocr_candidates or []], ensure_ascii=False, indent=2), encoding="utf-8")
    duplicate_file = root / "keyframe_duplicates.json"
    duplicate_file.write_text(
        json.dumps([_keyframe_duplicate_to_dict(item) for item in duplicate_keyframes or []], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    manifest_file = root / "artifact_manifest.json"
    manifest_file.write_text(
        json.dumps(
            {
                "keyframes_raw_all_files": raw_all_files,
                "keyframes_raw_files": raw_files,
                "keyframes_ocr_roi_files": roi_files,
                "keyframes_ocr_selected_files": selected_files,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    return {
        "root_dir": str(root),
        "keyframes_raw_all_dir": str(raw_all_dir),
        "keyframes_raw_dir": str(raw_dir),
        "keyframes_ocr_roi_dir": str(roi_dir),
        "keyframes_ocr_selected_dir": str(selected_dir),
        "ocr_results_file": str(ocr_file),
        "ocr_roi_regions_file": str(roi_file),
        "keyframe_duplicates_file": str(duplicate_file),
        "artifact_manifest_file": str(manifest_file),
        "counts": {
            "keyframes_raw_all_files": len(raw_all_files),
            "keyframes_raw_files": len(raw_files),
            "keyframes_ocr_roi_files": len(roi_files),
            "keyframes_ocr_selected_files": len(selected_files),
        },
    }


def _copy_frame_images(frames: list[Any], target_dir: Path) -> list[str]:
    copied: list[str] = []
    for index, frame in enumerate(frames):
        source = Path(str(getattr(frame, "image_path", "")))
        if not source.exists():
            continue
        timestamp = int(getattr(frame, "timestamp_ms", 0))
        reason = str(getattr(frame, "reason", "frame")).replace(":", "-").replace("/", "-").replace("\\", "-")
        target = target_dir / f"{index:03d}_{timestamp}ms_{reason}{source.suffix or '.jpg'}"
        shutil.copy2(source, target)
        copied.append(str(target))
    return copied


def _ocr_result_to_dict(result: OcrResult) -> dict[str, Any]:
    return {
        "frame_id": result.frame.frame_id,
        "timestamp_ms": result.frame.timestamp_ms,
        "image_path": result.frame.image_path,
        "reason": result.frame.reason,
        "window_id": result.frame.window_id,
        "text": result.text,
        "confidence": result.confidence,
        "provider": result.provider,
    }


def _ocr_candidate_to_dict(candidate: OcrFrameCandidate) -> dict[str, Any]:
    return {
        "frame_id": candidate.source_frame.frame_id,
        "roi_frame_id": candidate.frame.frame_id,
        "timestamp_ms": candidate.source_frame.timestamp_ms,
        "source_image_path": candidate.source_frame.image_path,
        "roi_image_path": candidate.frame.image_path,
        "selected_for_ocr": candidate.selected_for_ocr,
        "reason": candidate.reason,
        "regions": [
            {
                "x": region.x,
                "y": region.y,
                "width": region.width,
                "height": region.height,
                "text_density": region.text_density,
                "edge_density": region.edge_density,
            }
            for region in candidate.regions
        ],
    }


def _keyframe_duplicate_to_dict(duplicate: KeyFrameDuplicate) -> dict[str, Any]:
    return {
        "frame_id": duplicate.frame.frame_id,
        "timestamp_ms": duplicate.frame.timestamp_ms,
        "image_path": duplicate.frame.image_path,
        "reason": duplicate.frame.reason,
        "window_id": duplicate.frame.window_id,
        "kept_frame_id": duplicate.kept_frame_id,
        "duplicate_reason": duplicate.reason,
        "delta": duplicate.delta,
        "hash_distance": duplicate.hash_distance,
    }


def _select_ocr_frames_for_ocr(frames: list[Any], config: VisionConfig | None = None) -> list[Any]:
    """Select OCR input frames while keeping each analysis window represented."""

    unique: list[Any] = []
    seen_timestamps: set[int] = set()
    for frame in sorted(frames, key=lambda item: int(getattr(item, "timestamp_ms", 0))):
        timestamp = int(getattr(frame, "timestamp_ms", 0))
        if timestamp in seen_timestamps:
            continue
        seen_timestamps.add(timestamp)
        unique.append(frame)
    if config is None:
        return unique

    selected: list[Any] = []
    by_window: dict[str, list[Any]] = {}
    for frame in unique:
        by_window.setdefault(str(getattr(frame, "window_id", "") or "window_unknown"), []).append(frame)

    for _, group in sorted(by_window.items(), key=lambda item: min(int(getattr(frame, "timestamp_ms", 0)) for frame in item[1])):
        priority = _frame_priority(group[0])
        if priority == "strong":
            budget = config.max_keyframes_per_strong_window
        elif priority == "weak":
            budget = config.max_keyframes_per_weak_window
        else:
            budget = config.max_keyframes_per_medium_window
        anchor_frames = [frame for frame in group if "anchor" in str(getattr(frame, "reason", ""))]
        budget = max(budget, len(anchor_frames))
        selected.extend(_representative_frames(group, budget))
    return sorted(selected, key=lambda item: int(getattr(item, "timestamp_ms", 0)))


def _representative_frames(frames: list[Any], budget: int) -> list[Any]:
    ordered = sorted(frames, key=lambda item: int(getattr(item, "timestamp_ms", 0)))
    if len(ordered) <= budget:
        return ordered
    if budget <= 1:
        return [ordered[0]]
    selected: list[Any] = []
    last_index = len(ordered) - 1
    for slot in range(budget):
        selected.append(ordered[round(slot * last_index / (budget - 1))])
    return selected


def _frame_priority(frame: Any) -> str:
    reason = str(getattr(frame, "reason", ""))
    return reason.split(":", 1)[0].split("-", 1)[0].lower()


def _merge_roi_ocr_results(results: list[OcrResult], candidates: list[OcrFrameCandidate]) -> list[OcrResult]:
    candidates_by_frame = {candidate.frame.frame_id: candidate for candidate in candidates}
    grouped: dict[str, list[OcrResult]] = {}
    source_frames: dict[str, Any] = {}
    for result in results:
        candidate = candidates_by_frame.get(result.frame.frame_id)
        source_frame = candidate.source_frame if candidate is not None else result.frame
        key = source_frame.frame_id
        source_frames[key] = source_frame
        grouped.setdefault(key, []).append(result)

    merged: list[OcrResult] = []
    for frame_id, items in grouped.items():
        texts = [item.text.strip() for item in items if item.text.strip()]
        confidences = [item.confidence for item in items if item.confidence > 0]
        providers = sorted({item.provider for item in items})
        confidence = sum(confidences) / len(confidences) if confidences else 0.0
        provider = "+".join(providers) + ":roi" if providers else "roi"
        merged.append(OcrResult(frame=source_frames[frame_id], text=" ".join(texts), confidence=round(confidence, 3), provider=provider))
    return sorted(merged, key=lambda item: item.frame.timestamp_ms)


def _dedupe_ocr_results(results: list[OcrResult], config: VisionConfig) -> list[OcrResult]:
    kept: list[OcrResult] = []
    texts_by_window: dict[str, list[str]] = {}
    for result in results:
        text = " ".join(result.text.split()).lower()
        if "anchor" in str(getattr(result.frame, "reason", "")):
            kept.append(result)
            if text:
                window_id = result.frame.window_id or "window_unknown"
                texts_by_window.setdefault(window_id, []).append(text)
            continue
        if not text:
            kept.append(result)
            continue
        window_id = result.frame.window_id or "window_unknown"
        previous = texts_by_window.setdefault(window_id, [])
        if any(_text_similarity(text, item) >= config.ocr_text_similarity_threshold for item in previous):
            continue
        previous.append(text)
        kept.append(result)
    return kept


def _text_similarity(left: str, right: str) -> float:
    return SequenceMatcher(None, left, right).ratio()


OCR_FILE_RE = re.compile(
    r"[\w\u4e00-\u9fff ._\-()（）【】\[\]]+"
    r"\.(?:docx|doc|pdf|txt|png|jpg|jpeg|xlsx|xls|csv|sql|zip|7z|rar|pptx|ppt|mp4|mov)",
    re.IGNORECASE,
)


def _ocr_mentioned_files(text: str, sensitive_files: list[str]) -> list[str]:
    mentioned: list[str] = []
    for match in OCR_FILE_RE.finditer(text):
        candidate = _clean_ocr_file_candidate(match.group(0))
        if candidate:
            mentioned.append(candidate)

    normalized_text = normalize_path(text).lower()
    for sensitive in sensitive_files:
        resolved = _resolve_sensitive_from_text(normalized_text, sensitive)
        if resolved:
            mentioned.append(resolved)
    return list(dict.fromkeys(mentioned))


def _clean_ocr_file_candidate(value: str) -> str:
    candidate = normalize_path(value).strip(" .,:;，。：；|[]【】()（）")
    if "/" in candidate:
        return candidate
    parts = [part.strip(" .,:;，。：；|[]【】()（）") for part in candidate.split() if part.strip()]
    if parts and OCR_FILE_RE.fullmatch(parts[-1]):
        return parts[-1]
    return candidate


def _resolve_sensitive_from_text(normalized_text: str, sensitive_file: str) -> str:
    sensitive = normalize_path(sensitive_file)
    if not sensitive:
        return ""
    lowered = sensitive.lower()
    name = basename(sensitive).lower()
    stem = Path(name).stem.lower()
    if lowered in normalized_text or (name and name in normalized_text) or (len(stem) >= 4 and stem in normalized_text):
        return sensitive
    return ""


def _ocr_fact_prefix(text: str, mentioned_files: list[str]) -> str:
    facts: list[str] = []
    if mentioned_files:
        facts.append("mentioned_files=" + "|".join(mentioned_files[:8]))
    if contains_any(text, SINK_TOKENS):
        facts.append("sink_context=true")
    if contains_any(text, TRANSFER_TOKENS):
        facts.append("transfer_context=true")
    return f"OCR facts: {'; '.join(facts)}. " if facts else ""


def _ocr_observations(
    results: list[OcrResult],
    config: VisionConfig,
    *,
    sensitive_files: list[str] | None = None,
    start_index: int,
) -> list[FrameObservation]:
    observations: list[FrameObservation] = []
    sensitive_files = [normalize_path(item) for item in sensitive_files or []]
    for result in results:
        if result.confidence < config.ocr_min_confidence or not result.text.strip():
            continue
        app = identify_frontend_app(ocr_text=result.text)
        mentioned_files = _ocr_mentioned_files(result.text, sensitive_files)
        sink_context = contains_any(result.text, SINK_TOKENS)
        transfer_context = contains_any(result.text, TRANSFER_TOKENS)
        if sink_context:
            operation = "external_sink_interaction"
        elif transfer_context:
            operation = "file_or_content_transfer"
        else:
            operation = "visual_text_observed"
        resolved_files = list(
            dict.fromkeys(
                resolved
                for item in mentioned_files
                for sensitive in sensitive_files
                for resolved in (_resolve_sensitive_from_text(item.lower(), sensitive),)
                if resolved
            )
        )
        resource = resolved_files[0] if resolved_files else (mentioned_files[0] if mentioned_files else "")
        observations.append(
            FrameObservation(
                observation_id=f"ocr_{start_index + len(observations)}",
                start_ms=result.frame.timestamp_ms,
                end_ms=result.frame.timestamp_ms,
                app_name=app.app_name,
                operation_type=operation,
                resource=resource,
                related_resources=tuple(mentioned_files),
                description=f"{_ocr_fact_prefix(result.text, mentioned_files)}OCR text: {result.text[:500]}; app_category={app.category}; risk_hint={app.risk_hint}",
                confidence=result.confidence,
                source=result.provider,
            )
        )
    return observations
