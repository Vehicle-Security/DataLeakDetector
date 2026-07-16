"""Batch dispatch, retry, and response aggregation for VLM calls."""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path
from typing import Any

from .config import VisionConfig
from .parser import VlmParseResult, parse_vlm_response_detailed
from .vlm_client import OpenAICompatibleVlmClient


_VLM_ENDPOINT_LOCK_GUARD = threading.Lock()
_VLM_ENDPOINT_LOCKS: dict[tuple[str, str, int], threading.BoundedSemaphore] = {}
_VLM_DISPATCHER_GUARD = threading.Lock()
_VLM_DISPATCHERS: dict[tuple[tuple[tuple[str, str], ...], int], Any] = {}


def build_vlm_clients(config: VisionConfig) -> list[OpenAICompatibleVlmClient]:
    endpoints = config.effective_vlm_endpoints()
    if not endpoints:
        return [OpenAICompatibleVlmClient(config)]
    endpoints = endpoints[:1]
    return [
        OpenAICompatibleVlmClient(
            replace(config, vlm_base_url=endpoint.base_url, vlm_chat_url=endpoint.chat_url, vlm_api_key=endpoint.api_key, vlm_api_keys=())
        )
        for endpoint in endpoints
    ]


def effective_vlm_parallelism(config: VisionConfig) -> int:
    return config.vlm_workers


def vlm_frame_batches(frames: list[Any], workers: int) -> list[list[Any]]:
    if not frames:
        return []
    # A window is one chronological evidence packet. Worker count controls how
    # many packets run concurrently; it must never split a packet and discard
    # the relationship between an identity frame and its later result frame.
    del workers
    by_window: dict[str, list[Any]] = {}
    for item in frames:
        frame = getattr(item, "frame", item)
        window_id = str(getattr(frame, "window_id", "") or "window_unknown")
        by_window.setdefault(window_id, []).append(item)
    return list(by_window.values())


def vlm_batch_request_summary(
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


def vlm_request_artifact_payload(
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
        "request_metrics": combine_vlm_request_metrics(request_summaries),
        "batches": request_summaries,
    }


def combine_vlm_request_metrics(request_summaries: list[dict[str, Any]]) -> dict[str, Any]:
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


def run_vlm_batches(
    clients: list[OpenAICompatibleVlmClient],
    batches: list[list[Any]],
    *,
    sensitive_files: list[str],
    active_apps: list[str],
    workers_per_key: int,
    retry_attempts: int = 3,
    retry_backoff_seconds: float = 1.0,
) -> dict[str, Any]:
    if not clients:
        return {"batches": [], "errors": ["vlm_client_pool_empty"], "events": [], "parse_errors": [], "usage": {}}

    results: list[dict[str, Any]] = []
    errors: list[str] = []
    dispatcher = _shared_vlm_dispatcher(
        clients,
        workers_per_key=workers_per_key,
        retry_attempts=retry_attempts,
        retry_backoff_seconds=retry_backoff_seconds,
    )
    future_to_index = {
        dispatcher.submit(
            clients=clients,
            batch_index=index,
            frames=batch,
            sensitive_files=sensitive_files,
            active_apps=active_apps,
        ): index
        for index, batch in enumerate(batches)
    }
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
        "dispatch": _vlm_dispatch_metrics(dispatcher, results),
    }


def vlm_response_artifact_payload(vlm_results: dict[str, Any]) -> dict[str, Any]:
    batch_results = list(vlm_results.get("batches") or [])
    errors = list(vlm_results.get("errors") or [])
    if len(batch_results) == 1 and not errors:
        return vlm_response_to_dict(batch_results[0]["response"])
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
                **vlm_response_to_dict(result["response"]),
            }
            for result in batch_results
        ],
    }


def vlm_parse_artifact_payload(vlm_results: dict[str, Any]) -> dict[str, Any]:
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


def vlm_response_to_dict(response: Any) -> dict[str, Any]:
    return {
        "provider": response.provider,
        "model": response.model,
        "dry_run": response.dry_run,
        "usage": response.usage,
        "text": response.text,
        "raw_payload": response.raw_payload,
    }


def vlm_dispatcher_snapshots() -> list[dict[str, Any]]:
    """Return live, key-free queue metrics for Release progress reporting."""

    with _VLM_DISPATCHER_GUARD:
        return [dispatcher.snapshot() for dispatcher in _VLM_DISPATCHERS.values()]


def _validate_vlm_evidence(parse_result: VlmParseResult, frames: list[Any]) -> VlmParseResult:
    allowed: set[str] = set()
    for item in frames:
        frame = getattr(item, "frame", item)
        frame_id = str(getattr(frame, "frame_id", "") or "")
        if frame_id:
            allowed.add(frame_id)
        for source in getattr(item, "source_frames", ()) or ():
            if isinstance(source, dict) and str(source.get("frame_id") or ""):
                allowed.add(str(source["frame_id"]))

    events = []
    dropped = list(parse_result.dropped_events)
    errors = list(parse_result.parse_errors)
    for index, event in enumerate(parse_result.events):
        valid_ids = tuple(frame_id for frame_id in event.evidence_frame_ids if frame_id in allowed)
        invalid_ids = tuple(frame_id for frame_id in event.evidence_frame_ids if frame_id not in allowed)
        if invalid_ids:
            errors.append(f"event_invalid_evidence_frame_ids[{index}]: {'|'.join(invalid_ids)}")
        if not valid_ids:
            dropped.append(
                {
                    "reason": "missing_valid_evidence_frame_ids",
                    "event": {
                        "app_name": event.app_name,
                        "operation_type": event.operation_type,
                        "evidence_frame_ids": list(event.evidence_frame_ids),
                    },
                }
            )
            continue
        events.append(replace(event, evidence_frame_ids=valid_ids))
    return replace(parse_result, events=events, dropped_events=dropped, parse_errors=errors)


def _shared_vlm_endpoint_locks(
    clients: list[OpenAICompatibleVlmClient],
    *,
    workers_per_key: int,
) -> dict[int, threading.BoundedSemaphore]:
    limit = max(1, workers_per_key)
    locks: dict[int, threading.BoundedSemaphore] = {}
    with _VLM_ENDPOINT_LOCK_GUARD:
        for client in clients:
            config = client.config
            identity = (str(getattr(config, "vlm_base_url", "")).rstrip("/"), str(getattr(config, "vlm_api_key", "")), limit)
            lock = _VLM_ENDPOINT_LOCKS.get(identity)
            if lock is None:
                lock = threading.BoundedSemaphore(limit)
                _VLM_ENDPOINT_LOCKS[identity] = lock
            locks[id(client)] = lock
    return locks


class _SharedVlmDispatcher:
    """One process-wide FIFO queue for a stable endpoint/key pool."""

    def __init__(
        self,
        *,
        parallelism: int,
        workers_per_key: int,
        retry_attempts: int,
        retry_backoff_seconds: float,
    ):
        self.parallelism = max(1, parallelism)
        self.workers_per_key = max(1, workers_per_key)
        self.retry_attempts = max(1, retry_attempts)
        self.retry_backoff_seconds = max(0.0, retry_backoff_seconds)
        self.executor = ThreadPoolExecutor(max_workers=self.parallelism, thread_name_prefix="dld_vlm")
        self._lock = threading.Lock()
        self._next_client = 0
        self._queued = 0
        self._in_flight = 0
        self._submitted = 0
        self._completed = 0
        self._failed = 0
        self._endpoint_waiting: dict[str, int] = {}
        self._endpoint_active: dict[str, int] = {}
        self._endpoint_completed: dict[str, int] = {}

    def submit(
        self,
        *,
        clients: list[OpenAICompatibleVlmClient],
        batch_index: int,
        frames: list[Any],
        sensitive_files: list[str],
        active_apps: list[str],
    ):
        submitted_at = time.perf_counter()
        with self._lock:
            start_client = self._next_client % len(clients)
            self._next_client += 1
            self._queued += 1
            self._submitted += 1
        return self.executor.submit(
            self._run_one,
            clients,
            batch_index,
            frames,
            sensitive_files,
            active_apps,
            start_client,
            submitted_at,
        )

    def _run_one(
        self,
        clients: list[OpenAICompatibleVlmClient],
        batch_index: int,
        frames: list[Any],
        sensitive_files: list[str],
        active_apps: list[str],
        start_client: int,
        submitted_at: float,
    ) -> dict[str, Any]:
        with self._lock:
            self._queued -= 1
            self._in_flight += 1
        queue_wait_seconds = time.perf_counter() - submitted_at
        client_locks = _shared_vlm_endpoint_locks(clients, workers_per_key=self.workers_per_key)
        ordered_clients = clients[start_client:] + clients[:start_client]
        retry_warnings: list[str] = []
        response = None
        try:
            for client_index, client in enumerate(ordered_clients):
                lock = client_locks[id(client)]
                endpoint = str(getattr(client.config, "vlm_base_url", "")).rstrip("/")
                for retry_index in range(self.retry_attempts):
                    slot_acquired = False
                    with self._lock:
                        self._endpoint_waiting[endpoint] = self._endpoint_waiting.get(endpoint, 0) + 1
                    try:
                        with lock:
                            with self._lock:
                                self._endpoint_waiting[endpoint] -= 1
                                self._endpoint_active[endpoint] = self._endpoint_active.get(endpoint, 0) + 1
                            slot_acquired = True
                            try:
                                response = client.analyze(frames, sensitive_files=sensitive_files, active_apps=active_apps)
                            finally:
                                with self._lock:
                                    self._endpoint_active[endpoint] -= 1
                        with self._lock:
                            self._endpoint_completed[endpoint] = self._endpoint_completed.get(endpoint, 0) + 1
                        break
                    except Exception as exc:
                        can_retry = retry_index + 1 < self.retry_attempts and _is_transient_vlm_error(exc)
                        if can_retry:
                            delay = self.retry_backoff_seconds * (2**retry_index)
                            retry_warnings.append(
                                f"vlm_transient_retry[{batch_index}:{retry_index + 1}]: {type(exc).__name__}: {exc}"
                            )
                            if delay:
                                time.sleep(delay)
                            continue
                        if client_index + 1 == len(ordered_clients):
                            raise
                        retry_warnings.append(f"vlm_key_retry[{batch_index}]: {type(exc).__name__}: {exc}")
                        break
                    finally:
                        if not slot_acquired:
                            with self._lock:
                                self._endpoint_waiting[endpoint] -= 1
                if response is not None:
                    break
            if response is None:
                raise RuntimeError("vlm_response_unavailable")
            parse_result = _validate_vlm_evidence(
                parse_vlm_response_detailed(response.text, keywords=sensitive_files),
                frames,
            )
            with self._lock:
                self._completed += 1
            return {
                "batch_index": batch_index,
                "frame_count": len(frames),
                "response": response,
                "parse_result": parse_result,
                "retry_warnings": retry_warnings,
                "queue_wait_seconds": round(queue_wait_seconds, 6),
            }
        except Exception:
            with self._lock:
                self._failed += 1
            raise
        finally:
            with self._lock:
                self._in_flight -= 1

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "parallelism": self.parallelism,
                "queued_batches": self._queued,
                "in_flight_batches": self._in_flight,
                "submitted_batches": self._submitted,
                "completed_batches": self._completed,
                "failed_batches": self._failed,
                "endpoints": {
                    endpoint: {
                        "waiting_for_slot": self._endpoint_waiting.get(endpoint, 0),
                        "active_requests": self._endpoint_active.get(endpoint, 0),
                        "completed_requests": self._endpoint_completed.get(endpoint, 0),
                    }
                    for endpoint in sorted(set(self._endpoint_waiting) | set(self._endpoint_active) | set(self._endpoint_completed))
                },
            }


def _shared_vlm_dispatcher(
    clients: list[OpenAICompatibleVlmClient],
    *,
    workers_per_key: int,
    retry_attempts: int = 3,
    retry_backoff_seconds: float = 1.0,
) -> _SharedVlmDispatcher:
    identity = tuple(
        (
            str(getattr(client.config, "vlm_base_url", "")).rstrip("/"),
            str(getattr(client.config, "vlm_api_key", "")),
        )
        for client in clients
    )
    key = (identity, max(1, workers_per_key), max(1, retry_attempts), max(0.0, retry_backoff_seconds))
    with _VLM_DISPATCHER_GUARD:
        dispatcher = _VLM_DISPATCHERS.get(key)
        if dispatcher is None:
            dispatcher = _SharedVlmDispatcher(
                parallelism=len(clients) * max(1, workers_per_key),
                workers_per_key=workers_per_key,
                retry_attempts=retry_attempts,
                retry_backoff_seconds=retry_backoff_seconds,
            )
            _VLM_DISPATCHERS[key] = dispatcher
        return dispatcher


def _is_transient_vlm_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return (
        isinstance(exc, TimeoutError)
        or "timed out" in text
        or "429" in text
        or "throttl" in text
        or "temporar" in text
        or "connection reset" in text
        or "eof occurred" in text
        or "http_error: 5" in text
    )


def _vlm_dispatch_metrics(dispatcher: _SharedVlmDispatcher, results: list[dict[str, Any]]) -> dict[str, Any]:
    waits = [float(item.get("queue_wait_seconds", 0.0)) for item in results]
    return {
        "mode": "shared_process_queue",
        "batch_count": len(results),
        "queue_wait_seconds_total": round(sum(waits), 6),
        "queue_wait_seconds_max": round(max(waits, default=0.0), 6),
        "queue_wait_seconds_mean": round(sum(waits) / len(waits), 6) if waits else 0.0,
        "snapshot": dispatcher.snapshot(),
    }


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
