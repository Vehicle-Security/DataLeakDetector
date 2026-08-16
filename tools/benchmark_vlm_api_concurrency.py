#!/usr/bin/env python3
"""Replay one real 4x1 VLM request from each selected case at fixed concurrency.

The runner consumes the request summaries emitted by ScreenGuard itself. It
does not reselect keyframes, regenerate grids, or invoke post-VLM reasoning.
"""

from __future__ import annotations

import argparse
import base64
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import math
import mimetypes
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Callable


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "main"))

from data_leak_detector.frame_analyzer.config import VisionConfig


DEFAULT_CONCURRENCY = (1, 2, 4, 8, 12, 16, 20, 24, 32)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-list", default="spec/config/concurrency_benchmark_32_cases.txt")
    parser.add_argument("--case-debug-root", required=True, help="Release case_debug root containing vlm_request.json files.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model", default="qwen3.7-plus")
    parser.add_argument("--concurrency", default=",".join(map(str, DEFAULT_CONCURRENCY)))
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--request-timeout", type=int, default=180, help="Hard total timeout for one API request in seconds.")
    args = parser.parse_args()
    if args.repeats < 1 or args.request_timeout < 1:
        parser.error("--repeats and --request-timeout must be positive")

    case_ids = _load_case_ids(Path(args.case_list))
    limits = _parse_limits(args.concurrency)
    request_root = Path(args.case_debug_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config = VisionConfig.from_env()
    endpoint = next((item for item in config.effective_vlm_endpoints() if item.name == "token_plan"), None)
    if endpoint is None:
        raise RuntimeError("no token-plan VLM endpoint/key configured")

    requests = _load_real_requests(request_root, case_ids)
    manifest = {
        "case_count": len(case_ids),
        "request_count": len(requests),
        "case_debug_root": str(request_root),
        "model": args.model,
        "chat_url": endpoint.chat_url or endpoint.base_url.rstrip("/") + "/chat/completions",
        "requests": [{key: item[key] for key in ("case_id", "source_file", "batch_index", "image_count", "prompt_chars")} for item in requests],
    }
    _write_json(output_dir / "request_manifest.json", manifest)
    _write_json(output_dir / "progress.json", {"state": "prepared", **manifest})

    print(f"Loaded {len(requests)} fixed real requests from {len(case_ids)} cases.", flush=True)
    print(f"Model: {args.model}", flush=True)
    print(f"Endpoint: {manifest['chat_url']}", flush=True)
    print(f"Concurrency: {limits}; repeats: {args.repeats}", flush=True)

    # Warm the client/connection path without including this call in metrics.
    print("Starting unmeasured API warmup...", flush=True)
    warmup = _send_request(requests[0], endpoint.api_key, manifest["chat_url"], args.model, args.request_timeout)
    if not warmup["success"]:
        raise RuntimeError(f"API warmup failed: {warmup['error']}")
    print(f"Warmup complete: {warmup['latency_seconds']:.3f}s", flush=True)

    all_records: list[dict[str, Any]] = []
    events_path = output_dir / "request_results.jsonl"
    with events_path.open("w", encoding="utf-8") as events:
        for limit in limits:
            for repeat in range(1, args.repeats + 1):
                print(f"[{_now()}] Starting API run concurrency={limit} repeat={repeat} requests={len(requests)}", flush=True)
                run_started = time.perf_counter()
                completed_requests = 0
                successful_requests = 0

                def report_request(record: dict[str, Any]) -> None:
                    nonlocal completed_requests, successful_requests
                    completed_requests += 1
                    successful_requests += int(record["success"])
                    _write_json(output_dir / "progress.json", {
                        "state": "running", **manifest,
                        "completed_runs": len({(item["concurrency"], item["repeat"]) for item in all_records}),
                        "current": {
                            "concurrency": limit,
                            "repeat": repeat,
                            "completed_requests": completed_requests,
                            "successful_requests": successful_requests,
                            "total_requests": len(requests),
                        },
                    })
                    outcome = "success" if record["success"] else record.get("error", "failed")
                    print(
                        f"[{_now()}] API request concurrency={limit} repeat={repeat} "
                        f"progress={completed_requests}/{len(requests)} case={record['case_id']} "
                        f"result={outcome} latency={record['latency_seconds']:.3f}s",
                        flush=True,
                    )

                records = _run_once(
                    requests, limit, endpoint.api_key, manifest["chat_url"], args.model, args.request_timeout, report_request,
                )
                wall_seconds = time.perf_counter() - run_started
                for record in records:
                    record.update({"concurrency": limit, "repeat": repeat, "run_wall_seconds": round(wall_seconds, 6)})
                    events.write(json.dumps(record, ensure_ascii=False) + "\n")
                events.flush()
                all_records.extend(records)
                success = sum(item["success"] for item in records)
                _write_json(output_dir / "progress.json", {
                    "state": "running", **manifest, "completed_runs": len({(item["concurrency"], item["repeat"]) for item in all_records}),
                    "current": {"concurrency": limit, "repeat": repeat, "successful_requests": success, "wall_seconds": round(wall_seconds, 3)},
                })
                print(f"[{_now()}] Completed API run concurrency={limit} repeat={repeat} success={success}/{len(records)} wall={wall_seconds:.3f}s", flush=True)

    summary = {"manifest": manifest, "rows": _summarize(all_records, limits, args.repeats), "requests": all_records}
    _write_json(output_dir / "summary.json", summary)
    _write_json(output_dir / "progress.json", {"state": "completed", **manifest, "rows": summary["rows"]})
    print(json.dumps({"rows": summary["rows"]}, ensure_ascii=False, indent=2), flush=True)
    return 0


def _load_case_ids(path: Path) -> list[str]:
    if not path.is_file():
        raise ValueError(f"case list not found: {path}")
    ids = [line.strip().replace("\\", "/").strip("/") for line in path.read_text(encoding="utf-8").splitlines()]
    ids = [item for item in ids if item and not item.startswith("#")]
    if len(ids) != len(set(ids)):
        raise ValueError("case list contains duplicate IDs")
    if not ids:
        raise ValueError("case list is empty")
    return ids


def _parse_limits(value: str) -> tuple[int, ...]:
    try:
        limits = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise ValueError(f"invalid concurrency list: {value}") from exc
    if not limits or any(item < 1 for item in limits) or len(limits) != len(set(limits)):
        raise ValueError(f"invalid concurrency list: {value}")
    return limits


def _load_real_requests(root: Path, case_ids: list[str]) -> list[dict[str, Any]]:
    if not root.is_dir():
        raise ValueError(f"case debug root not found: {root}")
    selected: list[dict[str, Any]] = []
    for case_id in case_ids:
        candidates: list[dict[str, Any]] = []
        for request_file in sorted((root / Path(case_id)).rglob("vlm_request.json")):
            payload = json.loads(request_file.read_text(encoding="utf-8"))
            # Older release artifacts contain one request at the top level;
            # newer artifacts preserve their original split batches.
            batches = payload.get("batches") if isinstance(payload.get("batches"), list) else [payload]
            for batch in batches:
                frames = list(batch.get("frames") or [])
                images = [str(frame.get("image_path") or "") for frame in frames]
                if not images or any(not _resolve_image(path).is_file() for path in images):
                    continue
                candidates.append({
                    "case_id": case_id,
                    "source_file": str(request_file),
                    "batch_index": int(batch.get("batch_index", 0)),
                    "image_count": len(images),
                    "prompt_chars": len(str(batch.get("prompt") or "")),
                    "prompt": str(batch.get("prompt") or ""),
                    "image_paths": images,
                    "enable_thinking": batch.get("enable_thinking"),
                })
        if not candidates:
            raise ValueError(f"no replayable VLM request for case: {case_id}")
        max_images = max(item["image_count"] for item in candidates)
        largest = [item for item in candidates if item["image_count"] == max_images]
        selected.append(largest[len(largest) // 2])
    return selected


def _resolve_image(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def _run_once(
    requests: list[dict[str, Any]],
    limit: int,
    api_key: str,
    chat_url: str,
    model: str,
    timeout: int,
    on_result: Callable[[dict[str, Any]], None] | None = None,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=min(limit, len(requests))) as pool:
        futures = [pool.submit(_send_request, item, api_key, chat_url, model, timeout) for item in requests]
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            if on_result is not None:
                on_result(result)
    return results


def _send_request(item: dict[str, Any], api_key: str, chat_url: str, model: str, timeout: int) -> dict[str, Any]:
    started = time.perf_counter()
    record = {key: item[key] for key in ("case_id", "source_file", "batch_index", "image_count", "prompt_chars")}
    try:
        content: list[dict[str, Any]] = [{"type": "text", "text": item["prompt"]}]
        for image_path in item["image_paths"]:
            image = _resolve_image(image_path)
            mime = mimetypes.guess_type(image.name)[0] or "image/jpeg"
            encoded = base64.b64encode(image.read_bytes()).decode("ascii")
            content.append({"type": "image_url", "image_url": {"url": f"data:{mime};base64,{encoded}"}})
        body: dict[str, Any] = {"model": model, "messages": [{"role": "user", "content": content}], "temperature": 0}
        if item.get("enable_thinking") is not None:
            body["enable_thinking"] = bool(item["enable_thinking"])
        # curl's --max-time enforces a process-level deadline even when a TLS
        # read remains stuck after urllib's socket timeout should have fired.
        completed = subprocess.run(
            [
                "curl", "--silent", "--show-error", "--max-time", str(timeout),
                "-H", f"Authorization: Bearer {api_key}",
                "-H", "Content-Type: application/json",
                "-H", f"User-Agent: Python-urllib/{sys.version_info.major}.{sys.version_info.minor}",
                "--data-binary", "@-", "--write-out", "\n__DLD_HTTP_STATUS__:%{http_code}", chat_url,
            ],
            input=json.dumps(body, ensure_ascii=False).encode("utf-8"),
            capture_output=True,
            timeout=timeout + 10,
        )
        text = completed.stdout.decode("utf-8", errors="replace")
        body_text, marker, status_text = text.rpartition("\n__DLD_HTTP_STATUS__:")
        status = int(status_text) if marker and status_text.isdigit() else None
        if completed.returncode != 0:
            raise RuntimeError(f"curl_exit_{completed.returncode}: {completed.stderr.decode('utf-8', errors='replace').strip()[:400]}")
        payload = json.loads(body_text)
        if status is None or not 200 <= status < 300 or not payload.get("choices"):
            raise RuntimeError(f"http_{status}: {body_text[:400]}")
        record.update({"success": True, "http_status": status, "response_model": payload.get("model", ""), "usage": payload.get("usage") if isinstance(payload.get("usage"), dict) else {}})
    except subprocess.TimeoutExpired:
        record.update({"success": False, "http_status": None, "error": f"request_timeout_after_{timeout}s"})
    except Exception as exc:
        record.update({"success": False, "http_status": None, "error": f"{type(exc).__name__}: {exc}"[:500]})
    record["latency_seconds"] = round(time.perf_counter() - started, 6)
    return record


def _summarize(records: list[dict[str, Any]], limits: tuple[int, ...], repeats: int) -> list[dict[str, Any]]:
    rows = []
    for limit in limits:
        subset = [item for item in records if item["concurrency"] == limit]
        successful = [item for item in subset if item["success"]]
        latencies = sorted(item["latency_seconds"] for item in successful)
        per_repeat: list[float] = []
        for repeat in range(1, repeats + 1):
            run = [item for item in subset if item["repeat"] == repeat]
            if run:
                per_repeat.append(float(run[0]["run_wall_seconds"]))
        throughput = (len(successful) / sum(per_repeat) * 60) if per_repeat and sum(per_repeat) else None
        rows.append({
            "concurrency": limit,
            "requests": len(subset),
            "successful_requests": len(successful),
            "success_rate": round(len(successful) / len(subset) * 100, 3) if subset else 0.0,
            "throughput_request_per_min": round(throughput, 3) if throughput is not None else None,
            "p50_latency_seconds": round(_percentile(latencies, 0.5), 3) if latencies else None,
            "p95_latency_seconds": round(_percentile(latencies, 0.95), 3) if latencies else None,
        })
    return rows


def _percentile(values: list[float], fraction: float) -> float:
    position = (len(values) - 1) * fraction
    lower, upper = math.floor(position), math.ceil(position)
    return values[lower] if lower == upper else values[lower] + (values[upper] - values[lower]) * (position - lower)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


if __name__ == "__main__":
    raise SystemExit(main())
