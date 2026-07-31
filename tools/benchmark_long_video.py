#!/usr/bin/env python3
"""Reproducible Figure-2 runner: direct-video MLLM vs. formal ScreenGuard.

This runner does not implement either method.  It only prepares identical
time-prefix clips/logs, publishes clips when requested, invokes one direct
``video_url`` request, and invokes the repository's canonical ``run_pipeline``
for ScreenGuard.  In particular, it never samples video frames for Direct and
never calls log mining or frame selection itself for ScreenGuard.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import mimetypes
import os
from pathlib import Path
import platform
import shutil
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
import uuid
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "main"))

from data_leak_detector.datasets import discover_data_case
from data_leak_detector.frame_analyzer.config import VisionConfig
from data_leak_detector.io import load_json_records, normalize_logs
from data_leak_detector.pipeline import run_pipeline
from data_leak_detector.sensitivity import resolve_sensitive_files_config


DEFAULT_DURATIONS = (10, 20, 30, 45, 60)
METHODS = ("direct_mllm", "screenguard")
DIRECT_VIDEO_PROMPT = """Analyze this entire screen-recording video for possible sensitive-data leakage.
Inspect all visible applications and actions. Identify possible sensitive files, their transformations
(for example screenshots), and any external sharing/upload behavior. Return only a concise JSON object
with an `events` array and a `verdict` string. Do not assume that a sensitive file exists when the video
does not show one."""


@dataclass(frozen=True)
class PreparedInput:
    scenario: str
    case_dir: str
    duration_min: int
    video_file: str
    log_file: str
    video_duration_seconds: float
    prefix_log_records: int
    recording_start_ms: int

    @property
    def key(self) -> str:
        return f"{self.scenario}|{self.duration_min}"


@dataclass(frozen=True)
class Job:
    prepared: PreparedInput
    method: str

    @property
    def key(self) -> str:
        return f"{self.prepared.key}|{self.method}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", action="append", required=True, metavar="SCENARIO=PATH")
    parser.add_argument("--durations", default=",".join(map(str, DEFAULT_DURATIONS)))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--actions", default="prepare,publish,direct,screenguard",
                        help="comma-separated prepare,publish,direct,screenguard")
    parser.add_argument("--methods", default=",".join(METHODS),
                        help="comma-separated direct_mllm,screenguard")
    parser.add_argument("--publisher", choices=("uguu", "none"), default="uguu",
                        help="temporary public host used only by Direct")
    parser.add_argument("--direct-url-file", default="",
                        help="existing direct_video_urls.json; avoids a new upload")
    parser.add_argument("--ffmpeg", default="", help="ffmpeg executable; defaults to imageio-ffmpeg or PATH")
    parser.add_argument("--host-label", default=socket.gethostname())
    parser.add_argument("--vlm-workers", type=int, default=0,
                        help="override ScreenGuard VLM workers (0 keeps .env)")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)

    cases = _parse_cases(args.case, parser)
    durations = _parse_csv_ints(args.durations, parser)
    actions = _parse_csv(args.actions)
    methods = _parse_csv(args.methods)
    unknown_actions = set(actions) - {"prepare", "publish", "direct", "screenguard"}
    unknown_methods = set(methods) - set(METHODS)
    if unknown_actions or unknown_methods:
        parser.error(f"unknown actions={sorted(unknown_actions)}, methods={sorted(unknown_methods)}")
    if not actions:
        parser.error("--actions must not be empty")
    if args.vlm_workers < 0:
        parser.error("--vlm-workers must be non-negative")

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.vlm_workers:
        os.environ["DLD_VLM_WORKERS"] = str(args.vlm_workers)
    os.environ.setdefault("DLD_VLM_GRID_LAYOUT", "4x1")
    os.environ.setdefault("DLD_VLM_MAX_IMAGE_SIDE", "1280")
    config = VisionConfig.from_env()
    if not config.effective_vlm_endpoints():
        raise RuntimeError("no VLM endpoint/key configured")

    metadata_file = output_dir / "benchmark_metadata.json"
    events_file = output_dir / "benchmark_events.jsonl"
    results_file = output_dir / "benchmark_results.json"
    urls_file = output_dir / "direct_video_urls.json"
    metadata = {
        "schema_version": 2,
        "methodology": {
            "direct_mllm": "One OpenAI-compatible video_url request per clipped video; no client-side frame extraction.",
            "screenguard": "Canonical data_leak_detector.pipeline.run_pipeline over the same local clip and clipped monitor log.",
            "timing_scope": "Input clipping and temporary-host upload are excluded; each method is timed from its analysis call.",
        },
        "host_label": args.host_label,
        "hostname": socket.gethostname(),
        "started_at": _now(),
        "source_commit": _git_commit(),
        "runner_sha256": _sha256(Path(__file__)),
        "python": sys.version,
        "platform": platform.platform(),
        "gpu": _gpu_names(),
        "durations_min": list(durations),
        "methods": list(methods),
        "actions": list(actions),
        "cases": cases,
        "vlm_model": config.vlm_model,
        "vlm_grid_layout": config.vlm_grid_layout,
        "vlm_workers": config.vlm_workers,
        "publisher": args.publisher,
    }
    _write_json(metadata_file, metadata)

    prepared = _prepare_all(cases, durations, output_dir, args.ffmpeg) if "prepare" in actions else _load_prepared(output_dir)
    _append(events_file, {"event": "prepared_inputs", "at": _now(), "count": len(prepared)})

    urls_path = Path(args.direct_url_file).resolve() if args.direct_url_file else urls_file
    urls = _load_urls(urls_path)
    if "publish" in actions:
        if args.publisher == "none":
            raise RuntimeError("--publisher none cannot be used with publish")
        urls = _publish_missing(prepared, urls, publisher=args.publisher, events_file=events_file)
        _write_json(urls_file, {"schema_version": 1, "publisher": args.publisher, "videos": urls})
    if "direct" in actions:
        missing = [item.key for item in prepared if item.key not in urls]
        if missing:
            raise RuntimeError(f"Direct requires public URLs for: {', '.join(missing)}")

    prior = _load_completed(events_file) if args.resume else {}
    results = dict(prior)
    requested_methods = [method for method in methods if method == "direct_mllm" and "direct" in actions or method == "screenguard" and "screenguard" in actions]
    jobs = [Job(item, method) for item in prepared for method in requested_methods]
    _append(events_file, {"event": "run_started", "at": _now(), "job_count": len(jobs), "resumed": len(prior)})
    for index, job in enumerate(jobs, 1):
        if args.resume and job.key in results:
            _append(events_file, {"event": "job_skipped", "at": _now(), "job": job.key, "reason": "resume"})
            continue
        _append(events_file, {"event": "job_started", "at": _now(), "job": job.key, "job_index": index, "job_count": len(jobs)})
        started = time.perf_counter()
        try:
            metrics = (
                _direct_video(job.prepared, config, urls[job.prepared.key])
                if job.method == "direct_mllm"
                else _formal_screenguard(job.prepared, output_dir)
            )
            result = {
                "event": "job_completed", "at": _now(), "host_label": args.host_label,
                "job": job.key, "scenario": job.prepared.scenario, "duration_min": job.prepared.duration_min,
                "method": job.method, "analysis_time_seconds": round(time.perf_counter() - started, 3),
                "status": "completed", **asdict(job.prepared), **metrics,
            }
        except Exception as exc:  # preserve progress on service/network failures
            result = {
                "event": "job_failed", "at": _now(), "host_label": args.host_label,
                "job": job.key, "scenario": job.prepared.scenario, "duration_min": job.prepared.duration_min,
                "method": job.method, "analysis_time_seconds": round(time.perf_counter() - started, 3),
                "status": "failed", "error": f"{type(exc).__name__}: {exc}",
            }
        _append(events_file, result)
        results[job.key] = result
        _write_results(results_file, metadata, results)

    metadata.update({
        "finished_at": _now(),
        "completed_jobs": sum(item.get("status") == "completed" for item in results.values()),
        "failed_jobs": sum(item.get("status") == "failed" for item in results.values()),
    })
    _write_json(metadata_file, metadata)
    _write_results(results_file, metadata, results)
    _append(events_file, {"event": "run_finished", "at": _now(), **metadata})
    return 1 if metadata["failed_jobs"] else 0


def _prepare_all(cases: dict[str, str], durations: tuple[int, ...], output_dir: Path, ffmpeg_arg: str) -> list[PreparedInput]:
    ffmpeg = _ffmpeg(ffmpeg_arg)
    prepared: list[PreparedInput] = []
    for scenario, case_dir in cases.items():
        case = discover_data_case(case_dir)
        if not case.video_file:
            raise FileNotFoundError(f"no recording for {case_dir}")
        records = load_json_records(case.log_file)
        normalized = normalize_logs(records, session_start_ms=case.recording_start_ms)
        source_duration = _video_duration_seconds(ffmpeg, case.video_file)
        for duration_min in durations:
            duration_seconds = min(duration_min * 60.0, source_duration)
            if duration_seconds < duration_min * 60.0 - 0.5:
                raise ValueError(f"{scenario} is only {source_duration:.1f}s; cannot make {duration_min}m clip")
            target_dir = output_dir / "inputs" / scenario / f"{duration_min:02d}m"
            target_dir.mkdir(parents=True, exist_ok=True)
            video_file = target_dir / "recording.mp4"
            log_file = target_dir / "keyevents.json"
            if not video_file.exists() or _video_duration_seconds(ffmpeg, video_file) < duration_seconds - 1.0:
                _clip_video(ffmpeg, case.video_file, video_file, duration_seconds)
            prefix_records = [record for record, event in zip(records, normalized) if 0 <= event.video_time_ms < duration_seconds * 1000]
            _write_json(log_file, prefix_records)
            prepared.append(PreparedInput(
                scenario=scenario, case_dir=str(Path(case_dir).resolve()), duration_min=duration_min,
                video_file=str(video_file), log_file=str(log_file),
                video_duration_seconds=round(_video_duration_seconds(ffmpeg, video_file), 3),
                prefix_log_records=len(prefix_records), recording_start_ms=case.recording_start_ms,
            ))
    _write_json(output_dir / "prepared_inputs.json", {"schema_version": 1, "inputs": [asdict(item) for item in prepared]})
    return prepared


def _load_prepared(output_dir: Path) -> list[PreparedInput]:
    path = output_dir / "prepared_inputs.json"
    if not path.exists():
        raise FileNotFoundError(f"no prepared inputs at {path}; include prepare in --actions")
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [PreparedInput(**item) for item in payload.get("inputs", [])]


def _clip_video(ffmpeg: str, source: Path, target: Path, duration_seconds: float) -> None:
    temporary = target.with_suffix(".partial.mp4")
    command = [
        ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-i", str(source), "-t", f"{duration_seconds:.3f}",
        "-map", "0:v:0", "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
        "-movflags", "+faststart", str(temporary),
    ]
    subprocess.run(command, check=True)
    temporary.replace(target)


def _direct_video(prepared: PreparedInput, config: VisionConfig, url_entry: dict[str, Any]) -> dict[str, Any]:
    url = str(url_entry.get("url") or "")
    if not url.startswith("https://"):
        raise ValueError(f"Direct video URL must be HTTPS: {url!r}")
    endpoint = config.effective_vlm_endpoints()[0]
    chat_url = endpoint.chat_url.strip() or endpoint.base_url.rstrip("/") + "/chat/completions"
    body = {
        "model": config.vlm_model,
        "messages": [{"role": "user", "content": [
            {"type": "video_url", "video_url": {"url": url}},
            {"type": "text", "text": DIRECT_VIDEO_PROMPT},
        ]}],
        "temperature": 0,
    }
    request = urllib.request.Request(
        chat_url, data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
        headers={"Authorization": f"Bearer {endpoint.api_key}", "Content-Type": "application/json"}, method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=config.vlm_timeout_seconds) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"direct_video_http_error: {exc.code} {detail}") from exc
    usage = payload.get("usage") if isinstance(payload.get("usage"), dict) else {}
    text = payload.get("choices", [{}])[0].get("message", {}).get("content", "")
    return {
        "direct_video_url": url, "direct_response_file": _write_direct_response(prepared, payload),
        "direct_response_chars": len(str(text)), "usage": usage, "total_tokens": _total_tokens(usage),
        "direct_client_side_frame_extraction": False,
    }


def _write_direct_response(prepared: PreparedInput, payload: dict[str, Any]) -> str:
    path = Path(prepared.video_file).parent / "direct_mllm_response.json"
    _write_json(path, payload)
    return str(path)


def _formal_screenguard(prepared: PreparedInput, output_dir: Path) -> dict[str, Any]:
    case = discover_data_case(prepared.case_dir)
    sensitive_config = resolve_sensitive_files_config(case.case_dir)
    report = run_pipeline(
        log_file=prepared.log_file, video_file=prepared.video_file,
        output_dir=output_dir / "screenguard_reports" / prepared.scenario / f"{prepared.duration_min:02d}m",
        sensitive_files_config=sensitive_config, vision_enabled=True, max_vlm_frames=-1,
        neo4j_log_miner=False, vision_debug_artifacts=True,
        session_start_ms=prepared.recording_start_ms,
        case_name=f"figure2_{prepared.scenario}_{prepared.duration_min}m",
    )
    vision = _find_vision(report)
    usage = _find_usage(report)
    return {
        "screenguard_report_file": report.get("report_file", ""),
        "analysis_windows": vision.get("analysis_windows", 0),
        "keyframes": vision.get("keyframes", 0),
        "vlm_frames": vision.get("vlm_frames", 0),
        "vlm_batches": vision.get("vlm_batches", 0),
        "usage": usage, "total_tokens": _total_tokens(usage),
        "pipeline_entrypoint": "data_leak_detector.pipeline.run_pipeline",
    }


def _find_vision(report: dict[str, Any]) -> dict[str, Any]:
    frame_analyzer = report.get("frame_analyzer") if isinstance(report.get("frame_analyzer"), dict) else {}
    direct = frame_analyzer.get("statistics") if isinstance(frame_analyzer.get("statistics"), dict) else {}
    if isinstance(direct.get("vision"), dict):
        return direct["vision"]
    for session in frame_analyzer.get("sessions", []) if isinstance(frame_analyzer.get("sessions"), list) else []:
        stats = session.get("statistics") if isinstance(session, dict) else {}
        if isinstance(stats.get("vision"), dict):
            return stats["vision"]
    return {}


def _find_usage(report: dict[str, Any]) -> dict[str, Any]:
    vision = _find_vision(report)
    usage = vision.get("vlm_usage") if isinstance(vision.get("vlm_usage"), dict) else {}
    if usage:
        return usage
    metrics = vision.get("vlm_request_metrics") if isinstance(vision.get("vlm_request_metrics"), dict) else {}
    usage = metrics.get("usage") if isinstance(metrics.get("usage"), dict) else {}
    if usage:
        return usage
    response_files = report.get("detail_files") if isinstance(report.get("detail_files"), dict) else {}
    for value in response_files.values():
        path = Path(str(value))
        if path.name == "vlm_response.json" and path.exists():
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue
            if isinstance(payload.get("usage"), dict):
                return payload["usage"]
    return {}


def _publish_missing(prepared: list[PreparedInput], urls: dict[str, dict[str, Any]], *, publisher: str, events_file: Path) -> dict[str, dict[str, Any]]:
    for item in prepared:
        if item.key in urls and str(urls[item.key].get("url") or "").startswith("https://"):
            continue
        _append(events_file, {"event": "video_publish_started", "at": _now(), "input": item.key, "publisher": publisher})
        url = _upload_uguu(Path(item.video_file)) if publisher == "uguu" else ""
        urls[item.key] = {"url": url, "publisher": publisher, "uploaded_at": _now(), "sha256": _sha256(Path(item.video_file))}
        _append(events_file, {"event": "video_publish_completed", "at": _now(), "input": item.key, "publisher": publisher, "url": url})
    return urls


def _upload_uguu(path: Path) -> str:
    """Upload synthetic test media to Uguu's temporary host and return its HTTPS URL."""
    boundary = f"----dld-{uuid.uuid4().hex}"
    mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    prefix = (f"--{boundary}\r\nContent-Disposition: form-data; name=\"files[]\"; filename=\"{path.name}\"\r\n"
              f"Content-Type: {mime}\r\n\r\n").encode("utf-8")
    body = prefix + path.read_bytes() + f"\r\n--{boundary}--\r\n".encode("utf-8")
    request = urllib.request.Request(
        "https://uguu.se/upload.php", data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}", "Content-Length": str(len(body))}, method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=600) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"temporary_upload_http_error: {exc.code} {detail}") from exc
    files = payload.get("files") if isinstance(payload.get("files"), list) else []
    url = str(files[0].get("url") or "") if files and isinstance(files[0], dict) else ""
    if not url.startswith("https://"):
        raise RuntimeError(f"temporary_upload_invalid_response: {payload}")
    return url


def _video_duration_seconds(ffmpeg: str, path: Path) -> float:
    probe = subprocess.run([ffmpeg, "-hide_banner", "-i", str(path)], capture_output=True, text=True)
    text = f"{probe.stdout}\n{probe.stderr}"
    import re
    match = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", text)
    if not match:
        raise RuntimeError(f"unable to read video duration: {path}")
    hours, minutes, seconds = match.groups()
    return int(hours) * 3600 + int(minutes) * 60 + float(seconds)


def _ffmpeg(explicit: str) -> str:
    if explicit:
        return explicit
    found = shutil.which("ffmpeg")
    if found:
        return found
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError as exc:
        raise RuntimeError("ffmpeg is required; install imageio-ffmpeg or set --ffmpeg") from exc


def _load_urls(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    raw = payload.get("videos", payload)
    return {str(key): value for key, value in raw.items() if isinstance(value, dict)} if isinstance(raw, dict) else {}


def _load_completed(path: Path) -> dict[str, dict[str, Any]]:
    completed: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return completed
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if item.get("event") == "job_completed" and item.get("job"):
            completed[str(item["job"])] = item
    return completed


def _write_results(path: Path, metadata: dict[str, Any], results: dict[str, dict[str, Any]]) -> None:
    _write_json(path, {"metadata": metadata, "results": [results[key] for key in sorted(results)]})
    fields = ["scenario", "duration_min", "method", "status", "analysis_time_seconds", "total_tokens", "analysis_windows", "keyframes", "vlm_frames", "error"]
    with path.with_suffix(".csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({name: result.get(name, "") for name in fields} for result in results.values())


def _parse_cases(values: list[str], parser: argparse.ArgumentParser) -> dict[str, str]:
    cases: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            parser.error(f"invalid --case {value!r}; expected SCENARIO=PATH")
        scenario, raw = value.split("=", 1)
        path = Path(raw.strip()).resolve()
        if not scenario.strip() or not path.is_dir():
            parser.error(f"invalid --case {value!r}")
        cases[scenario.strip()] = str(path)
    return cases


def _parse_csv(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


def _parse_csv_ints(value: str, parser: argparse.ArgumentParser) -> tuple[int, ...]:
    try:
        result = tuple(int(item) for item in _parse_csv(value))
    except ValueError:
        parser.error("--durations must contain whole minutes")
    if not result or any(item <= 0 for item in result):
        parser.error("--durations must contain positive minutes")
    return result


def _total_tokens(usage: Any) -> int:
    if not isinstance(usage, dict):
        return 0
    for name in ("total_tokens", "total_token_count"):
        if isinstance(usage.get(name), (int, float)):
            return int(usage[name])
    if isinstance(usage.get("batches"), list):
        return sum(_total_tokens(item) for item in usage["batches"])
    return int(usage.get("prompt_tokens") or 0) + int(usage.get("completion_tokens") or 0)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _append(path: Path, payload: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _gpu_names() -> list[str]:
    try:
        output = subprocess.check_output(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], text=True, stderr=subprocess.DEVNULL)
    except (OSError, subprocess.CalledProcessError):
        return []
    return [line.strip() for line in output.splitlines() if line.strip()]


def _now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


if __name__ == "__main__":
    raise SystemExit(main())
