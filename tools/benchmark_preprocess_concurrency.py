#!/usr/bin/env python3
"""Benchmark ScreenGuard preparation through VLM request construction only."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "main"))

from data_leak_detector.datasets import discover_data_case
from data_leak_detector.frame_analyzer import analyze_video_behavior
from data_leak_detector.frame_analyzer.config import VisionConfig
from data_leak_detector.io import normalize_logs
from data_leak_detector.log_mining import mine_analysis_windows
from data_leak_detector.pipeline import _analysis_sensitive_context, _load_pipeline_records, _vlm_file_context


DEFAULT_WORKERS = (1, 2, 4, 8, 12, 16, 20, 24, 32)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-root", default="spec/data/nas_samples")
    parser.add_argument("--case-list", default="spec/config/concurrency_benchmark_32_cases.txt")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--workers", default=",".join(map(str, DEFAULT_WORKERS)))
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup-case", default="stage0/0-normal-git-github-1")
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")

    case_root = Path(args.case_root)
    case_ids = _load_case_ids(Path(args.case_list))
    workers = _parse_workers(args.workers)
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    # The measured stage finishes immediately after the real VLM request body
    # has been assembled. No API request, parser, correlator, or reasoner runs.
    os.environ["DLD_VLM_GRID_LAYOUT"] = "4x1"
    os.environ["DLD_VLM_MAX_IMAGE_SIDE"] = "1280"
    os.environ["DLD_VLM_DRY_RUN"] = "1"

    metadata = {
        "stage": "request_preparation_only",
        "case_root": str(case_root),
        "case_count": len(case_ids),
        "case_list": str(args.case_list),
        "workers": list(workers),
        "repeats": args.repeats,
        "grid_layout": "4x1",
        "max_image_side": 1280,
        "excluded": ["VLM API transport", "VLM response parsing", "EventCorrelator", "Datalog reasoning", "debug artifact export"],
    }
    _write_json(run_dir / "metadata.json", metadata)
    _write_json(run_dir / "progress.json", {"state": "prepared", **metadata})
    print(f"Run directory: {run_dir}", flush=True)
    print("Stage: request preparation only (no VLM API or downstream reasoning)", flush=True)
    print(f"Case list: {args.case_list} ({len(case_ids)} cases)", flush=True)
    print(f"Workers: {' '.join(map(str, workers))}; repeats: {args.repeats}", flush=True)

    print(f"[{_now()}] Starting unmeasured runtime warmup case={args.warmup_case}", flush=True)
    warmup = _prepare_case(case_root, args.warmup_case, run_dir / "warmup")
    if not warmup["success"]:
        raise RuntimeError(f"warmup failed: {warmup['error']}")
    print(f"[{_now()}] Warmup complete in {warmup['seconds']:.3f}s", flush=True)

    all_runs: list[dict[str, Any]] = []
    for limit in workers:
        for repeat in range(1, args.repeats + 1):
            output = run_dir / f"workers_{limit}" / f"repeat_{repeat}"
            print(f"[{_now()}] Starting measured run workers={limit} repeat={repeat} cases={len(case_ids)}", flush=True)
            started = time.perf_counter()
            records = _run_once(case_root, case_ids, limit, output, run_dir, limit, repeat, all_runs, metadata)
            wall_seconds = time.perf_counter() - started
            successful = sum(record["success"] for record in records)
            run = {
                "workers": limit,
                "repeat": repeat,
                "case_count": len(case_ids),
                "successful_cases": successful,
                "failed_cases": len(case_ids) - successful,
                "wall_seconds": round(wall_seconds, 3),
                "case_records": records,
            }
            all_runs.append(run)
            _write_json(output / "run.json", run)
            _write_json(run_dir / "progress.json", {
                "state": "running", **metadata, "completed_runs": len(all_runs), "current": run,
            })
            print(f"[{_now()}] Completed workers={limit} repeat={repeat} success={successful}/{len(case_ids)} wall={wall_seconds:.3f}s", flush=True)

    summary = {"metadata": metadata, "rows": _summarize(all_runs, workers, args.repeats), "runs": all_runs}
    _write_json(run_dir / "summary.json", summary)
    _write_json(run_dir / "progress.json", {"state": "completed", **metadata, "rows": summary["rows"]})
    print(json.dumps({"rows": summary["rows"]}, ensure_ascii=False, indent=2), flush=True)
    return 0


def _run_once(
    case_root: Path,
    case_ids: list[str],
    limit: int,
    output: Path,
    run_dir: Path,
    current_workers: int,
    current_repeat: int,
    prior_runs: list[dict[str, Any]],
    metadata: dict[str, Any],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    output.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=min(limit, len(case_ids))) as executor:
        futures = {
            executor.submit(_prepare_case, case_root, case_id, output / "cases" / Path(case_id)): case_id
            for case_id in case_ids
        }
        for future in as_completed(futures):
            record = future.result()
            records.append(record)
            _write_json(
                run_dir / "progress.json",
                {
                    "state": "running", **metadata, "completed_runs": len(prior_runs),
                    "current": {
                        "workers": current_workers,
                        "repeat": current_repeat,
                        "completed_cases": len(records),
                        "failed_cases": sum(not item["success"] for item in records),
                        "running_cases": max(0, min(limit, len(case_ids) - len(records))),
                    },
                },
            )
    return sorted(records, key=lambda item: item["case_id"])


def _prepare_case(case_root: Path, case_id: str, output: Path) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        case = discover_data_case(case_root / Path(case_id), case_root=case_root, inherit_ancestor_groundtruth=True)
        vision = VisionConfig.from_env().with_overrides(enabled=True, max_vlm_frames=-1)
        session_stats: list[dict[str, Any]] = []
        for session in case.sessions:
            records = _load_pipeline_records(session.log_file)
            logs = normalize_logs(records, session_start_ms=session.recording_start_ms)
            analysis_sensitive, _ = _analysis_sensitive_context(
                records, logs, list(case.sensitive_files), session_start_ms=session.recording_start_ms
            )
            mining = mine_analysis_windows(
                case_id=f"{case.case_id}:{session.session_id}",
                log_file=session.log_file,
                records=records,
                logs=logs,
                sensitive_files=analysis_sensitive,
                vision_config=vision,
                neo4j_log_miner=False,
            )
            frame_bundle = analyze_video_behavior(
                session.video_file or "",
                logs=logs,
                sensitive_files=analysis_sensitive,
                vlm_sensitive_files=_vlm_file_context(logs, analysis_sensitive),
                vision_enabled=True,
                max_vlm_frames=-1,
                artifact_dir=output / "sessions" / session.session_id,
                analysis_windows=mining.windows,
                log_mining={"source": mining.source, **mining.metadata},
                debug_artifacts=False,
                export_artifacts=False,
                request_preparation_only=True,
            )
            session_stats.append(dict(frame_bundle["statistics"]["vision"]))
        return {
            "case_id": case_id,
            "success": True,
            "seconds": round(time.perf_counter() - started, 6),
            "sessions": len(session_stats),
            "raw_keyframes": sum(int(item.get("keyframes_raw_all") or 0) for item in session_stats),
            "deduplicated_keyframes": sum(int(item.get("keyframes") or 0) for item in session_stats),
            "vlm_grid_images": sum(int(item.get("vlm_frames") or 0) for item in session_stats),
            "request_batches": sum(int(item.get("vlm_batches") or 0) for item in session_stats),
            "prompt_chars": sum(int(dict(item.get("vlm_request_metrics") or {}).get("prompt_chars") or 0) for item in session_stats),
        }
    except Exception as exc:
        return {"case_id": case_id, "success": False, "seconds": round(time.perf_counter() - started, 6), "error": f"{type(exc).__name__}: {exc}"}


def _summarize(runs: list[dict[str, Any]], workers: tuple[int, ...], repeats: int) -> list[dict[str, Any]]:
    rows = []
    for limit in workers:
        selected = [item for item in runs if item["workers"] == limit]
        successful = [item for item in selected if item["failed_cases"] == 0]
        durations = sorted(item["wall_seconds"] for item in successful)
        throughput = [item["case_count"] / item["wall_seconds"] * 60 for item in successful]
        rows.append({
            "workers": limit,
            "measured_repeats": len(selected),
            "successful_repeats": len(successful),
            "successful_cases_per_repeat": successful[0]["case_count"] if successful else 0,
            "throughput_case_per_min": round(sum(throughput) / len(throughput), 3) if throughput else None,
            "batch_seconds_p50": round(_percentile(durations, 0.5), 3) if durations else None,
            "batch_seconds_p95": round(_percentile(durations, 0.95), 3) if durations else None,
        })
    return rows


def _load_case_ids(path: Path) -> list[str]:
    if not path.is_file():
        raise ValueError(f"case list not found: {path}")
    ids = [line.strip().replace("\\", "/").strip("/") for line in path.read_text(encoding="utf-8").splitlines()]
    ids = [item for item in ids if item and not item.startswith("#")]
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("case list must contain unique case IDs")
    return ids


def _parse_workers(value: str) -> tuple[int, ...]:
    try:
        workers = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise ValueError(f"invalid workers: {value}") from exc
    if not workers or any(item < 1 for item in workers) or len(workers) != len(set(workers)):
        raise ValueError(f"invalid workers: {value}")
    return workers


def _percentile(values: list[float], fraction: float) -> float:
    position = (len(values) - 1) * fraction
    lower, upper = math.floor(position), math.ceil(position)
    return values[lower] if lower == upper else values[lower] + (values[upper] - values[lower]) * (position - lower)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


if __name__ == "__main__":
    raise SystemExit(main())
