#!/usr/bin/env python3
"""
Benchmark log-driven E2E runs from logs + video.

This script measures deployment cost and throughput. It does not require
groundtruth and does not score accuracy.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
import re
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any


VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi"}


def slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")[:180] or "case"


def parse_levels(value: str) -> list[int]:
    levels = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        levels.append(int(part))
    if not levels or any(level < 1 for level in levels):
        raise argparse.ArgumentTypeError("concurrency must contain positive integers")
    return levels


def find_video(case_dir: Path) -> Path | None:
    video_dir = case_dir / "video"
    if not video_dir.exists():
        return None
    videos = [
        path for path in sorted(video_dir.iterdir())
        if path.is_file() and path.suffix.casefold() in VIDEO_EXTENSIONS
    ]
    return videos[0] if videos else None


def discover_cases(dataset_root: Path, cases_file: Path | None = None) -> list[Path]:
    if cases_file:
        cases = []
        for raw in cases_file.read_text(encoding="utf-8").splitlines():
            raw = raw.strip()
            if not raw or raw.startswith("#"):
                continue
            cases.append(Path(raw).expanduser())
    else:
        seen = set()
        cases = []
        for log_path in sorted(dataset_root.rglob("logs/logs.json")):
            case_dir = log_path.parent.parent
            key = str(case_dir.resolve())
            if key in seen:
                continue
            seen.add(key)
            cases.append(case_dir)

    valid = []
    for case_dir in cases:
        if (case_dir / "logs" / "logs.json").exists() and find_video(case_dir):
            valid.append(case_dir)
    return valid


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * pct)))
    return ordered[index]


def extract_report_path(text: str) -> str:
    matches = re.findall(r"(/\S*full_evidence_\S+?\.json)", text)
    return matches[-1] if matches else ""


def parse_case_log(text: str, report_path: str) -> dict[str, Any]:
    metrics: dict[str, Any] = {
        "pipeline_completed": "Full E2E Pipeline" in text,
        "module_errors": text.count("处理事件失败"),
        "api_key_errors": text.count("api_key client option"),
        "worklist_events": "",
        "module3_alert_events": "",
        "module3_upload_events": "",
        "module4_leak_paths": "",
        "module4_facts": "",
        "evidence_logs": "",
        "total_logs": "",
        "llm_status": "",
        "risk_found": "",
        "report": report_path,
    }

    worklist_match = re.search(r"worklist:\s*.*?(\d+)", text, flags=re.IGNORECASE)
    if worklist_match:
        metrics["worklist_events"] = int(worklist_match.group(1))

    if report_path and Path(report_path).exists():
        try:
            report = json.loads(Path(report_path).read_text(encoding="utf-8"))
            summary = report.get("summary", {})
            module4 = report.get("module4_threat_detector", {})
            stats = module4.get("stats", {})
            metrics.update({
                "module3_alert_events": summary.get("module3_alert_events", ""),
                "module3_upload_events": summary.get("module3_upload_events", ""),
                "module4_leak_paths": summary.get("module4_leak_paths", ""),
                "module4_facts": summary.get("module4_datalog_facts", ""),
                "evidence_logs": stats.get("evidence_logs", ""),
                "total_logs": stats.get("total_logs", ""),
                "llm_status": stats.get("llm_status", "disabled" if not stats.get("llm_enabled") else ""),
            })
            metrics["risk_found"] = bool(
                int(summary.get("module3_alert_events", 0) or 0)
                or int(summary.get("module4_leak_paths", 0) or 0)
            )
        except Exception as exc:
            metrics["report_error"] = str(exc)

    return metrics


def classify_failure(rc: int, timed_out: bool, text: str) -> str:
    lowered = text.casefold()
    if timed_out or rc == 124:
        return "timeout"
    if "cuda out of memory" in lowered or "cudaerror memoryallocation" in lowered:
        return "cuda_oom"
    if "traceback (most recent call last)" in lowered:
        return "traceback"
    if "bad request" in lowered:
        return "bad_request"
    if rc != 0:
        return "nonzero_rc"
    return ""


async def sample_gpus(stop: asyncio.Event, interval: float, samples: list[dict[str, int]]) -> None:
    if interval <= 0:
        return
    while not stop.is_set():
        try:
            proc = await asyncio.create_subprocess_exec(
                "nvidia-smi",
                "--query-gpu=index,memory.used,utilization.gpu",
                "--format=csv,noheader,nounits",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.DEVNULL,
            )
            stdout, _ = await proc.communicate()
            if proc.returncode == 0:
                for raw_line in stdout.decode("utf-8", errors="ignore").splitlines():
                    parts = [part.strip() for part in raw_line.split(",")]
                    if len(parts) != 3:
                        continue
                    try:
                        samples.append({
                            "gpu": int(parts[0]),
                            "memory_mib": int(parts[1]),
                            "util_pct": int(parts[2]),
                        })
                    except ValueError:
                        continue
        except FileNotFoundError:
            return
        try:
            await asyncio.wait_for(stop.wait(), timeout=interval)
        except asyncio.TimeoutError:
            pass


def summarize_gpu_samples(samples: list[dict[str, int]]) -> dict[str, Any]:
    if not samples:
        return {}
    by_gpu: dict[int, dict[str, int]] = {}
    for sample in samples:
        gpu = sample["gpu"]
        item = by_gpu.setdefault(gpu, {"max_memory_mib": 0, "max_util_pct": 0})
        item["max_memory_mib"] = max(item["max_memory_mib"], sample["memory_mib"])
        item["max_util_pct"] = max(item["max_util_pct"], sample["util_pct"])
    return {
        "gpu_max": {str(gpu): data for gpu, data in sorted(by_gpu.items())},
        "cluster_max_memory_mib": max(item["max_memory_mib"] for item in by_gpu.values()),
        "cluster_max_util_pct": max(item["max_util_pct"] for item in by_gpu.values()),
    }


async def run_level(cases: list[Path], concurrency: int, args: argparse.Namespace, run_dir: Path) -> dict[str, Any]:
    level_dir = run_dir / f"concurrency_{concurrency}"
    logs_dir = level_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    gpus = [item.strip() for item in (args.ocr_gpus or "").split(",") if item.strip()]
    queue: asyncio.Queue[tuple[int, Path]] = asyncio.Queue()
    for index, case_dir in enumerate(cases):
        queue.put_nowait((index, case_dir))

    rows: list[dict[str, Any]] = []
    gpu_samples: list[dict[str, int]] = []
    stop_sampler = asyncio.Event()
    sampler_task = asyncio.create_task(sample_gpus(stop_sampler, args.gpu_sample_interval, gpu_samples))

    start = time.time()
    total = len(cases)
    completed = 0

    def print_progress() -> None:
        width = 30
        filled = int(completed * width / total) if total else width
        bar = "#" * filled + "-" * (width - filled)
        print(f"\r[c={concurrency}] [{bar}] {completed}/{total}", end="", flush=True)

    async def run_one(index: int, case_dir: Path, slot_id: int) -> dict[str, Any]:
        name = slug(str(case_dir.relative_to(args.dataset_root)))
        case_log = logs_dir / f"{index + 1:04d}_{name}.log"
        log_file = case_dir / "logs" / "logs.json"
        video_file = find_video(case_dir)

        env = os.environ.copy()
        openai_api_key = args.openai_api_key or env.get("OPENAI_API_KEY") or "EMPTY"
        openai_base_url = args.openai_base_url or env.get("OPENAI_BASE_URL") or "http://127.0.0.1:8000/v1"
        model_name = (
            args.model_name
            or env.get("VL_MODEL_NAME")
            or env.get("MODEL_NAME")
            or "qwen2.5-vl-72b"
        )
        env.update({
            "OPENAI_API_KEY": openai_api_key,
            "OPENAI_BASE_URL": openai_base_url,
            "VL_MODEL_NAME": model_name,
            "MODEL_NAME": model_name,
            "LLM_API_KEY": env.get("LLM_API_KEY") or openai_api_key,
            "LLM_BASE_URL": env.get("LLM_BASE_URL") or openai_base_url,
            "LLM_MODEL_NAME": env.get("LLM_MODEL_NAME") or model_name,
            "DLD_THREAT_USE_LLM": args.threat_use_llm,
            "DLD_THREAT_MAX_LOGS": str(args.threat_max_logs),
            "DLD_THREAT_LOG_WINDOW_SECONDS": str(args.threat_window_seconds),
            "FRAME_ANALYZER_SAMPLE_FPS": str(args.sample_fps),
            "FRAME_ANALYZER_MAX_VLM_IMAGES": str(args.max_images),
            "FRAME_ANALYZER_FALLBACK_VLM_IMAGES": str(args.fallback_images),
            "FRAME_ANALYZER_VLM_MAX_SIDE": str(args.max_side),
            "FRAME_ANALYZER_ALLOW_VLM_FALLBACK": str(args.allow_vlm_fallback).lower(),
            "OMP_NUM_THREADS": str(args.omp_threads),
            "MKL_NUM_THREADS": str(args.mkl_threads),
            "TOKENIZERS_PARALLELISM": "false",
        })
        if gpus:
            env["CUDA_VISIBLE_DEVICES"] = gpus[slot_id % len(gpus)]

        cmd = [
            sys.executable,
            "-u",
            str(Path(__file__).with_name("run_e2e.py")),
            "--log",
            str(log_file),
            "--video",
            str(video_file),
        ]

        t0 = time.time()
        timed_out = False
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            env=env,
            cwd=str(Path(__file__).resolve().parent),
        )
        try:
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=args.timeout)
            rc = proc.returncode
        except asyncio.TimeoutError:
            timed_out = True
            proc.kill()
            stdout, _ = await proc.communicate()
            rc = 124

        wall_sec = time.time() - t0
        text = stdout.decode("utf-8", errors="replace")
        case_log.write_text(text, encoding="utf-8")

        report_path = extract_report_path(text)
        metrics = parse_case_log(text, report_path)
        failure = classify_failure(rc, timed_out, text)

        status = "completed" if rc == 0 and metrics.get("pipeline_completed") else "failed"
        if failure and status != "completed":
            status = failure

        row: dict[str, Any] = {
            "index": index + 1,
            "case": str(case_dir),
            "rc": rc,
            "status": status,
            "wall_sec": round(wall_sec, 3),
            "slot": slot_id,
            "ocr_gpu": env.get("CUDA_VISIBLE_DEVICES", ""),
            "log": str(case_log),
        }
        row.update(metrics)
        return row

    async def worker(slot_id: int) -> None:
        nonlocal completed
        while True:
            try:
                index, case_dir = queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            try:
                row = await run_one(index, case_dir, slot_id)
                rows.append(row)
            finally:
                completed += 1
                print_progress()
                queue.task_done()

    print_progress()
    await asyncio.gather(*(worker(slot_id) for slot_id in range(concurrency)))
    print()

    stop_sampler.set()
    await sampler_task
    elapsed = time.time() - start

    rows.sort(key=lambda item: int(item["index"]))
    if rows:
        with (level_dir / "jobs.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    walls = [float(row["wall_sec"]) for row in rows if row.get("wall_sec")]
    completed_rows = [row for row in rows if row["status"] == "completed"]
    risk_rows = [row for row in completed_rows if str(row.get("risk_found", "")).casefold() == "true"]
    summary: dict[str, Any] = {
        "concurrency": concurrency,
        "cases": len(rows),
        "elapsed_sec": elapsed,
        "samples_per_min": len(rows) / elapsed * 60 if elapsed else 0,
        "completed": len(completed_rows),
        "risk_found": len(risk_rows),
        "failed": sum(1 for row in rows if row["status"] != "completed"),
        "timeouts": sum(1 for row in rows if row["status"] == "timeout"),
        "cuda_oom": sum(1 for row in rows if row["status"] == "cuda_oom"),
        "mean_wall_sec": statistics.mean(walls) if walls else 0,
        "p50_wall_sec": percentile(walls, 0.50),
        "p95_wall_sec": percentile(walls, 0.95),
        "total_worklist_events": sum(int(row.get("worklist_events") or 0) for row in rows),
        "total_alert_events": sum(int(row.get("module3_alert_events") or 0) for row in rows),
        "total_leak_paths": sum(int(row.get("module4_leak_paths") or 0) for row in rows),
        "module_errors": sum(int(row.get("module_errors") or 0) for row in rows),
        "api_key_errors": sum(int(row.get("api_key_errors") or 0) for row in rows),
        "run_dir": str(run_dir),
    }
    summary.update(summarize_gpu_samples(gpu_samples))

    (level_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary


async def async_main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark log-driven E2E runs")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--cases-file", type=Path)
    parser.add_argument("--output-root", type=Path, default=Path("~/logs/log_driven_e2e_bench").expanduser())
    parser.add_argument("--concurrency", type=parse_levels, default=[1])
    parser.add_argument("--ocr-gpus", default="")
    parser.add_argument("--timeout", type=float, default=1800)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--sample-fps", type=float, default=1.0)
    parser.add_argument("--max-images", type=int, default=3)
    parser.add_argument("--fallback-images", type=int, default=3)
    parser.add_argument("--max-side", type=int, default=560)
    parser.add_argument("--allow-vlm-fallback", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--threat-use-llm", default="false")
    parser.add_argument("--openai-api-key", default="")
    parser.add_argument("--openai-base-url", default="")
    parser.add_argument("--model-name", default="")
    parser.add_argument("--threat-max-logs", type=int, default=80)
    parser.add_argument("--threat-window-seconds", type=int, default=90)
    parser.add_argument("--omp-threads", type=int, default=2)
    parser.add_argument("--mkl-threads", type=int, default=2)
    parser.add_argument("--gpu-sample-interval", type=float, default=5.0)
    args = parser.parse_args()

    args.dataset_root = args.dataset_root.expanduser().resolve()
    args.output_root = args.output_root.expanduser()
    args.output_root.mkdir(parents=True, exist_ok=True)

    cases = discover_cases(args.dataset_root, args.cases_file)
    if args.limit:
        cases = cases[:args.limit]
    if not cases:
        raise SystemExit("No cases with logs/logs.json and video found")

    run_dir = args.output_root / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "cases.txt").write_text("\n".join(str(case) for case in cases), encoding="utf-8")

    print(f"cases={len(cases)}")
    print(f"run_dir={run_dir}")

    summaries = []
    for concurrency in args.concurrency:
        summaries.append(await run_level(cases, concurrency, args, run_dir))

    (run_dir / "summary_all.json").write_text(
        json.dumps(summaries, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print()
    print("concurrency\tcases\tcompleted\tfailed\ttimeouts\tcuda_oom\tsamples/min\tmean_wall\tp95_wall\trisk_found\tmodule_errors\tapi_key_errors\trun_dir")
    for item in summaries:
        print(
            f"{item['concurrency']}\t{item['cases']}\t{item['completed']}\t{item['failed']}\t"
            f"{item['timeouts']}\t{item['cuda_oom']}\t{item['samples_per_min']:.3f}\t"
            f"{item['mean_wall_sec']:.3f}\t{item['p95_wall_sec']:.3f}\t{item['risk_found']}\t"
            f"{item['module_errors']}\t{item['api_key_errors']}\t{item['run_dir']}"
        )


def main() -> None:
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
