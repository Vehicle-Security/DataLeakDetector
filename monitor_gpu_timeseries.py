#!/usr/bin/env python3
"""
Collect GPU time-series data for deployment benchmark figures.

The script samples nvidia-smi at a fixed interval and writes a long-form CSV:
one row per GPU per sample. It can run in two modes:

1. Standalone duration mode:
   python monitor_gpu_timeseries.py --output gpu_trace.csv --duration 600

2. Progress-log mode:
   python monitor_gpu_timeseries.py --output gpu_trace.csv --progress-log master.log

When --progress-log is provided, the script watches benchmark progress lines like
"[c=3] [####---] 12/74" and records the latest concurrency phase in the CSV.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any


QUERY_FIELDS = [
    "index",
    "name",
    "utilization.gpu",
    "memory.used",
    "memory.total",
    "temperature.gpu",
    "power.draw",
]

CSV_FIELDS = [
    "timestamp",
    "elapsed_sec",
    "phase",
    "gpu_index",
    "gpu_name",
    "gpu_util_pct",
    "gpu_memory_used_mib",
    "gpu_memory_total_mib",
    "gpu_memory_used_pct",
    "gpu_temperature_c",
    "gpu_power_w",
]

PROGRESS_RE = re.compile(r"\[c=(\d+)\]\s+\[[#\-]+\]\s+(\d+)/(\d+)")


def parse_float(value: str) -> float:
    text = str(value or "").strip()
    if text in {"", "[N/A]", "N/A"}:
        return 0.0
    match = re.search(r"-?\d+(?:\.\d+)?", text)
    return float(match.group(0)) if match else 0.0


def read_phase(progress_log: Path | None) -> str:
    if not progress_log or not progress_log.exists():
        return "unknown"

    try:
        text = progress_log.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return "unknown"

    matches = PROGRESS_RE.findall(text)
    if not matches:
        return "startup"
    concurrency, done, total = matches[-1]
    return f"c{concurrency}_{done}_of_{total}"


def sample_nvidia_smi() -> list[dict[str, Any]]:
    cmd = [
        "nvidia-smi",
        f"--query-gpu={','.join(QUERY_FIELDS)}",
        "--format=csv,noheader,nounits",
    ]
    proc = subprocess.run(
        cmd,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or "nvidia-smi failed")

    rows: list[dict[str, Any]] = []
    for line in proc.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != len(QUERY_FIELDS):
            continue

        gpu_index = int(parse_float(parts[0]))
        gpu_name = parts[1]
        util_pct = parse_float(parts[2])
        memory_used = parse_float(parts[3])
        memory_total = parse_float(parts[4])
        temp_c = parse_float(parts[5])
        power_w = parse_float(parts[6])
        memory_pct = memory_used / memory_total * 100 if memory_total else 0.0

        rows.append({
            "gpu_index": gpu_index,
            "gpu_name": gpu_name,
            "gpu_util_pct": round(util_pct, 3),
            "gpu_memory_used_mib": round(memory_used, 3),
            "gpu_memory_total_mib": round(memory_total, 3),
            "gpu_memory_used_pct": round(memory_pct, 3),
            "gpu_temperature_c": round(temp_c, 3),
            "gpu_power_w": round(power_w, 3),
        })
    return rows


def write_summary(csv_path: Path, summary_path: Path) -> None:
    if not csv_path.exists():
        return

    by_gpu: dict[str, dict[str, Any]] = {}
    sample_count = 0
    phases: set[str] = set()

    with csv_path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            sample_count += 1
            phases.add(row.get("phase", "unknown"))
            gpu = row["gpu_index"]
            item = by_gpu.setdefault(gpu, {
                "samples": 0,
                "max_util_pct": 0.0,
                "avg_util_pct_sum": 0.0,
                "max_memory_used_mib": 0.0,
                "avg_memory_used_mib_sum": 0.0,
                "max_memory_used_pct": 0.0,
                "max_power_w": 0.0,
                "max_temperature_c": 0.0,
            })
            util = float(row["gpu_util_pct"] or 0)
            mem = float(row["gpu_memory_used_mib"] or 0)
            mem_pct = float(row["gpu_memory_used_pct"] or 0)
            power = float(row["gpu_power_w"] or 0)
            temp = float(row["gpu_temperature_c"] or 0)

            item["samples"] += 1
            item["max_util_pct"] = max(item["max_util_pct"], util)
            item["avg_util_pct_sum"] += util
            item["max_memory_used_mib"] = max(item["max_memory_used_mib"], mem)
            item["avg_memory_used_mib_sum"] += mem
            item["max_memory_used_pct"] = max(item["max_memory_used_pct"], mem_pct)
            item["max_power_w"] = max(item["max_power_w"], power)
            item["max_temperature_c"] = max(item["max_temperature_c"], temp)

    for item in by_gpu.values():
        samples = item.pop("samples") or 1
        item["avg_util_pct"] = round(item.pop("avg_util_pct_sum") / samples, 3)
        item["avg_memory_used_mib"] = round(item.pop("avg_memory_used_mib_sum") / samples, 3)
        for key, value in list(item.items()):
            if isinstance(value, float):
                item[key] = round(value, 3)

    summary = {
        "csv": str(csv_path),
        "total_rows": sample_count,
        "phases": sorted(phases),
        "gpu_summary": by_gpu,
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Collect GPU time-series data from nvidia-smi")
    parser.add_argument("--output", type=Path, required=True, help="Output CSV path")
    parser.add_argument("--interval", type=float, default=1.0, help="Sampling interval in seconds")
    parser.add_argument("--duration", type=float, default=0.0, help="Stop after this many seconds; 0 means run until interrupted")
    parser.add_argument("--progress-log", type=Path, help="Optional benchmark master log used to infer c=1/2/3/4 phase")
    parser.add_argument("--summary", type=Path, help="Optional JSON summary path; defaults to output path with .summary.json")
    parser.add_argument("--print-every", type=int, default=30, help="Print one status line every N samples; 0 disables")
    args = parser.parse_args()

    if args.interval <= 0:
        raise SystemExit("--interval must be positive")

    output = args.output.expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    summary = args.summary.expanduser() if args.summary else output.with_suffix(".summary.json")
    progress_log = args.progress_log.expanduser() if args.progress_log else None

    stop = False

    def handle_signal(signum: int, frame: Any) -> None:
        nonlocal stop
        stop = True

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    start = time.time()
    samples = 0
    print(f"writing GPU trace to {output}", flush=True)
    print(f"summary will be written to {summary}", flush=True)

    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()

        while not stop:
            now = time.time()
            elapsed = now - start
            if args.duration and elapsed > args.duration:
                break

            timestamp = datetime.now().isoformat(timespec="seconds")
            phase = read_phase(progress_log)
            try:
                gpu_rows = sample_nvidia_smi()
            except Exception as exc:
                print(f"sample failed: {exc}", file=sys.stderr, flush=True)
                time.sleep(args.interval)
                continue

            for row in gpu_rows:
                writer.writerow({
                    "timestamp": timestamp,
                    "elapsed_sec": round(elapsed, 3),
                    "phase": phase,
                    **row,
                })
            handle.flush()

            samples += 1
            if args.print_every and samples % args.print_every == 0:
                print(f"samples={samples} elapsed={elapsed:.1f}s phase={phase}", flush=True)

            sleep_for = args.interval - (time.time() - now)
            if sleep_for > 0:
                time.sleep(sleep_for)

    write_summary(output, summary)
    print(f"done: {output}", flush=True)
    print(f"summary: {summary}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
