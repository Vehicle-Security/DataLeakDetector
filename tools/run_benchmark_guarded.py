"""Run NAS benchmark with sleep prevention and runtime heartbeat logging."""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


REPO_ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_SCRIPT = REPO_ROOT / "tools" / "benchmark_nas_samples.py"


def _now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat()


def _json_output_arg(args: List[str]) -> Optional[Path]:
    for index, item in enumerate(args):
        if item == "--json-output" and index + 1 < len(args):
            return Path(args[index + 1])
        if item.startswith("--json-output="):
            return Path(item.split("=", 1)[1])
    return None


def _ensure_json_output(args: List[str], output_root: Path, run_name: str) -> tuple[List[str], Path]:
    existing = _json_output_arg(args)
    if existing is not None:
        return args, existing
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output = output_root / f"{run_name}_{timestamp}" / "report.json"
    return args + ["--json-output", str(output)], output


def _runtime_path(json_output: Path) -> Path:
    if json_output.name == "report.json":
        return json_output.with_name("runtime.jsonl")
    return json_output.with_name(f"{json_output.stem}.runtime.jsonl")


def _write_event(path: Path, event: str, **fields: Any) -> None:
    payload: Dict[str, Any] = {"timestamp": _now(), "event": event}
    payload.update(fields)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n")


class SleepGuard:
    ES_CONTINUOUS = 0x80000000
    ES_SYSTEM_REQUIRED = 0x00000001
    ES_AWAYMODE_REQUIRED = 0x00000040

    def __init__(self, runtime_path: Path) -> None:
        self.runtime_path = runtime_path
        self.enabled = False

    def __enter__(self) -> "SleepGuard":
        if os.name != "nt":
            _write_event(self.runtime_path, "sleep_guard_unavailable", platform=os.name)
            return self
        flags = self.ES_CONTINUOUS | self.ES_SYSTEM_REQUIRED | self.ES_AWAYMODE_REQUIRED
        result = ctypes.windll.kernel32.SetThreadExecutionState(flags)
        self.enabled = bool(result)
        _write_event(self.runtime_path, "sleep_guard", enabled=self.enabled, result=int(result or 0))
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        if os.name == "nt":
            ctypes.windll.kernel32.SetThreadExecutionState(self.ES_CONTINUOUS)


def _heartbeat(runtime_path: Path, stop_event: threading.Event, interval: int) -> None:
    while not stop_event.wait(interval):
        _write_event(runtime_path, "heartbeat")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run benchmark_nas_samples.py with sleep prevention and runtime logging.",
        add_help=True,
    )
    parser.add_argument(
        "--heartbeat-seconds",
        type=int,
        default=60,
        help="Heartbeat interval for runtime JSONL metadata.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPO_ROOT / "spec" / "output",
        help="Root directory for timestamped run folders when --json-output is not provided.",
    )
    parser.add_argument(
        "--run-name",
        default="nas_vlm_guarded",
        help="Prefix for the timestamped run folder when --json-output is not provided.",
    )
    parsed, benchmark_args = parser.parse_known_args(argv)

    benchmark_args = list(benchmark_args)
    if benchmark_args and benchmark_args[0] == "--":
        benchmark_args = benchmark_args[1:]
    output_root = parsed.output_root if parsed.output_root.is_absolute() else (Path.cwd() / parsed.output_root)
    output_root = output_root.resolve()
    benchmark_args, json_output = _ensure_json_output(benchmark_args, output_root, parsed.run_name)
    json_output = json_output if json_output.is_absolute() else (Path.cwd() / json_output)
    json_output = json_output.resolve()
    json_output.parent.mkdir(parents=True, exist_ok=True)
    runtime_path = _runtime_path(json_output)

    command = [sys.executable, str(BENCHMARK_SCRIPT), *benchmark_args]
    _write_event(runtime_path, "start", json_output=str(json_output), command=command)

    stop_event = threading.Event()
    heartbeat = threading.Thread(
        target=_heartbeat,
        args=(runtime_path, stop_event, max(5, int(parsed.heartbeat_seconds))),
        daemon=True,
    )
    started = time.monotonic()
    exit_code = 1
    try:
        heartbeat.start()
        with SleepGuard(runtime_path):
            process = subprocess.run(command, cwd=str(REPO_ROOT))
            exit_code = int(process.returncode)
            return exit_code
    finally:
        elapsed = time.monotonic() - started
        stop_event.set()
        heartbeat.join(timeout=5)
        _write_event(
            runtime_path,
            "end",
            exit_code=exit_code,
            elapsed_seconds=round(elapsed, 3),
            elapsed_hms=time.strftime("%H:%M:%S", time.gmtime(elapsed)),
        )
        print(f"Runtime metadata: {runtime_path}")


if __name__ == "__main__":
    raise SystemExit(main())
