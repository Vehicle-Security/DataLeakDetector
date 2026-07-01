"""Run NAS benchmark with sleep prevention and runtime heartbeat logging."""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import platform
import shlex
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


REPO_ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_SCRIPT = REPO_ROOT / "tools" / "benchmark_nas_samples.py"
ENV_PREFIXES = ("DLD_", "VL_", "OPENAI_", "DASHSCOPE_", "QWEN_", "LLM_")
SECRET_ENV_MARKERS = ("KEY", "TOKEN", "SECRET", "PASSWORD", "PASSWD", "AUTH")


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


def _metadata_path(json_output: Path) -> Path:
    return json_output.with_name("run_metadata.json")


def _command_path(json_output: Path) -> Path:
    return json_output.with_name("run_command.txt")


def _write_event(path: Path, event: str, **fields: Any) -> None:
    payload: Dict[str, Any] = {"timestamp": _now(), "event": event}
    payload.update(fields)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n")


def _quote_command(command: List[str]) -> str:
    if os.name == "nt":
        return subprocess.list2cmdline(command)
    return shlex.join(command)


def _redact_env_value(name: str, value: str) -> str:
    upper_name = name.upper()
    if any(marker in upper_name for marker in SECRET_ENV_MARKERS):
        if not value:
            return ""
        if len(value) <= 8:
            return "<redacted>"
        return f"{value[:4]}...{value[-4:]}"
    return value


def _relevant_environment() -> Dict[str, str]:
    items: Dict[str, str] = {}
    for name, value in sorted(os.environ.items()):
        upper_name = name.upper()
        if any(upper_name.startswith(prefix) for prefix in ENV_PREFIXES):
            items[name] = _redact_env_value(name, value)
    return items


def _benchmark_arg_config(args: List[str]) -> Dict[str, Any]:
    config: Dict[str, Any] = {}
    positional: List[str] = []
    index = 0
    while index < len(args):
        item = args[index]
        if item.startswith("--"):
            key = item[2:]
            if "=" in key:
                key, value = key.split("=", 1)
                config[key.replace("-", "_")] = value
            elif index + 1 < len(args) and not args[index + 1].startswith("--"):
                config[key.replace("-", "_")] = args[index + 1]
                index += 1
            else:
                config[key.replace("-", "_")] = True
        else:
            positional.append(item)
        index += 1
    if positional:
        config["_positional"] = positional
    return config


def _git_value(args: List[str]) -> Optional[str]:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=str(REPO_ROOT),
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except Exception:
        return None
    return result.stdout.strip()


def _git_metadata() -> Dict[str, Any]:
    status = _git_value(["status", "--short"])
    return {
        "commit": _git_value(["rev-parse", "HEAD"]),
        "branch": _git_value(["branch", "--show-current"]),
        "status_short": status.splitlines() if status else [],
    }


def _write_run_metadata(
    metadata_path: Path,
    command_path: Path,
    *,
    command: List[str],
    launcher_command: List[str],
    benchmark_args: List[str],
    json_output: Path,
    runtime_path: Path,
    parsed: argparse.Namespace,
) -> None:
    metadata = {
        "created_at": _now(),
        "repo_root": str(REPO_ROOT),
        "benchmark_script": str(BENCHMARK_SCRIPT),
        "output_dir": str(json_output.parent),
        "outputs": {
            "report_json": str(json_output),
            "report_log": str(json_output.with_suffix(".log")),
            "report_errors_json": str(json_output.with_name(f"{json_output.stem}_errors.json")),
            "runtime_jsonl": str(runtime_path),
            "run_metadata_json": str(metadata_path),
            "run_command_txt": str(command_path),
        },
        "command": {
            "argv": command,
            "line": _quote_command(command),
            "launcher_argv": launcher_command,
            "launcher_line": _quote_command(launcher_command),
            "benchmark_args": benchmark_args,
        },
        "config": {
            "guard": {
                "heartbeat_seconds": int(parsed.heartbeat_seconds),
                "output_root": str(parsed.output_root),
                "run_name": str(parsed.run_name),
            },
            "benchmark": _benchmark_arg_config(benchmark_args),
            "environment": _relevant_environment(),
        },
        "runtime": {
            "python": sys.version,
            "python_executable": sys.executable,
            "platform": platform.platform(),
            "cwd": str(Path.cwd()),
        },
        "git": _git_metadata(),
    }
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    command_text = [
        "# Guarded command",
        metadata["command"]["launcher_line"],
        "",
        "# Command",
        metadata["command"]["line"],
        "",
        "# Output directory",
        str(json_output.parent),
        "",
        "# Benchmark config",
        json.dumps(metadata["config"]["benchmark"], ensure_ascii=False, indent=2),
        "",
        "# Relevant environment (secrets redacted)",
        json.dumps(metadata["config"]["environment"], ensure_ascii=False, indent=2),
        "",
    ]
    command_path.write_text("\n".join(command_text), encoding="utf-8")


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
    launcher_args = list(sys.argv[1:] if argv is None else argv)
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
    parsed, benchmark_args = parser.parse_known_args(launcher_args)

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
    metadata_path = _metadata_path(json_output)
    command_path = _command_path(json_output)

    command = [sys.executable, str(BENCHMARK_SCRIPT), *benchmark_args]
    launcher_command = [sys.executable, str(Path(__file__).resolve()), *launcher_args]
    _write_run_metadata(
        metadata_path,
        command_path,
        command=command,
        launcher_command=launcher_command,
        benchmark_args=benchmark_args,
        json_output=json_output,
        runtime_path=runtime_path,
        parsed=parsed,
    )
    _write_event(
        runtime_path,
        "start",
        json_output=str(json_output),
        command=command,
        metadata=str(metadata_path),
        command_file=str(command_path),
    )

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
