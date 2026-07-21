"""Preload Neo4j log graphs without running the detector pipeline."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

from neo4j import GraphDatabase

from data_leak_detector.datasets import discover_data_case, discover_data_case_directories
from data_leak_detector.io import normalize_logs
from data_leak_detector.neo4j.config import Neo4jConfig
from data_leak_detector.neo4j.importer import Neo4jLogImporter
from data_leak_detector.pipeline import (
    _build_report_id,
    _composite_session_records,
    _load_pipeline_records,
    _merge_composite_records,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Preload all discovered case logs into Neo4j.")
    parser.add_argument("--case-root", default="spec/data/nas_samples", help="Root directory searched recursively for cases.")
    parser.add_argument("--output", default="artifacts/neo4j_warmup_progress.json", help="Progress JSON written after every case.")
    parser.add_argument("--limit", type=int, default=0, help="Only preload the first N cases; 0 means all cases.")
    parser.add_argument("--strict", action="store_true", help="Stop on the first import failure.")
    args = parser.parse_args(argv)

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    config = Neo4jConfig.from_env()
    cases = discover_data_case_directories(args.case_root)
    if args.limit > 0:
        cases = cases[: args.limit]
    if not cases:
        parser.error(f"no case directories found below {args.case_root}")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    state: dict[str, Any] = {
        "case_root": str(Path(args.case_root)),
        "total_cases": len(cases),
        "completed_cases": 0,
        "imported_cases": 0,
        "reused_cases": 0,
        "failed_cases": 0,
        "events": 0,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "cases": [],
    }
    importer = Neo4jLogImporter(config)
    driver = GraphDatabase.driver(config.uri, auth=(config.user, config.password), connection_timeout=20, max_connection_pool_size=1)
    started = time.perf_counter()

    def write_state() -> None:
        state["elapsed_seconds"] = round(time.perf_counter() - started, 3)
        output.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")

    try:
        driver.verify_connectivity()
        for index, case_dir in enumerate(cases, start=1):
            case_started = time.perf_counter()
            case_name = str(case_dir.relative_to(Path(args.case_root)))
            print(f"[{index}/{len(cases)}] {case_name}", flush=True)
            try:
                case = discover_data_case(case_dir)
                if len(case.sessions) > 1:
                    session_records = _composite_session_records(case.sessions, {})
                    records = _merge_composite_records(case.sessions, session_records)
                    logs = normalize_logs(records)
                else:
                    records = _load_pipeline_records(case.log_file)
                    logs = normalize_logs(records, session_start_ms=case.recording_start_ms)
                case_id = _build_report_id(case.log_file, len(records), case.case_dir.name)
                state["current_case"] = {
                    "case": case_name,
                    "sessions": len(case.sessions),
                    "events": len(logs),
                    "batches": 0,
                    "imported_events": 0,
                }
                write_state()

                def on_batch(batch_number: int, imported_events: int, total_events: int) -> None:
                    state["current_case"] = {
                        "case": case_name,
                        "events": total_events,
                        "batches": batch_number,
                        "imported_events": imported_events,
                    }
                    write_state()
                    print(f"  batch={batch_number} events={imported_events}/{total_events}", flush=True)

                with driver.session(database=config.database) as session:
                    summary = importer.ensure_import(
                        session,
                        case_id=case_id,
                        log_file=case.log_file,
                        records=records,
                        logs=logs,
                        sensitive_files=list(case.sensitive_files),
                        progress=on_batch,
                    )
                state["imported_cases"] += int(summary.imported)
                state["reused_cases"] += int(summary.reused)
                state["events"] += summary.imported_events
                state["cases"].append(
                    {
                        "case": case_name,
                        "status": "imported" if summary.imported else "reused",
                        "events": summary.imported_events,
                        "batches": summary.import_batches,
                        "seconds": round(time.perf_counter() - case_started, 3),
                    }
                )
            except Exception as exc:
                state["failed_cases"] += 1
                state["cases"].append(
                    {"case": case_name, "status": "failed", "error": f"{type(exc).__name__}: {exc}", "seconds": round(time.perf_counter() - case_started, 3)}
                )
                print(f"  failed: {type(exc).__name__}: {exc}", flush=True)
                if args.strict:
                    raise
            finally:
                state["completed_cases"] = index
                state.pop("current_case", None)
                write_state()
    finally:
        driver.close()

    print(json.dumps({key: state[key] for key in ("total_cases", "completed_cases", "imported_cases", "reused_cases", "failed_cases", "events", "elapsed_seconds")}, ensure_ascii=False), flush=True)
    return 1 if state["failed_cases"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
