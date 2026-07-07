"""End-to-end orchestration for the single canonical detection pipeline.

This module is the only place that wires FrameAnalyzer, EventCorrelator,
LeakReasoner, report serialization, and optional Neo4j persistence together.
Keeping orchestration here makes each stage independently testable and prevents
the old three-directory structure from returning as hidden compatibility code.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .event_correlator import EventCorrelator
from .frame_analyzer import analyze_video_behavior
from .graph import Neo4jConfig, write_report_to_neo4j
from .io import iso_now, load_json_records, normalize_logs
from .leak_reasoner import DatalogEngine
from .models import DetectionReport


def run_pipeline(
    log_file: str | Path,
    video_file: str | Path = "",
    output_dir: str | Path | None = None,
    sensitive_files: list[str] | None = None,
    observations_file: str | Path | None = None,
    neo4j_enabled: bool | None = None,
    neo4j_strict: bool | None = None,
) -> dict[str, Any]:
    """Run the canonical FrameAnalyzer -> EventCorrelator -> LeakReasoner flow."""

    log_path = Path(log_file)
    video_text = str(video_file or "")
    video_path = Path(video_text) if video_text else None
    records = load_json_records(log_path)
    logs = normalize_logs(records)

    frame_bundle = analyze_video_behavior(
        video_path or "",
        logs=logs,
        sensitive_files=sensitive_files or [],
        observations_file=observations_file,
    )
    correlation_bundle = EventCorrelator().run(
        {
            "session_id": video_path.stem if video_path else log_path.stem,
            "log_events": records,
            "frame_segments": frame_bundle["observations"],
            "sensitive_files": sensitive_files or [],
        }
    )

    engine = DatalogEngine()
    for fact in correlation_bundle["datalog_facts"]:
        engine.add_fact(fact["relation"], *fact["args"])
    leak_paths = engine.query_leak()

    report = DetectionReport(
        report_id=f"dld_{log_path.stem}_{len(records)}",
        generated_at=iso_now(),
        input={"log_file": str(log_path), "video_file": video_text},
        summary={
            "logs": len(logs),
            "frame_observations": len(frame_bundle["observations"]),
            "correlated_events": len(correlation_bundle["correlated_events"]),
            "upload_candidates": len(correlation_bundle["upload_candidates"]),
            "datalog_facts": len(correlation_bundle["datalog_facts"]),
            "leak_paths": len(leak_paths),
        },
        frame_analyzer=frame_bundle,
        event_correlator=correlation_bundle,
        leak_reasoner={
            "engine": "python_taint",
            "leak_paths": [item.to_dict() for item in leak_paths],
        },
        conclusion="发现数据泄露风险" if leak_paths else "未发现已确认的数据泄露",
    )
    payload = report.to_dict()
    payload["event_correlator"]["raw_log_events"] = records
    payload["graph"] = _write_graph(payload, neo4j_enabled=neo4j_enabled, neo4j_strict=neo4j_strict)

    if output_dir is not None:
        target_dir = Path(output_dir)
        target_dir.mkdir(parents=True, exist_ok=True)
        report_path = target_dir / f"{report.report_id}.json"
        report_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        payload["report_file"] = str(report_path)
    return payload


def _write_graph(
    payload: dict[str, Any],
    *,
    neo4j_enabled: bool | None,
    neo4j_strict: bool | None,
) -> dict[str, Any]:
    config = Neo4jConfig.from_env()
    if neo4j_enabled is not None:
        config = Neo4jConfig(
            enabled=neo4j_enabled,
            uri=config.uri,
            user=config.user,
            password=config.password,
            database=config.database,
            strict=config.strict,
            clear_session=config.clear_session,
        )
    if neo4j_strict is not None:
        config = Neo4jConfig(
            enabled=config.enabled,
            uri=config.uri,
            user=config.user,
            password=config.password,
            database=config.database,
            strict=neo4j_strict,
            clear_session=config.clear_session,
        )

    try:
        return write_report_to_neo4j(payload, config)
    except Exception as exc:
        if config.strict:
            raise
        return {
            "enabled": config.enabled,
            "status": "error",
            "uri": config.uri,
            "database": config.database,
            "error": f"{type(exc).__name__}: {exc}",
        }
