from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .event_correlator import EventCorrelator
from .frame_analyzer import analyze_video_behavior
from .io import iso_now, load_json_records, normalize_logs
from .leak_reasoner import DatalogEngine
from .models import DetectionReport


def run_pipeline(
    log_file: str | Path,
    video_file: str | Path = "",
    output_dir: str | Path | None = None,
    sensitive_files: list[str] | None = None,
) -> dict[str, Any]:
    """Run the canonical three-stage leak detection pipeline."""

    log_path = Path(log_file)
    video_path = Path(video_file) if video_file else Path("")
    records = load_json_records(log_path)
    logs = normalize_logs(records)

    frame_bundle = analyze_video_behavior(video_path, logs=logs, sensitive_files=sensitive_files or [])
    correlation_bundle = EventCorrelator().run(
        {
            "session_id": video_path.stem or log_path.stem,
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
        report_id=f"dld_{log_path.stem}_{int(len(logs))}",
        generated_at=iso_now(),
        input={"log_file": str(log_path), "video_file": str(video_file or "")},
        summary={
            "logs": len(logs),
            "frame_observations": len(frame_bundle["observations"]),
            "correlated_events": len(correlation_bundle["correlated_events"]),
            "upload_candidates": len(correlation_bundle["upload_candidates"]),
            "datalog_facts": len(correlation_bundle["datalog_facts"]),
            "leak_paths": len(leak_paths),
        },
        event_correlator=correlation_bundle,
        frame_analyzer=frame_bundle,
        leak_reasoner={
            "leak_paths": [item.to_dict() for item in leak_paths],
            "engine": "python",
        },
        conclusion="发现数据泄露风险" if leak_paths else "未发现已确认的数据泄露",
    )

    payload = report.to_dict()
    if output_dir is not None:
        target_dir = Path(output_dir)
        target_dir.mkdir(parents=True, exist_ok=True)
        report_path = target_dir / f"{report.report_id}.json"
        report_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        payload["report_file"] = str(report_path)
    return payload
