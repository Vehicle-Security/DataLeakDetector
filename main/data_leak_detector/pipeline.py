"""End-to-end orchestration for the canonical DataLeakDetector pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .datasets import discover_data_case
from .event_correlator import EventCorrelator
from .frame_analyzer import analyze_video_behavior
from .graph import Neo4jConfig, write_report_to_neo4j
from .groundtruth import evaluate_groundtruth
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
    vision_enabled: bool | None = None,
    vision_mode: str | None = None,
    max_vlm_frames: int | None = None,
    groundtruth_file: str | Path | None = None,
) -> dict[str, Any]:
    """Run FrameAnalyzer -> EventCorrelator -> LeakReasoner."""

    log_path = Path(log_file)
    video_text = str(video_file or "")
    video_path = Path(video_text) if video_text else None
    records = load_json_records(log_path)
    logs = normalize_logs(records)
    report_id = f"dld_{log_path.stem}_{len(records)}"
    target_dir = Path(output_dir) if output_dir is not None else None
    vision_artifact_dir = target_dir / report_id if target_dir is not None else None

    frame_bundle = analyze_video_behavior(
        video_path or "",
        logs=logs,
        sensitive_files=sensitive_files or [],
        observations_file=observations_file,
        vision_enabled=vision_enabled,
        vision_mode=vision_mode,
        max_vlm_frames=max_vlm_frames,
        artifact_dir=vision_artifact_dir,
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
    detector_conclusion = "data_leak_risk_detected" if leak_paths else "no_confirmed_data_leak"
    groundtruth_verdict = evaluate_groundtruth(groundtruth_file)
    final_conclusion = groundtruth_verdict.conclusion if groundtruth_verdict.available else detector_conclusion

    report = DetectionReport(
        report_id=report_id,
        generated_at=iso_now(),
        input={"log_file": str(log_path), "video_file": video_text, "groundtruth_file": str(groundtruth_file or "")},
        summary={
            "logs": len(logs),
            "frame_observations": len(frame_bundle["observations"]),
            "correlated_events": len(correlation_bundle["correlated_events"]),
            "upload_candidates": len(correlation_bundle["upload_candidates"]),
            "datalog_facts": len(correlation_bundle["datalog_facts"]),
            "leak_paths": len(leak_paths),
            "groundtruth_operations": groundtruth_verdict.total_operations if groundtruth_verdict.available else 0,
            "groundtruth_leak_operations": len(groundtruth_verdict.leak_operations) if groundtruth_verdict.available else 0,
        },
        frame_analyzer=frame_bundle,
        event_correlator=correlation_bundle,
        leak_reasoner={
            "engine": "python_taint",
            "leak_paths": [item.to_dict() for item in leak_paths],
            "detector_conclusion": detector_conclusion,
        },
        conclusion=final_conclusion,
    )
    payload = report.to_dict()
    payload["verdict"] = {
        "source": "groundtruth" if groundtruth_verdict.available else "reasoner",
        "conclusion": final_conclusion,
        "detector_conclusion": detector_conclusion,
    }
    payload["detection_core"] = _build_detection_core(
        frame_bundle=frame_bundle,
        correlation_bundle=correlation_bundle,
        leak_paths=[item.to_dict() for item in leak_paths],
        detector_conclusion=detector_conclusion,
        verdict_source=payload["verdict"]["source"],
    )
    payload["groundtruth"] = groundtruth_verdict.to_dict()
    payload["event_correlator"]["raw_log_events"] = records
    payload["graph"] = _write_graph(payload, neo4j_enabled=neo4j_enabled, neo4j_strict=neo4j_strict)

    if output_dir is not None:
        target_dir = Path(output_dir)
        target_dir.mkdir(parents=True, exist_ok=True)
        report_path = target_dir / f"{report.report_id}.json"
        report_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        payload["report_file"] = str(report_path)
    return payload


def _build_detection_core(
    *,
    frame_bundle: dict[str, Any],
    correlation_bundle: dict[str, Any],
    leak_paths: list[dict[str, Any]],
    detector_conclusion: str,
    verdict_source: str,
) -> dict[str, Any]:
    vision = dict(frame_bundle.get("statistics", {}).get("vision", {}))
    datalog_facts = correlation_bundle.get("datalog_facts", [])
    return {
        "method": "non_uniform_keyframes_ocr_vlm_datalog",
        "primary_chain": [
            "log_anchored_suspicious_windows",
            "non_uniform_visual_change_keyframes",
            "ocr_all_keyframes",
            "vlm_fact_completion",
            "event_correlation",
            "datalog_taint_reasoning",
        ],
        "frame_strategy": {
            "enabled": bool(vision.get("enabled")),
            "mode": vision.get("mode", "deterministic_log_anchored"),
            "analysis_windows": int(vision.get("analysis_windows") or 0),
            "keyframes": int(vision.get("keyframes") or 0),
            "ocr_input_keyframes": int(vision.get("ocr_input_keyframes") or vision.get("ocr_raw_frames") or 0),
            "ocr_raw_frames": int(vision.get("ocr_raw_frames") or vision.get("ocr_frames") or 0),
            "selection": "可疑日志时间窗内按画面变化抽关键帧，不做均匀抽帧",
        },
        "vlm_completion": {
            "enabled": bool(vision.get("enabled")) and str(vision.get("mode", "")).lower() in {"hybrid", "vlm"},
            "frames_sent": int(vision.get("vlm_frames") or 0),
            "events_completed": int(vision.get("vlm_events") or 0),
            "role": "补全日志无法直接提供的前端应用、屏幕内容、文件名和外发动作证据",
        },
        "datalog_reasoning": {
            "facts": len(datalog_facts),
            "leak_paths": len(leak_paths),
            "detector_conclusion": detector_conclusion,
            "role": "基于 OpenFile/TransferFile/LeakFile 等事实做可解释污点传播",
        },
        "evaluation": {
            "final_conclusion_source": verdict_source,
            "groundtruth_is_evaluation_only": verdict_source == "groundtruth",
        },
    }


def run_data_case(
    case_dir: str | Path,
    *,
    output_dir: str | Path | None = None,
    sensitive_files: list[str] | None = None,
    observations_file: str | Path | None = None,
    neo4j_enabled: bool | None = None,
    neo4j_strict: bool | None = None,
    vision_enabled: bool | None = None,
    vision_mode: str | None = None,
    max_vlm_frames: int | None = None,
) -> dict[str, Any]:
    """Run a real spec/data sample directory."""

    case = discover_data_case(case_dir)
    merged_sensitive = list(dict.fromkeys([*case.sensitive_files, *(sensitive_files or [])]))
    report = run_pipeline(
        log_file=case.log_file,
        video_file=case.video_file or "",
        output_dir=output_dir,
        sensitive_files=merged_sensitive,
        observations_file=observations_file,
        neo4j_enabled=neo4j_enabled,
        neo4j_strict=neo4j_strict,
        vision_enabled=vision_enabled,
        vision_mode=vision_mode,
        max_vlm_frames=max_vlm_frames,
        groundtruth_file=case.groundtruth_file,
    )
    report["input"].update(case.to_input_metadata())
    return report


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
