"""End-to-end orchestration for the canonical DataLeakDetector pipeline."""

from __future__ import annotations

import json
import os
from pathlib import Path
import re
import shutil
from typing import Any

from .datasets import DataCase, DataSession, discover_data_case
from .event_correlator import EventCorrelator
from .frame_analyzer import analyze_video_behavior
from .frame_analyzer.config import VisionConfig
from .log_mining import mine_analysis_windows
from .groundtruth import evaluate_groundtruth
from .io import iso_now, load_json_records, normalize_logs, normalize_path, parse_timestamp_ms, same_file
from .leak_reasoner import DatalogEngine
from .models import DetectionReport
from .sensitivity import load_sensitive_files_config


def run_pipeline(
    log_file: str | Path,
    video_file: str | Path = "",
    output_dir: str | Path | None = None,
    detail_output_dir: str | Path | None = None,
    sensitive_files_config: str | Path | None = None,
    observations_file: str | Path | None = None,
    vision_enabled: bool | None = None,
    max_vlm_frames: int | None = None,
    vision_precompute_file: str | Path | None = None,
    precomputed_baseline_file: str | Path | None = None,
    groundtruth_file: str | Path | None = None,
    neo4j_log_miner: bool | None = None,
    reuse_neo4j_import: bool | None = None,
    non_vlm_enabled: bool | None = None,
    vision_debug_artifacts: bool = True,
    inherit_ancestor_groundtruth: bool = False,
    case_name: str | None = None,
    session_start_ms: int | None = None,
    case_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run FrameAnalyzer -> EventCorrelator -> LeakReasoner."""

    log_path = Path(log_file)
    video_text = str(video_file or "")
    video_path = Path(video_text) if video_text else None
    baseline = _load_precomputed_baseline(precomputed_baseline_file)
    if baseline and vision_precompute_file is None:
        vision_precompute_file = str(baseline.get("vision_precompute_file") or "") or None
    records = (
        _merge_baseline_records(log_path, [item for item in baseline.get("records", []) if isinstance(item, dict)])
        if baseline
        else _load_pipeline_records(log_path)
    )
    logs = [] if baseline else normalize_logs(records, session_start_ms=session_start_ms)
    initial_sensitive_files = _dedupe_paths(list(load_sensitive_files_config(sensitive_files_config)))
    analysis_sensitive_files, derived_sensitive_context = _analysis_sensitive_context(
        records,
        logs,
        initial_sensitive_files,
        session_start_ms=session_start_ms,
    )
    report_id = _build_report_id(log_path, len(records), case_name)
    target_dir = Path(output_dir) if output_dir is not None else None
    vision_artifact_dir = target_dir / report_id if target_dir is not None else None
    vision_config = VisionConfig.from_env().with_overrides(
        enabled=vision_enabled,
        max_vlm_frames=max_vlm_frames,
    )
    effective_neo4j_log_miner = neo4j_log_miner
    if effective_neo4j_log_miner is None and not vision_config.enabled:
        effective_neo4j_log_miner = False
    log_mining = mine_analysis_windows(
        case_id=report_id,
        log_file=log_path,
        records=records,
        logs=logs,
        sensitive_files=analysis_sensitive_files,
        vision_config=vision_config,
        neo4j_log_miner=effective_neo4j_log_miner,
        reuse_import=reuse_neo4j_import,
    ) if not baseline else None
    context_logs = logs or normalize_logs(
        [item for item in records if isinstance(item, dict)],
        session_start_ms=session_start_ms,
    )
    vlm_sensitive_files = _vlm_file_context(context_logs, analysis_sensitive_files)

    frame_bundle = analyze_video_behavior(
        video_path or "",
        logs=logs,
        sensitive_files=analysis_sensitive_files,
        vlm_sensitive_files=vlm_sensitive_files,
        observations_file=observations_file,
        vision_enabled=vision_enabled,
        max_vlm_frames=max_vlm_frames,
        vision_precompute_file=vision_precompute_file,
        artifact_dir=vision_artifact_dir,
        analysis_windows=log_mining.windows if log_mining else None,
        log_mining={"source": log_mining.source, **log_mining.metadata} if log_mining else {"source": "precomputed_baseline"},
        debug_artifacts=vision_debug_artifacts,
    )
    use_non_vlm = True if non_vlm_enabled is None else bool(non_vlm_enabled)
    if baseline and use_non_vlm:
        frame_bundle["observations"] = [*baseline.get("log_observations", []), *frame_bundle["observations"]]
    return _finalize_pipeline(
        report_id=report_id,
        case_id=str((case_metadata or {}).get("case_id") or report_id),
        session_id=video_path.stem if video_path else log_path.stem,
        log_path=log_path,
        video_text=video_text,
        records=records,
        logs=logs,
        frame_bundle=frame_bundle,
        initial_sensitive_files=initial_sensitive_files,
        derived_sensitive_context=derived_sensitive_context,
        analysis_sensitive_files=analysis_sensitive_files,
        vlm_sensitive_files=vlm_sensitive_files,
        sensitive_files_config=sensitive_files_config,
        groundtruth_file=groundtruth_file,
        non_vlm_enabled=use_non_vlm,
        recording_start_ms=int(session_start_ms or 0),
        input_recording_start_ms=None,
        case_metadata=case_metadata,
        log_mining_payload={"source": log_mining.source, **log_mining.metadata}
        if log_mining
        else {"source": "precomputed_baseline"},
        output_dir=output_dir,
        detail_output_dir=detail_output_dir,
    )


def _finalize_pipeline(
    *,
    report_id: str,
    case_id: str,
    session_id: str,
    log_path: Path,
    video_text: str,
    records: list[dict[str, Any]],
    logs: list[Any],
    frame_bundle: dict[str, Any],
    initial_sensitive_files: list[str],
    derived_sensitive_context: list[str],
    analysis_sensitive_files: list[str],
    vlm_sensitive_files: list[str],
    sensitive_files_config: str | Path | None,
    groundtruth_file: str | Path | None,
    non_vlm_enabled: bool,
    recording_start_ms: int,
    input_recording_start_ms: int | None,
    case_metadata: dict[str, Any] | None,
    log_mining_payload: dict[str, Any],
    output_dir: str | Path | None,
    detail_output_dir: str | Path | None,
) -> dict[str, Any]:
    correlation_bundle = EventCorrelator().run(
        {
            "case_id": case_id,
            "session_id": session_id,
            "log_events": records,
            "frame_segments": frame_bundle["observations"],
            "sensitive_files": initial_sensitive_files,
            "recording_start_ms": recording_start_ms,
            "non_vlm_enabled": non_vlm_enabled,
        }
    )

    engine = DatalogEngine(case_id=case_id)
    for fact in correlation_bundle["datalog_facts"]:
        engine.add_fact(fact["relation"], *fact["args"], case_id=fact.get("case_id"))
    leak_paths = engine.query_leak()
    suspicious_facts = _suspicious_datalog_facts(correlation_bundle)
    detector_conclusion = _detector_conclusion(leak_paths, suspicious_facts)
    groundtruth_verdict = evaluate_groundtruth(groundtruth_file)

    displayed_recording_start_ms = recording_start_ms if input_recording_start_ms is None else input_recording_start_ms
    input_metadata = {
        "case_id": case_id,
        "log_file": str(log_path),
        "video_file": video_text,
        "groundtruth_file": str(groundtruth_file or ""),
        "sensitive_files_config": str(sensitive_files_config or ""),
        "recording_start_ms": displayed_recording_start_ms,
        "sensitive_source_files": initial_sensitive_files,
        "derived_sensitive_context": derived_sensitive_context,
        "analysis_sensitive_files": analysis_sensitive_files,
        "vlm_sensitive_files": vlm_sensitive_files,
    }
    if case_metadata:
        input_metadata.update(case_metadata)
        input_metadata.update(
            {
                "log_file": str(log_path),
                "video_file": video_text,
                "groundtruth_file": str(groundtruth_file or ""),
                "sensitive_files_config": str(sensitive_files_config or ""),
                "recording_start_ms": displayed_recording_start_ms,
                "sensitive_source_files": initial_sensitive_files,
                "derived_sensitive_context": derived_sensitive_context,
                "analysis_sensitive_files": analysis_sensitive_files,
                "vlm_sensitive_files": vlm_sensitive_files,
            }
        )

    leak_payloads = [item.to_dict() for item in leak_paths]
    report = DetectionReport(
        report_id=report_id,
        generated_at=iso_now(),
        input=input_metadata,
        summary={
            "logs": len(logs),
            "frame_observations": len(frame_bundle["observations"]),
            "correlated_events": len(correlation_bundle["correlated_events"]),
            "upload_candidates": len(correlation_bundle["upload_candidates"]),
            "datalog_facts": len(correlation_bundle["datalog_facts"]),
            "leak_paths": len(leak_paths),
            "suspicious_behaviors": len(suspicious_facts),
            "sensitive_source_files": len(initial_sensitive_files),
            "derived_sensitive_context": len(derived_sensitive_context),
            "analysis_sensitive_files": len(analysis_sensitive_files),
            "sessions": int(input_metadata.get("session_count") or 1),
            "groundtruth_operations": groundtruth_verdict.total_operations if groundtruth_verdict.available else 0,
            "groundtruth_leak_operations": len(groundtruth_verdict.leak_operations) if groundtruth_verdict.available else 0,
            "groundtruth_unknown_risk_operations": len(groundtruth_verdict.unknown_risk_operations)
            if groundtruth_verdict.available
            else 0,
        },
        frame_analyzer=frame_bundle,
        event_correlator=correlation_bundle,
        leak_reasoner={
            "engine": "python_taint",
            "case_id": case_id,
            "leak_paths": leak_payloads,
            "suspicious_behaviors": suspicious_facts,
            "detector_conclusion": detector_conclusion,
        },
        conclusion=detector_conclusion,
    )
    payload = report.to_dict()
    payload["verdict"] = {
        "source": "reasoner",
        "conclusion": detector_conclusion,
        "detector_conclusion": detector_conclusion,
        "groundtruth_conclusion": groundtruth_verdict.conclusion if groundtruth_verdict.available else "",
    }
    payload["detection_core"] = _build_detection_core(
        frame_bundle=frame_bundle,
        correlation_bundle=correlation_bundle,
        leak_paths=leak_payloads,
        detector_conclusion=detector_conclusion,
        verdict_source=payload["verdict"]["source"],
        groundtruth_available=groundtruth_verdict.available,
    )
    payload["groundtruth"] = groundtruth_verdict.to_dict()
    payload["log_miner"] = log_mining_payload
    payload["event_correlator"]["raw_log_events"] = records
    payload["graph"] = {"enabled": False, "status": "not_supported", "role": "neo4j_log_miner_only"}

    if output_dir is not None:
        report_files = _write_report_files(payload, Path(output_dir), report.report_id)
        payload.update(report_files)
    elif detail_output_dir is not None:
        _write_detail_files(payload, Path(detail_output_dir))
    return payload


def _write_report_files(payload: dict[str, Any], target_dir: Path, report_id: str) -> dict[str, Any]:
    target_dir.mkdir(parents=True, exist_ok=True)
    detail_dir = target_dir / report_id
    detail_dir.mkdir(parents=True, exist_ok=True)

    detail_files = _write_detail_files(payload, detail_dir)
    readable = _build_readable_report(payload, detail_files)
    report_path = target_dir / f"{report_id}.json"
    report_path.write_text(json.dumps(readable, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"report_file": str(report_path), "detail_files": detail_files}


def _write_detail_files(payload: dict[str, Any], detail_dir: Path) -> dict[str, str]:
    event_correlator = payload.get("event_correlator", {})
    frame_analyzer = payload.get("frame_analyzer", {})
    leak_reasoner = payload.get("leak_reasoner", {})
    details: dict[str, str] = {}

    event_detail = {
        "correlated_events": event_correlator.get("correlated_events", []),
        "operation_records": event_correlator.get("operation_records", []),
        "upload_candidates": event_correlator.get("upload_candidates", []),
        "file_lineage": event_correlator.get("file_lineage", {}),
        "datalog_facts": event_correlator.get("datalog_facts", []),
        "raw_log_events_count": len(event_correlator.get("raw_log_events", [])),
        "raw_log_events_source": payload.get("input", {}).get("log_file", ""),
        "raw_log_events_sources": payload.get("input", {}).get("log_files", []),
    }
    if _env_bool("DLD_WRITE_RAW_LOG_DETAILS", False):
        event_detail["raw_log_events"] = event_correlator.get("raw_log_events", [])
    details["event_correlator_details"] = _write_json(detail_dir / "event_correlator_details.json", event_detail)
    details["frame_observations"] = _write_json(detail_dir / "frame_observations.json", frame_analyzer.get("observations", []))
    details["leak_paths"] = _write_json(detail_dir / "leak_paths.json", leak_reasoner.get("leak_paths", []))
    details["verdict_check"] = _write_json(detail_dir / "verdict_check.json", _build_verdict_check(payload))
    groundtruth_copy = _copy_groundtruth_file(payload, detail_dir)
    if groundtruth_copy:
        details["groundtruth"] = groundtruth_copy
    return details


def _write_json(path: Path, payload: Any) -> str:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(path)


def _load_precomputed_baseline(path: str | Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError(f"unsupported_precomputed_baseline: {path}")
    return payload


def _copy_groundtruth_file(payload: dict[str, Any], detail_dir: Path) -> str:
    source_text = str(payload.get("input", {}).get("groundtruth_file") or "")
    if not source_text:
        return ""
    source = Path(source_text)
    if not source.exists() or not source.is_file():
        return ""
    target = detail_dir / "groundtruth.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    return str(target)


def _build_verdict_check(payload: dict[str, Any]) -> dict[str, Any]:
    groundtruth = payload.get("groundtruth", {})
    verdict = payload.get("verdict", {})
    leak_reasoner = payload.get("leak_reasoner", {})
    input_metadata = dict(payload.get("input", {}))
    expected = str(groundtruth.get("conclusion") or "")
    detector = str(leak_reasoner.get("detector_conclusion") or "")
    final = str(payload.get("conclusion") or "")
    available = bool(groundtruth.get("available"))
    is_scorable_expected = _is_scorable_conclusion(expected)
    if not available:
        score_status = "unscored"
        unscored_reason = "missing_groundtruth"
    elif is_scorable_expected:
        score_status = "scored"
        unscored_reason = ""
    else:
        score_status = "unscored"
        unscored_reason = f"unsupported_groundtruth:{expected or 'unknown'}"
    return {
        "case_id": input_metadata.get("case_id", ""),
        "case_relative_path": input_metadata.get("case_relative_path", ""),
        "groundtruth_status": input_metadata.get("groundtruth_status", ""),
        "nearest_ancestor_groundtruth_file": input_metadata.get("nearest_ancestor_groundtruth_file", ""),
        "groundtruth_available": available,
        "expected_conclusion": expected,
        "detector_conclusion": detector,
        "final_conclusion": final,
        "final_conclusion_source": verdict.get("source", ""),
        "score_status": score_status,
        "unscored_reason": unscored_reason,
        "detector_correct": detector == expected if available and is_scorable_expected else None,
        "final_correct": final == expected if available and is_scorable_expected else None,
        "detector_leak_paths": len(leak_reasoner.get("leak_paths", [])),
        "detector_suspicious_behaviors": len(leak_reasoner.get("suspicious_behaviors", [])),
        "groundtruth_operations": int(payload.get("summary", {}).get("groundtruth_operations") or 0),
        "groundtruth_leak_operations": int(payload.get("summary", {}).get("groundtruth_leak_operations") or 0),
        "groundtruth_unknown_risk_operations": int(
            payload.get("summary", {}).get("groundtruth_unknown_risk_operations") or 0
        ),
    }


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _build_readable_report(payload: dict[str, Any], detail_files: dict[str, str]) -> dict[str, Any]:
    frame_analyzer = dict(payload.get("frame_analyzer", {}))
    frame_observations = frame_analyzer.pop("observations", [])
    event_correlator = dict(payload.get("event_correlator", {}))
    leak_reasoner = dict(payload.get("leak_reasoner", {}))

    compact_event_correlator = {
        "case_id": event_correlator.get("case_id", ""),
        "session_id": event_correlator.get("session_id", ""),
        "analysis_status": event_correlator.get("analysis_status", ""),
        "analysis_windows": event_correlator.get("analysis_windows", []),
        "statistics": event_correlator.get("statistics", {}),
        "errors": event_correlator.get("errors", []),
        "counts": {
            "correlated_events": len(event_correlator.get("correlated_events", [])),
            "operation_records": len(event_correlator.get("operation_records", [])),
            "upload_candidates": len(event_correlator.get("upload_candidates", [])),
            "datalog_facts": len(event_correlator.get("datalog_facts", [])),
            "raw_log_events": len(event_correlator.get("raw_log_events", [])),
            "lineage_mappings": len(event_correlator.get("file_lineage", {}).get("direct_file_mappings", {})),
        },
        "details_file": detail_files.get("event_correlator_details", ""),
    }
    frame_analyzer["observations_count"] = len(frame_observations)
    frame_analyzer["observations_file"] = detail_files.get("frame_observations", "")
    leak_reasoner["leak_path_count"] = len(leak_reasoner.get("leak_paths", []))
    leak_reasoner["suspicious_behavior_count"] = len(leak_reasoner.get("suspicious_behaviors", []))
    leak_reasoner["leak_paths"] = []
    leak_reasoner["leak_paths_file"] = detail_files.get("leak_paths", "")

    readable = dict(payload)
    readable["frame_analyzer"] = frame_analyzer
    readable["event_correlator"] = compact_event_correlator
    readable["leak_reasoner"] = leak_reasoner
    readable["detail_files"] = detail_files
    return readable


def _build_detection_core(
    *,
    frame_bundle: dict[str, Any],
    correlation_bundle: dict[str, Any],
    leak_paths: list[dict[str, Any]],
    detector_conclusion: str,
    verdict_source: str,
    groundtruth_available: bool,
) -> dict[str, Any]:
    vision = dict(frame_bundle.get("statistics", {}).get("vision", {}))
    datalog_facts = correlation_bundle.get("datalog_facts", [])
    return {
        "method": "non_uniform_keyframes_vlm_datalog",
        "primary_chain": [
            "log_anchored_suspicious_windows",
            "non_uniform_visual_change_keyframes",
            "direct_keyframes",
            "vlm_fact_completion",
            "event_correlation",
            "datalog_taint_reasoning",
        ],
        "frame_strategy": {
            "enabled": bool(vision.get("enabled")),
            "analysis_windows": int(vision.get("analysis_windows") or 0),
            "keyframes": int(vision.get("keyframes") or 0),
            "selection": "在可疑日志时间窗口内按画面变化抽取关键帧，不做均匀抽帧",
        },
        "vlm_completion": {
            "enabled": bool(vision.get("enabled")),
            "frames_sent": int(vision.get("vlm_frames") or 0),
            "events_completed": int(vision.get("vlm_events") or 0),
            "role": "补全日志无法直接提供的前端应用、屏幕内容、文件名和外发动作证据",
        },
        "datalog_reasoning": {
            "facts": len(datalog_facts),
            "leak_paths": len(leak_paths),
            "suspicious_behaviors": sum(1 for fact in datalog_facts if fact.get("relation") == "SuspiciousBehavior"),
            "detector_conclusion": detector_conclusion,
            "role": "基于 OpenFile/TransferFile/LeakFile/SuspiciousBehavior 等事实做可解释污点传播",
        },
        "evaluation": {
            "verdict_source": verdict_source,
            "groundtruth_available": groundtruth_available,
            "groundtruth_is_evaluation_only": True,
            "note": "groundtruth 只用于最终评测，不参与 detector 判定。",
        },
    }


def _run_composite_data_case(
    case: DataCase,
    *,
    output_dir: str | Path | None,
    detail_output_dir: str | Path | None,
    observations_file: str | Path | None,
    vision_enabled: bool | None,
    max_vlm_frames: int | None,
    vision_precompute_file: str | Path | None,
    precomputed_baseline_file: str | Path | None,
    neo4j_log_miner: bool | None,
    reuse_neo4j_import: bool | None,
    non_vlm_enabled: bool | None,
    vision_debug_artifacts: bool,
    sensitive_files_config: str | Path | None,
    report_case_name: str | None,
) -> dict[str, Any]:
    if observations_file is not None and len(case.sessions) != 1:
        raise ValueError("composite_case_observations_file_requires_session_mapping")
    if vision_precompute_file is not None and len(case.sessions) != 1:
        raise ValueError("composite_case_requires_one_vision_precompute_per_session")

    baseline = _load_precomputed_baseline(precomputed_baseline_file)
    cached_precomputes = {
        str(key): str(value)
        for key, value in dict(baseline.get("session_vision_precompute_files") or {}).items()
        if str(key) and str(value)
    }
    if len(case.sessions) == 1:
        session_id = case.sessions[0].session_id
        legacy_precompute = str(vision_precompute_file or baseline.get("vision_precompute_file") or "")
        if legacy_precompute:
            cached_precomputes.setdefault(session_id, legacy_precompute)
    session_records = _composite_session_records(case.sessions, baseline)
    records = _merge_composite_records(case.sessions, session_records)
    logs = normalize_logs(records)
    initial_sensitive_files = _dedupe_paths(list(load_sensitive_files_config(sensitive_files_config)))
    analysis_sensitive_files, derived_sensitive_context = _analysis_sensitive_context(
        records,
        logs,
        initial_sensitive_files,
        session_start_ms=None,
    )
    report_id = _build_report_id(case.log_file, len(records), report_case_name or case.case_id)
    report_artifact_dir = Path(output_dir) / report_id if output_dir is not None else None
    vision_config = VisionConfig.from_env().with_overrides(enabled=vision_enabled, max_vlm_frames=max_vlm_frames)
    effective_neo4j_log_miner = neo4j_log_miner
    if effective_neo4j_log_miner is None and not vision_config.enabled:
        effective_neo4j_log_miner = False

    session_runs: list[dict[str, Any]] = []
    all_vlm_sensitive_files: list[str] = []
    log_mining_sessions: list[dict[str, Any]] = []
    cumulative_records: list[dict[str, Any]] = []
    for session in case.sessions:
        current_records = session_records[session.session_id]
        cumulative_records.extend(current_records)
        session_start_ms = _effective_session_start_ms(session, current_records)
        session_logs = normalize_logs(current_records, session_start_ms=session_start_ms)
        cumulative_logs = normalize_logs(cumulative_records)
        _, cumulative_derived_context = _analysis_sensitive_context(
            cumulative_records,
            cumulative_logs,
            initial_sensitive_files,
            session_start_ms=None,
        )
        vlm_sensitive_files = _dedupe_paths(
            [*_vlm_file_context(session_logs, analysis_sensitive_files), *cumulative_derived_context]
        )
        all_vlm_sensitive_files.extend(vlm_sensitive_files)
        cached_precompute = cached_precomputes.get(session.session_id)
        if baseline and not cached_precompute:
            raise ValueError(f"composite_precompute_missing_session: {session.session_id}")

        log_mining = None
        if not baseline:
            log_mining = mine_analysis_windows(
                case_id=f"{report_id}:{session.session_id}",
                log_file=session.log_file,
                records=current_records,
                logs=session_logs,
                sensitive_files=analysis_sensitive_files,
                vision_config=vision_config,
                neo4j_log_miner=effective_neo4j_log_miner,
                reuse_import=reuse_neo4j_import,
            )
        log_mining_payload = (
            {"source": log_mining.source, **log_mining.metadata}
            if log_mining
            else {"source": "precomputed_baseline"}
        )
        artifact_dir = report_artifact_dir / "sessions" / session.session_id if report_artifact_dir else None
        frame_bundle = analyze_video_behavior(
            session.video_file or "",
            logs=session_logs,
            sensitive_files=analysis_sensitive_files,
            vlm_sensitive_files=vlm_sensitive_files,
            observations_file=observations_file if len(case.sessions) == 1 else None,
            vision_enabled=vision_enabled,
            max_vlm_frames=max_vlm_frames,
            vision_precompute_file=cached_precompute,
            artifact_dir=artifact_dir,
            analysis_windows=log_mining.windows if log_mining else None,
            log_mining=log_mining_payload,
            debug_artifacts=vision_debug_artifacts,
        )
        use_non_vlm = True if non_vlm_enabled is None else bool(non_vlm_enabled)
        if baseline and use_non_vlm and vision_config.enabled:
            session_log_observations = dict(baseline.get("session_log_observations") or {})
            cached_log_observations = session_log_observations.get(session.session_id, [])
            if len(case.sessions) == 1 and not session_log_observations:
                cached_log_observations = baseline.get("log_observations", [])
            frame_bundle["observations"] = [*cached_log_observations, *frame_bundle["observations"]]
        frame_bundle["observations"] = [
            _namespace_absolute_observation(item, session.session_id, session_start_ms)
            for item in frame_bundle.get("observations", [])
            if isinstance(item, dict)
        ]
        session_runs.append(
            {
                "session": session,
                "recording_start_ms": session_start_ms,
                "frame_bundle": frame_bundle,
                "log_mining": log_mining_payload,
            }
        )
        log_mining_sessions.append({"session_id": session.session_id, **log_mining_payload})

    frame_bundle = _merge_session_frame_bundles(session_runs, report_artifact_dir)
    case_metadata = case.to_input_metadata()
    case_metadata.update(
        {
            "composite_case": True,
            "timeline_mode": "absolute_epoch_ms",
            "sessions": [
                {
                    **run["session"].to_input_metadata(),
                    "recording_start_ms": run["recording_start_ms"],
                }
                for run in session_runs
            ],
        }
    )
    return _finalize_pipeline(
        report_id=report_id,
        case_id=case.case_id,
        session_id=case.case_id,
        log_path=case.log_file,
        video_text="",
        records=records,
        logs=logs,
        frame_bundle=frame_bundle,
        initial_sensitive_files=initial_sensitive_files,
        derived_sensitive_context=derived_sensitive_context,
        analysis_sensitive_files=analysis_sensitive_files,
        vlm_sensitive_files=_dedupe_paths(all_vlm_sensitive_files),
        sensitive_files_config=sensitive_files_config,
        groundtruth_file=case.groundtruth_file,
        non_vlm_enabled=True if non_vlm_enabled is None else bool(non_vlm_enabled),
        recording_start_ms=0,
        input_recording_start_ms=case.recording_start_ms,
        case_metadata=case_metadata,
        log_mining_payload={
            "source": "multi_session",
            "status": "ready",
            "session_count": len(session_runs),
            "sessions": log_mining_sessions,
        },
        output_dir=output_dir,
        detail_output_dir=detail_output_dir,
    )


def _composite_session_records(
    sessions: tuple[DataSession, ...],
    baseline: dict[str, Any],
) -> dict[str, list[dict[str, Any]]]:
    result = {session.session_id: [] for session in sessions}
    if baseline:
        baseline_records = [item for item in baseline.get("records", []) if isinstance(item, dict)]
        if len(sessions) == 1 and not any(item.get("_dld_session_id") for item in baseline_records):
            session_id = sessions[0].session_id
            result[session_id] = [
                _namespace_log_record(item, session_id, index)
                for index, item in enumerate(baseline_records)
            ]
        for item in baseline_records:
            if not isinstance(item, dict):
                continue
            session_id = str(item.get("_dld_session_id") or "")
            if session_id in result:
                result[session_id].append(dict(item))
        for session in sessions:
            existing = result[session.session_id]
            known = {_pipeline_record_identity(item) for item in existing}
            for record_index, item in enumerate(_load_pipeline_records(session.log_file), start=len(existing)):
                identity = _pipeline_record_identity(item)
                if identity in known:
                    continue
                existing.append(_namespace_log_record(item, session.session_id, record_index))
                known.add(identity)
        missing = [session_id for session_id, items in result.items() if not items]
        if missing:
            raise ValueError(f"composite_baseline_records_missing_sessions: {', '.join(missing)}")
        return result

    for session in sessions:
        result[session.session_id] = [
            _namespace_log_record(item, session.session_id, record_index)
            for record_index, item in enumerate(_load_pipeline_records(session.log_file))
            if isinstance(item, dict)
        ]
    return result


def _namespace_log_record(record: dict[str, Any], session_id: str, record_index: int) -> dict[str, Any]:
    payload = dict(record)
    payload["_dld_session_id"] = session_id
    event_id = str(payload.get("event_id") or f"log_{record_index}")
    if not event_id.startswith(f"{session_id}:"):
        payload["event_id"] = f"{session_id}:{event_id}"
    else:
        payload["event_id"] = event_id
    return payload


def _merge_composite_records(
    sessions: tuple[DataSession, ...],
    records_by_session: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    ordered: list[tuple[int, int, int, dict[str, Any]]] = []
    for session_index, session in enumerate(sessions):
        records = records_by_session[session.session_id]
        start_ms = _effective_session_start_ms(session, records)
        for record_index, record in enumerate(records):
            ordered.append(
                (
                    _record_absolute_time_ms(record, start_ms, record_index),
                    session_index,
                    record_index,
                    record,
                )
            )
    return [item[3] for item in sorted(ordered, key=lambda item: item[:3])]


def _effective_session_start_ms(session: DataSession, records: list[dict[str, Any]]) -> int:
    if session.recording_start_ms:
        return session.recording_start_ms
    return next(
        (
            parsed
            for record in records
            if (parsed := parse_timestamp_ms(record.get("timestamp") or record.get("time") or ""))
        ),
        0,
    )


def _record_absolute_time_ms(record: dict[str, Any], session_start_ms: int, fallback_index: int) -> int:
    parsed = parse_timestamp_ms(record.get("timestamp") or record.get("time") or "")
    if parsed:
        return parsed
    extra = record.get("extra") if isinstance(record.get("extra"), dict) else {}
    try:
        return session_start_ms + int(float(extra.get("relative_timestamp")) * 1000)
    except (TypeError, ValueError):
        return session_start_ms + fallback_index


def _namespace_absolute_observation(item: dict[str, Any], session_id: str, session_start_ms: int) -> dict[str, Any]:
    payload = dict(item)
    observation_id = str(payload.get("observation_id") or "observation")
    if not observation_id.startswith(f"{session_id}:"):
        payload["observation_id"] = f"{session_id}:{observation_id}"
    for field in ("start_ms", "end_ms"):
        value = int(payload.get(field) or 0)
        payload[field] = value if value > 10_000_000_000 else session_start_ms + value
    payload["description"] = _namespace_observation_markers(str(payload.get("description") or ""), session_id)
    payload["session_id"] = session_id
    return payload


def _namespace_observation_markers(description: str, session_id: str) -> str:
    def replace_marker(match: re.Match[str]) -> str:
        marker, raw_values = match.groups()
        values = [
            value if value.startswith(f"{session_id}:") else f"{session_id}:{value}"
            for value in raw_values.split("|")
            if value
        ]
        return f"{marker}={'|'.join(values)}"

    return re.sub(
        r"\b(evidence_frame_ids|visual_identity_frame_ids|visual_identity|log_identity)=([^\s.]+)",
        replace_marker,
        description,
    )


def _merge_session_frame_bundles(
    session_runs: list[dict[str, Any]],
    report_artifact_dir: Path | None,
) -> dict[str, Any]:
    observations = sorted(
        [
            item
            for run in session_runs
            for item in run["frame_bundle"].get("observations", [])
            if isinstance(item, dict)
        ],
        key=lambda item: (int(item.get("start_ms") or 0), str(item.get("observation_id") or "")),
    )
    warnings = [
        f"{run['session'].session_id}: {item}"
        for run in session_runs
        for item in run["frame_bundle"].get("warnings", [])
    ]
    errors = [
        f"{run['session'].session_id}: {item}"
        for run in session_runs
        for item in run["frame_bundle"].get("errors", [])
    ]
    visions = [dict(run["frame_bundle"].get("statistics", {}).get("vision", {})) for run in session_runs]
    numeric_keys = (
        "analysis_windows",
        "keyframes",
        "keyframes_raw_all",
        "keyframe_duplicates",
        "vlm_frames",
        "vlm_source_frames",
        "vlm_events",
        "vlm_batches",
    )
    vision = dict(visions[0]) if visions else {}
    for key in numeric_keys:
        vision[key] = sum(int(item.get(key) or 0) for item in visions)
    timing_keys = {key for item in visions for key in dict(item.get("timing_seconds") or {})}
    vision["timing_seconds"] = {
        key: round(sum(float(dict(item.get("timing_seconds") or {}).get(key) or 0.0) for item in visions), 3)
        for key in timing_keys
    }
    session_artifacts: dict[str, dict[str, Any]] = {}
    for run in session_runs:
        session_id = run["session"].session_id
        session_vision = dict(run["frame_bundle"].get("statistics", {}).get("vision", {}))
        artifacts = dict(session_vision.get("artifacts", {}))
        precompute_file = str(session_vision.get("vision_precompute_file") or "")
        if precompute_file:
            artifacts.setdefault("vision_precompute_file", precompute_file)
        session_artifacts[session_id] = artifacts
    precompute_files = {
        session_id: str(artifacts.get("vision_precompute_file") or "")
        for session_id, artifacts in session_artifacts.items()
        if str(artifacts.get("vision_precompute_file") or "")
    }
    vision.update(
        {
            "enabled": any(bool(item.get("enabled")) for item in visions),
            "window_source": "multi_session",
            "log_mining": {
                "source": "multi_session",
                "sessions": [run["log_mining"] for run in session_runs],
            },
            "vlm_enabled_for_run": any(bool(item.get("vlm_enabled_for_run")) for item in visions),
            "vision_precompute_reused": bool(visions) and all(
                bool(item.get("vision_precompute_reused")) for item in visions
            ),
            "vision_precompute_file": "",
            "sessions": [
                {
                    "session_id": run["session"].session_id,
                    "recording_start_ms": run["recording_start_ms"],
                    **dict(run["frame_bundle"].get("statistics", {}).get("vision", {})),
                }
                for run in session_runs
            ],
            "artifacts": {
                "root_dir": str(report_artifact_dir or ""),
                "sessions": session_artifacts,
                "session_vision_precompute_files": precompute_files,
            },
        }
    )
    return {
        "video_file": "",
        "video_files": [str(run["session"].video_file or "") for run in session_runs],
        "observations": observations,
        "sessions": [
            {
                "session_id": run["session"].session_id,
                "recording_start_ms": run["recording_start_ms"],
                "log_file": str(run["session"].log_file),
                "video_file": str(run["session"].video_file or ""),
                "statistics": run["frame_bundle"].get("statistics", {}),
                "warnings": run["frame_bundle"].get("warnings", []),
                "errors": run["frame_bundle"].get("errors", []),
            }
            for run in session_runs
        ],
        "statistics": {
            "mode": "multi_session_direct_keyframe_vlm" if vision.get("enabled") else "multi_session_deterministic",
            "observations": len(observations),
            "session_count": len(session_runs),
            "vision": vision,
        },
        "warnings": warnings,
        "errors": errors,
    }


def run_data_case(
    case_dir: str | Path,
    *,
    output_dir: str | Path | None = None,
    detail_output_dir: str | Path | None = None,
    observations_file: str | Path | None = None,
    vision_enabled: bool | None = None,
    max_vlm_frames: int | None = None,
    vision_precompute_file: str | Path | None = None,
    precomputed_baseline_file: str | Path | None = None,
    neo4j_log_miner: bool | None = None,
    reuse_neo4j_import: bool | None = None,
    non_vlm_enabled: bool | None = None,
    vision_debug_artifacts: bool = True,
    case_root: str | Path | None = None,
    report_case_name: str | None = None,
    inherit_ancestor_groundtruth: bool = False,
    sensitive_files_config: str | Path | None = None,
) -> dict[str, Any]:
    """Run a real spec/data sample directory."""

    case = discover_data_case(
        case_dir,
        case_root=case_root,
        inherit_ancestor_groundtruth=inherit_ancestor_groundtruth,
        sensitive_files_config=sensitive_files_config,
    )
    if case.sessions:
        return _run_composite_data_case(
            case,
            output_dir=output_dir,
            detail_output_dir=detail_output_dir,
            observations_file=observations_file,
            vision_enabled=vision_enabled,
            max_vlm_frames=max_vlm_frames,
            vision_precompute_file=vision_precompute_file,
            precomputed_baseline_file=precomputed_baseline_file,
            neo4j_log_miner=neo4j_log_miner,
            reuse_neo4j_import=reuse_neo4j_import,
            non_vlm_enabled=non_vlm_enabled,
            vision_debug_artifacts=vision_debug_artifacts,
            sensitive_files_config=sensitive_files_config,
            report_case_name=report_case_name,
        )
    raise ValueError(f"case_has_no_sessions: {case.case_id}")


def _build_report_id(log_path: Path, record_count: int, case_name: str | None) -> str:
    prefix = _slugify(case_name or "")
    if not prefix:
        prefix = f"dld_{_slugify(log_path.stem)}"
    return f"{prefix}_{_slugify(log_path.stem)}_{record_count}"


def _load_pipeline_records(log_path: Path) -> list[dict[str, Any]]:
    """Use key events as primary input and merge only transfer-relevant raw events."""

    keyevents_path = log_path.with_name("keyevents.json")
    raw_path = log_path.with_name("logs.json")
    if log_path.name.lower() == "keyevents.json":
        keyevents_path = log_path
    elif log_path.name.lower() == "logs.json" and keyevents_path.exists() and keyevents_path.stat().st_size > 2:
        pass
    else:
        return load_json_records(log_path)

    records = load_json_records(keyevents_path)
    if not raw_path.exists() or raw_path.resolve() == keyevents_path.resolve():
        return records
    known_resources = {
        normalize_path(item.get("file_path") or item.get("path") or "").lower()
        for item in records
        if Path(normalize_path(item.get("file_path") or item.get("path") or "")).suffix.lower()
        in {".csv", ".doc", ".docx", ".jpeg", ".jpg", ".m4a", ".pdf", ".png", ".ppt", ".pptx", ".sql", ".txt", ".xls", ".xlsx", ".zip"}
    }
    known = {_pipeline_record_identity(item) for item in records}
    for item in load_json_records(raw_path):
        if not (_is_transfer_supplement(item, known_resources) or _is_identity_window_supplement(item)):
            continue
        identity = _pipeline_record_identity(item)
        if identity not in known:
            records.append(item)
            known.add(identity)
    return records


def _merge_baseline_records(log_path: Path, baseline_records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Refresh reusable baselines with newly retained identity/clipboard events."""

    records = list(baseline_records)
    known = {_pipeline_record_identity(item) for item in records}
    for item in _load_pipeline_records(log_path):
        identity = _pipeline_record_identity(item)
        if identity not in known:
            records.append(item)
            known.add(identity)
    return records


def _is_transfer_supplement(record: dict[str, Any], known_resources: set[str]) -> bool:
    event_type = str(record.get("event_type") or record.get("type") or "").lower()
    extra = record.get("extra") if isinstance(record.get("extra"), dict) else {}
    operation = str(extra.get("raw_operation") or record.get("operation") or "").lower()
    if event_type in {
        "file_selected",
        "file_upload",
        "upload",
        "uploaded",
        "upload_complete",
        "send",
        "sent",
        "clipboard_text",
        "clipboard_write",
        "clipboard_image",
    }:
        return True
    path = normalize_path(record.get("file_path") or record.get("path") or "").lower()
    if path and path in known_resources and event_type in {"opened", "read"} and "browser_file_access" in operation:
        return True
    if event_type not in {"created", "modified", "renamed"} or not _is_document_path(path):
        return False
    parent = str(Path(path).parent).lower()
    return any(parent == str(Path(resource).parent).lower() for resource in known_resources)


def _is_identity_window_supplement(record: dict[str, Any]) -> bool:
    """Keep concise foreground-window records that can bind a generic filename."""

    event_type = str(record.get("event_type") or record.get("type") or "").lower()
    if event_type != "app_switch":
        return False
    window = record.get("window_info") if isinstance(record.get("window_info"), dict) else {}
    title = str(record.get("window_title") or window.get("window_title") or "").strip()
    if not title or len(title) > 240:
        return False
    if re.search(r"\.(?:doc|docx|pdf|ppt|pptx|xls|xlsx|sql|txt|png|jpg|jpeg|zip)(?:\s|$|-)", title, re.IGNORECASE):
        return True
    normalized = title.lower()
    return any(
        marker in normalized
        for marker in (
            "文件资源管理器",
            "file explorer",
            "onedrive",
            "飞书",
            "feishu",
            "lark",
            "teams",
            "微信",
            "wechat",
            "qq(浏览)",
            "vmware",
            "virtualbox",
            "parallels",
        )
    )


def _is_document_path(path: str) -> bool:
    return Path(path).suffix.lower() in {
        ".csv",
        ".doc",
        ".docx",
        ".jpeg",
        ".jpg",
        ".m4a",
        ".pdf",
        ".png",
        ".ppt",
        ".pptx",
        ".sql",
        ".txt",
        ".xls",
        ".xlsx",
        ".zip",
    }


def _pipeline_record_identity(record: dict[str, Any]) -> tuple[str, str, str]:
    extra = record.get("extra") if isinstance(record.get("extra"), dict) else {}
    return (
        str(extra.get("relative_timestamp") or record.get("timestamp") or record.get("time") or ""),
        str(record.get("event_type") or record.get("type") or "").lower(),
        normalize_path(record.get("file_path") or record.get("path") or "").lower(),
    )


def _slugify(value: str) -> str:
    slug = re.sub(r"[^0-9A-Za-z._-]+", "-", value.strip())
    slug = slug.strip("-._")
    return slug or ""


def _detector_conclusion(leak_paths: list[Any], suspicious_facts: list[dict[str, Any]]) -> str:
    if leak_paths:
        return "data_leak_risk_detected"
    if suspicious_facts:
        return "suspicious_behavior_detected"
    return "no_confirmed_data_leak"


def _suspicious_datalog_facts(correlation_bundle: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        dict(item)
        for item in correlation_bundle.get("datalog_facts", [])
        if isinstance(item, dict) and item.get("relation") == "SuspiciousBehavior"
    ]


def _is_scorable_conclusion(value: str) -> bool:
    return value in {"data_leak_risk_detected", "suspicious_behavior_detected", "no_confirmed_data_leak"}


def _dedupe_paths(paths: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for path in paths:
        text = normalize_path(path)
        key = text.lower()
        if not key or key in seen:
            continue
        seen.add(key)
        result.append(text)
    return result


def _analysis_sensitive_context(
    records: list[Any],
    logs: list[Any],
    initial_sensitive_files: list[str],
    *,
    session_start_ms: int | None,
) -> tuple[list[str], list[str]]:
    """Build the recursive, log-evidenced context used before VLM analysis."""

    lineage_logs = logs or normalize_logs(
        [item for item in records if isinstance(item, dict)],
        session_start_ms=session_start_ms,
    )
    derived = EventCorrelator().derived_sensitive_files(lineage_logs, initial_sensitive_files)
    return _dedupe_paths([*initial_sensitive_files, *derived]), derived


def _vlm_file_context(logs: list[Any], analysis_sensitive_files: list[str]) -> list[str]:
    """Keep VLM file context limited to sensitive paths observed in this case."""

    observed: list[str] = []
    for event in logs:
        path = normalize_path(getattr(event, "file_path", ""))
        if path and any(same_file(path, sensitive) for sensitive in analysis_sensitive_files):
            observed.append(path)
    return _dedupe_paths(observed)



