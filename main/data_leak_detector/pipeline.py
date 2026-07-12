"""End-to-end orchestration for the canonical DataLeakDetector pipeline."""

from __future__ import annotations

import json
import os
from pathlib import Path
import re
import shutil
from typing import Any

from .datasets import discover_data_case
from .event_correlator import EventCorrelator
from .frame_analyzer import analyze_video_behavior
from .frame_analyzer.config import VisionConfig
from .log_mining import mine_analysis_windows
from .groundtruth import evaluate_groundtruth
from .io import iso_now, load_json_records, normalize_logs, normalize_path
from .leak_reasoner import DatalogEngine
from .models import DetectionReport


def run_pipeline(
    log_file: str | Path,
    video_file: str | Path = "",
    output_dir: str | Path | None = None,
    detail_output_dir: str | Path | None = None,
    sensitive_files: list[str] | None = None,
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
    records = list(baseline.get("records", [])) if baseline else load_json_records(log_path)
    logs = [] if baseline else normalize_logs(records, session_start_ms=session_start_ms)
    initial_sensitive_files = list(sensitive_files or [])
    vision_sensitive_files = _vision_sensitive_files(records, logs, initial_sensitive_files, session_start_ms=session_start_ms)
    initial_sensitive_keys = {normalize_path(item).lower() for item in initial_sensitive_files if normalize_path(item)}
    vision_derived_sensitive_files = [
        item for item in vision_sensitive_files if normalize_path(item).lower() not in initial_sensitive_keys
    ]
    report_id = _build_report_id(log_path, len(records), case_name)
    target_dir = Path(output_dir) if output_dir is not None else None
    vision_artifact_dir = target_dir / report_id if target_dir is not None else None
    if vision_artifact_dir is not None:
        _copy_groundtruth_file({"input": {"groundtruth_file": str(groundtruth_file or "")}}, vision_artifact_dir)
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
        sensitive_files=vision_sensitive_files,
        vision_config=vision_config,
        neo4j_log_miner=effective_neo4j_log_miner,
        reuse_import=reuse_neo4j_import,
    ) if not baseline else None

    frame_bundle = analyze_video_behavior(
        video_path or "",
        logs=logs,
        sensitive_files=vision_sensitive_files,
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
    correlation_bundle = EventCorrelator().run(
        {
            "session_id": video_path.stem if video_path else log_path.stem,
            "log_events": records,
            "frame_segments": frame_bundle["observations"],
            "sensitive_files": initial_sensitive_files,
            "recording_start_ms": int(session_start_ms or 0),
            "non_vlm_enabled": use_non_vlm,
        }
    )

    engine = DatalogEngine()
    for fact in correlation_bundle["datalog_facts"]:
        engine.add_fact(fact["relation"], *fact["args"])
    leak_paths = engine.query_leak()
    suspicious_facts = _suspicious_datalog_facts(correlation_bundle)
    detector_conclusion = _detector_conclusion(leak_paths, suspicious_facts)
    groundtruth_verdict = evaluate_groundtruth(groundtruth_file)
    final_conclusion = detector_conclusion

    input_metadata = {
        "log_file": str(log_path),
        "video_file": video_text,
        "groundtruth_file": str(groundtruth_file or ""),
        "recording_start_ms": int(session_start_ms or 0),
        "vision_sensitive_files": vision_sensitive_files,
        "vision_derived_sensitive_files": vision_derived_sensitive_files,
    }
    if case_metadata:
        input_metadata.update(case_metadata)
        input_metadata["log_file"] = str(log_path)
        input_metadata["video_file"] = video_text
        input_metadata["groundtruth_file"] = str(groundtruth_file or "")
        input_metadata["recording_start_ms"] = int(session_start_ms or 0)
        input_metadata["vision_sensitive_files"] = vision_sensitive_files
        input_metadata["vision_derived_sensitive_files"] = vision_derived_sensitive_files

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
            "vision_sensitive_files": len(vision_sensitive_files),
            "vision_derived_sensitive_files": len(vision_derived_sensitive_files),
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
            "leak_paths": [item.to_dict() for item in leak_paths],
            "suspicious_behaviors": suspicious_facts,
            "detector_conclusion": detector_conclusion,
        },
        conclusion=final_conclusion,
    )
    payload = report.to_dict()
    payload["verdict"] = {
        "source": "reasoner",
        "conclusion": final_conclusion,
        "detector_conclusion": detector_conclusion,
        "groundtruth_conclusion": groundtruth_verdict.conclusion if groundtruth_verdict.available else "",
    }
    payload["detection_core"] = _build_detection_core(
        frame_bundle=frame_bundle,
        correlation_bundle=correlation_bundle,
        leak_paths=[item.to_dict() for item in leak_paths],
        detector_conclusion=detector_conclusion,
        verdict_source=payload["verdict"]["source"],
        groundtruth_available=groundtruth_verdict.available,
    )
    payload["groundtruth"] = groundtruth_verdict.to_dict()
    payload["log_miner"] = {"source": log_mining.source, **log_mining.metadata} if log_mining else {"source": "precomputed_baseline"}
    payload["event_correlator"]["raw_log_events"] = records
    payload["graph"] = {"enabled": False, "status": "not_supported", "role": "neo4j_log_miner_only"}

    if output_dir is not None:
        target_dir = Path(output_dir)
        report_files = _write_report_files(payload, target_dir, report.report_id)
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

def run_data_case(
    case_dir: str | Path,
    *,
    output_dir: str | Path | None = None,
    detail_output_dir: str | Path | None = None,
    sensitive_files: list[str] | None = None,
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
) -> dict[str, Any]:
    """Run a real spec/data sample directory."""

    case = discover_data_case(
        case_dir,
        case_root=case_root,
        inherit_ancestor_groundtruth=inherit_ancestor_groundtruth,
    )
    merged_sensitive = list(dict.fromkeys([*case.sensitive_files, *(sensitive_files or [])]))
    report = run_pipeline(
        log_file=case.log_file,
        video_file=case.video_file or "",
        output_dir=output_dir,
        detail_output_dir=detail_output_dir,
        sensitive_files=merged_sensitive,
        observations_file=observations_file,
        vision_enabled=vision_enabled,
        max_vlm_frames=max_vlm_frames,
        vision_precompute_file=vision_precompute_file,
        precomputed_baseline_file=precomputed_baseline_file,
        groundtruth_file=case.groundtruth_file,
        neo4j_log_miner=neo4j_log_miner,
        reuse_neo4j_import=reuse_neo4j_import,
        non_vlm_enabled=non_vlm_enabled,
        vision_debug_artifacts=vision_debug_artifacts,
        case_name=report_case_name or case.case_id,
        session_start_ms=case.recording_start_ms,
        case_metadata=case.to_input_metadata(),
    )
    report["input"].update(case.to_input_metadata())
    return report


def _build_report_id(log_path: Path, record_count: int, case_name: str | None) -> str:
    prefix = _slugify(case_name or "")
    if not prefix:
        prefix = f"dld_{_slugify(log_path.stem)}"
    return f"{prefix}_{_slugify(log_path.stem)}_{record_count}"


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


def _vision_sensitive_files(
    records: list[Any],
    logs: list[Any],
    sensitive_files: list[str],
    *,
    session_start_ms: int | None,
) -> list[str]:
    normalized_initial = _dedupe_paths(sensitive_files)
    if not normalized_initial:
        return []
    lineage_logs = logs or normalize_logs(
        [item for item in records if isinstance(item, dict)],
        session_start_ms=session_start_ms,
    )
    derived = EventCorrelator().derived_sensitive_files(lineage_logs, normalized_initial)
    return _dedupe_paths([*normalized_initial, *derived])


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



