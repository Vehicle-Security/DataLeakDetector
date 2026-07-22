"""Contract tests for the pipeline, dataset discovery, VLM parsing, and Neo4j adapter."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest
import run_e2e as run_e2e_module

from data_leak_detector import run_data_case, run_pipeline
from data_leak_detector.datasets import discover_data_case, discover_data_case_directories
from data_leak_detector.event_correlator import EventCorrelator
from data_leak_detector.event_correlator.facts import build_datalog_facts
from data_leak_detector.event_correlator.lineage import Lineage
from data_leak_detector.frame_analyzer import analyze_video_behavior
from data_leak_detector.frame_analyzer.artifacts import export_vision_artifacts
from data_leak_detector.frame_analyzer.vlm_dispatch import (
    _combine_vlm_usage,
    _shared_vlm_dispatcher,
    _shared_vlm_endpoint_locks,
    _validate_vlm_evidence,
    build_vlm_clients,
    combine_vlm_request_metrics,
    effective_vlm_parallelism,
    run_vlm_batches,
    vlm_frame_batches,
    vlm_request_artifact_payload,
)
from data_leak_detector.frame_analyzer.apps import identify_frontend_app
from data_leak_detector.frame_analyzer.config import VisionConfig
from data_leak_detector.frame_analyzer.frames import (
    AnalysisWindow,
    _FrameCandidate,
    KeyFrame,
    KeyFrameDuplicate,
    KeyFrameSelection,
    _clamp_window_to_duration,
    _coverage_timestamps,
    _dedupe_keyframes_globally,
    _ffmpeg_cuda_frame_command,
    _focus_actionable_keyframes,
    _focus_file_dialog_flows,
    _focus_semantic_action_phases,
    _budget_window_candidates,
    _frame_entropy,
    _hamming,
    _post_action_visual_evidence,
    _probe_timestamps,
    _read_frames_for_timestamps,
    _select_context_evidence,
    _select_window_candidates,
    _should_keep_frame,
    _timestamp_groups,
    _trim_mandatory_evidence,
    build_video_coverage_windows,
    merge_analysis_windows,
    select_keyframes_detailed,
)
from data_leak_detector.frame_analyzer.parser import ParsedVisionEvent, parse_vlm_response, parse_vlm_response_detailed, vision_events_to_observations
from data_leak_detector.frame_analyzer.vlm_client import VlmRequestFrame, VlmResponse, _prompt, build_vlm_frame_grids, choose_keyframes_for_vlm, prepare_vlm_frame_images
from data_leak_detector.log_mining import _action_kind, _compact_event_view, _may_need_analysis_window, build_analysis_windows, mine_analysis_windows
from data_leak_detector.neo4j.importer import fingerprint_records, records_to_graph_events
from data_leak_detector.groundtruth import evaluate_groundtruth
from data_leak_detector.io import normalize_logs, normalize_path, parse_timestamp_ms, same_file
from run_e2e import _precompute_baseline_matches_mode, _release_direct_defaults, _reusable_precompute_baseline
from data_leak_detector.io import load_json_records
from data_leak_detector.leak_reasoner import DatalogEngine
from data_leak_detector.models import CorrelatedEvent, LeakPath, UploadCandidate
from data_leak_detector.policy import contains_any, load_policy_config
from data_leak_detector.policy import classify_sink
from data_leak_detector.sensitivity import load_sensitive_files_config
from data_leak_detector.pipeline import _build_report_id, _load_pipeline_records


def _records() -> list[dict]:
    original = "C:/Users/alice/Documents/customer_salary.xlsx"
    derived = "C:/Users/alice/Desktop/customer_salary_part1.xlsx"
    return [
        {
            "timestamp": "2026-06-28T10:00:00.000",
            "event_type": "file_open",
            "file_path": original,
            "process_info": {"process_name": "excel.exe"},
            "window_info": {"window_title": "customer_salary.xlsx - Excel"},
        },
        {
            "timestamp": "2026-06-28T10:00:30.000",
            "event_type": "created",
            "file_path": derived,
            "source_file": original,
            "process_info": {"process_name": "python.exe"},
            "window_info": {"window_title": "split customer salary"},
        },
        {
            "timestamp": "2026-06-28T10:01:00.000",
            "event_type": "file_upload",
            "file_path": derived,
            "process_info": {"process_name": "msedge.exe"},
            "window_info": {"window_title": "Gmail attach file upload completed"},
        },
    ]


def test_frame_analyzer_creates_log_anchored_observations() -> None:
    logs = normalize_logs(_records())

    bundle = analyze_video_behavior("", logs=logs, sensitive_files=["customer_salary.xlsx"])

    assert bundle["statistics"]["observations"] >= 3
    assert any(item["operation_type"] == "external_sink_interaction" for item in bundle["observations"])


def test_vlm_prompt_treats_email_send_confirmation_as_direct_leak() -> None:
    frame = KeyFrame(
        frame_id="email_send_confirmation",
        timestamp_ms=20_000,
        image_path="email-confirmation.jpg",
        score=1.0,
        reason="medium:anchor",
        window_id="window_0",
    )

    prompt = _prompt(
        [VlmRequestFrame(frame=frame, visual_note="", visual_confidence=0.0)],
        ["C:/Users/alice/Desktop/financial_report.docx"],
        ["Edge"],
    )

    assert "email send confirmation" in prompt
    assert "Do not require an inbox update" in prompt


def test_vlm_prompt_requires_executed_transfer_evidence() -> None:
    frame = KeyFrame(
        frame_id="copy_preparation",
        timestamp_ms=20_000,
        image_path="copy-preparation.jpg",
        score=1.0,
        reason="medium:anchor",
        window_id="window_0",
    )

    prompt = _prompt(
        [VlmRequestFrame(frame=frame, visual_note="", visual_confidence=0.0)],
        ["C:/Users/alice/Desktop/customer_contacts.pdf"],
        ["Explorer"],
    )

    assert "merely visible as an executed leak" in prompt
    assert "unselected context-menu" in prompt
    assert "hidden_transfer rather than direct_leak" in prompt


def test_reusable_precompute_baseline_excludes_nested_session_cache(tmp_path: Path) -> None:
    parent = tmp_path / "stage1" / "outlook"
    direct = parent / "outlook_logs_402" / "pipeline_baseline.json"
    nested = parent / "session_20260420_191957" / "session_logs_3543" / "pipeline_baseline.json"
    direct.parent.mkdir(parents=True)
    nested.parent.mkdir(parents=True)
    direct.write_text('{"precompute_mode":"direct_keyframes_only"}', encoding="utf-8")
    nested.write_text('{"precompute_mode":"direct_keyframes_only"}', encoding="utf-8")

    assert _reusable_precompute_baseline(parent) == direct


def test_release_precompute_rejects_stale_keyframe_strategy(tmp_path: Path) -> None:
    baseline = tmp_path / "pipeline_baseline.json"
    baseline.write_text(
        json.dumps({"schema_version": 1, "precompute_mode": "direct_keyframes_only"}),
        encoding="utf-8",
    )

    assert _precompute_baseline_matches_mode(baseline, "direct_keyframes_only") is False

    baseline.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "precompute_mode": "direct_keyframes_only",
                "vision_strategy_version": run_e2e_module.VISION_PRECOMPUTE_STRATEGY_VERSION,
            }
        ),
        encoding="utf-8",
    )

    assert _precompute_baseline_matches_mode(baseline, "direct_keyframes_only") is True


def test_release_precompute_writes_every_composite_session_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "stage4"
    case_dir = root / "e2e-1"
    case_dir.mkdir(parents=True)
    (case_dir / "groundtruth.json").write_text(json.dumps({"operations": []}), encoding="utf-8")
    session_ids = ["session_20260101_090000", "session_20260103_110000"]
    for session_id in session_ids:
        _write_composite_session(case_dir, session_id, "2026-01-01 09:00:00", [])

    def fake_run_data_case(case: Path, **kwargs: object) -> dict:
        artifact_root = Path(str(kwargs["output_dir"])) / "e2e-1_keyevents_2"
        caches: dict[str, str] = {}
        for session_id in session_ids:
            cache = artifact_root / "sessions" / session_id / "vision_precompute.json"
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_text("{}", encoding="utf-8")
            caches[session_id] = str(cache)
        return {
            "frame_analyzer": {
                "observations": [],
                "statistics": {
                    "vision": {
                        "artifacts": {
                            "root_dir": str(artifact_root),
                            "session_vision_precompute_files": caches,
                        }
                    }
                },
            },
            "event_correlator": {
                "raw_log_events": [
                    {"_dld_session_id": session_id, "event_type": "test"} for session_id in session_ids
                ]
            },
        }

    monkeypatch.setattr(run_e2e_module, "run_data_case", fake_run_data_case)
    completed = run_e2e_module._build_release_vision_precompute(
        str(root),
        common_args={},
        cache_root=tmp_path / "cache",
        workers=1,
        neo4j_log_miner=False,
    )

    baseline = Path(completed["e2e-1"])
    payload = json.loads(baseline.read_text(encoding="utf-8"))
    assert set(payload["session_vision_precompute_files"]) == set(session_ids)
    assert run_e2e_module._precompute_baseline_covers_case(baseline, case_dir) is True

    payload["session_vision_precompute_files"].pop(session_ids[-1])
    baseline.write_text(json.dumps(payload), encoding="utf-8")
    assert run_e2e_module._precompute_baseline_covers_case(baseline, case_dir) is False


def test_release_keeps_deterministic_log_evidence_enabled() -> None:
    args = SimpleNamespace(
        max_vlm_frames=None,
        release_debug_artifacts=True,
        neo4j_log_miner=False,
    )

    release_args = _release_direct_defaults({}, args)

    assert release_args["vision_enabled"] is True
    assert release_args["non_vlm_enabled"] is True


def test_release_marks_vlm_errors_failed_and_continues_remaining_cases(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    case_root = tmp_path / "cases"
    case_dirs = [case_root / "case-a", case_root / "case-b"]
    for case_dir in case_dirs:
        case_dir.mkdir(parents=True)
    output_dir = tmp_path / "release"

    monkeypatch.setattr(run_e2e_module, "discover_data_case_directories", lambda root: case_dirs)
    monkeypatch.setattr(run_e2e_module, "data_case_id", lambda case, root: case.name)

    def run_case(case: Path, **kwargs: object) -> dict:
        if case.name == "case-a":
            return {
                "frame_analyzer": {"errors": ["vlm_batch_failed[0]: quota exceeded"]},
                "conclusion": "no_confirmed_data_leak",
            }
        return {"frame_analyzer": {"errors": []}, "conclusion": "no_confirmed_data_leak"}

    monkeypatch.setattr(run_e2e_module, "run_data_case", run_case)

    result = run_e2e_module._run_case_root(
        str(case_root),
        common_args={},
        output_dir=str(output_dir),
        workers=1,
        release=True,
    )

    batch = result["batch"]
    assert batch["completed_cases"] == 1
    assert batch["failed_cases"] == 1
    assert batch["aborted"] is False
    assert batch["abort_reason"] == ""
    assert Path(batch["retry_case_list"]).read_text(encoding="utf-8") == "case-a\n"
    progress = json.loads((output_dir / "release_progress.json").read_text(encoding="utf-8"))
    assert progress["state"] == "failed"


def test_event_correlator_skips_file_system_noise_but_keeps_upload_action() -> None:
    sensitive = "C:/Users/alice/Desktop/customer_contract.docx"
    records = [
        {
            "timestamp": f"2026-01-01T00:00:{index:02d}",
            "event_type": "opened" if index % 2 == 0 else "closed",
            "file_path": sensitive,
            "process_info": {"process_name": "chrome.exe"},
        }
        for index in range(20)
    ]
    records.append(
        {
            "timestamp": "2026-01-01T00:01:00",
            "event_type": "file_upload",
            "file_path": sensitive,
            "process_info": {"process_name": "chrome.exe"},
            "extra": {"category": "upload"},
        }
    )

    bundle = EventCorrelator().run(
        {"log_events": records, "frame_segments": [], "sensitive_files": [sensitive], "non_vlm_enabled": True}
    )

    assert len(bundle["correlated_events"]) == 1
    assert len(bundle["upload_candidates"]) == 1
    assert [item["relation"] for item in bundle["datalog_facts"]].count("LeakFile") == 1


def test_same_file_ignores_whitespace_before_extension() -> None:
    assert same_file(
        "C:/Users/alice/Desktop/customer_contacts .pdf",
        "customer_contacts.pdf",
    )


def test_dataset_prefers_keyevents_over_raw_logs(tmp_path: Path) -> None:
    case_dir = tmp_path / "case"
    logs_dir = case_dir / "logs"
    video_dir = case_dir / "video"
    logs_dir.mkdir(parents=True)
    video_dir.mkdir()
    (logs_dir / "logs.json").write_text('[{"event_type":"modified"}]', encoding="utf-8")
    (logs_dir / "keyevents.json").write_text('[{"event_type":"clipboard_operation"}]', encoding="utf-8")

    case = discover_data_case(case_dir)

    assert case.log_file == (logs_dir / "keyevents.json").resolve()


def test_high_risk_app_switch_does_not_create_behavior_window() -> None:
    logs = normalize_logs(
        [
            {
                "timestamp": "2026-01-01T12:00:00",
                "event_type": "app_switch",
                "file_path": "",
                "app_name": "Edge",
                "window_info": {"window_title": "Unknown Business Portal"},
                "extra": {
                    "source": "window_monitor",
                    "category": "浏览器",
                    "risk_level": "高",
                    "relative_timestamp": 42.5,
                },
            }
        ]
    )

    windows = build_analysis_windows(
        logs,
        [],
        VisionConfig(frame_window_before_ms=30_000, frame_window_after_ms=120_000, include_weak_windows=True),
    )

    assert windows == []


def test_analysis_windows_sample_strong_upload_events_more_densely() -> None:
    logs = normalize_logs(
        [
            {
                "timestamp": "2026-01-01T12:00:00",
                "event_type": "file_selected",
                "file_path": "C:/Users/alice/Documents/secret.docx",
                "window_info": {"window_title": "Upload files - Unknown Cloud"},
                "extra": {
                    "source": "file_dialog_monitor",
                    "category": "文件上传",
                    "raw_operation": "file_selected",
                    "relative_timestamp": 60.0,
                },
            }
        ]
    )

    windows = build_analysis_windows(logs, [], VisionConfig())

    assert windows[0].priority == "strong"
    assert windows[0].start_ms == 55_000
    assert windows[0].end_ms == 90_000
    assert windows[0].step_ms == 250
    assert windows[0].max_keyframes == VisionConfig().max_keyframes_per_strong_window


def test_analysis_windows_keep_derived_transfer_anchor_when_upload_exists() -> None:
    original = "C:/Users/alice/Documents/board_minutes.docx"
    derived = "C:/Users/alice/Desktop/share.pdf"
    logs = normalize_logs(
        [
            {
                "timestamp": "2026-01-01T12:00:00",
                "event_type": "file_open",
                "file_path": original,
                "extra": {"relative_timestamp": 0.0},
                "process_info": {"process_name": "wps.exe"},
            },
            {
                "timestamp": "2026-01-01T12:00:30",
                "event_type": "created",
                "file_path": derived,
                "source_file": original,
                "extra": {"raw_operation": "export", "relative_timestamp": 30.0},
                "process_info": {"process_name": "wps.exe"},
            },
            {
                "timestamp": "2026-01-01T12:01:00",
                "event_type": "file_selected",
                "file_path": derived,
                "extra": {"category": "upload", "raw_operation": "file_selected", "relative_timestamp": 60.0},
                "process_info": {"process_name": "msedge.exe"},
                "window_info": {"window_title": "ChatGPT upload"},
            },
        ]
    )

    windows = build_analysis_windows(logs, [original, derived], VisionConfig())

    assert any(window.priority == "strong" and 30_000 in window.anchor_ms for window in windows)
    assert any(window.priority == "strong" and 60_000 in window.anchor_ms for window in windows)


def test_unrelated_browser_document_access_is_not_transfer_evidence() -> None:
    logs = normalize_logs(
        [
            {
                "timestamp": "2026-01-01T12:00:30",
                "event_type": "opened",
                "file_path": "C:/Users/alice/Downloads/public-paper.pdf",
                "app_name": "Edge",
                "process_info": {"process_name": "msedge.exe"},
                "extra": {"raw_operation": "browser_file_access", "relative_timestamp": 30.0},
            },
            {
                "timestamp": "2026-01-01T12:05:00",
                "event_type": "app_switch",
                "app_name": "Edge",
                "window_info": {"window_title": "Video site"},
                "extra": {"relative_timestamp": 300.0},
            },
        ]
    )

    assert build_analysis_windows(logs, ["C:/Users/alice/Desktop/secret.docx"], VisionConfig()) == []


def test_direct_sensitive_browser_access_becomes_file_selection_window() -> None:
    sensitive = "C:/Users/alice/Desktop/secret.docx"
    logs = normalize_logs(
        [
            {
                "timestamp": "2026-01-01T12:00:12",
                "event_type": "created",
                "file_path": sensitive,
                "app_name": "Edge",
                "process_info": {"process_name": "msedge.exe"},
                "extra": {"raw_operation": "browser_file_access", "relative_timestamp": 12.0},
            }
        ]
    )

    windows = build_analysis_windows(logs, [sensitive], VisionConfig())

    assert len(windows) == 1
    assert windows[0].action_phases == ((12_000, "file_selected"),)


def test_clipboard_near_sensitive_window_title_survives_noise_path_before_external_session() -> None:
    sensitive = "C:/Users/alice/Desktop/strategy.docx"
    logs = normalize_logs(
        [
            {
                "timestamp": "2026-01-01T12:00:10",
                "event_type": "modified",
                "file_path": "C:/Users/alice/AppData/Roaming/QQ/nt_db/nt_msg.db-wal",
                "window_info": {"window_title": "strategy.docx - Word"},
                "extra": {"relative_timestamp": 10.0},
            },
            {
                "timestamp": "2026-01-01T12:00:15",
                "event_type": "clipboard_text",
                "app_name": "WINWORD",
                "content_preview": "sensitive paragraph",
                "extra": {"relative_timestamp": 15.0, "raw_operation": "clipboard_text"},
            },
            {
                "timestamp": "2026-01-01T12:00:20",
                "event_type": "app_switch",
                "app_name": "ChatGPT",
                "window_info": {"window_title": "ChatGPT"},
                "extra": {"relative_timestamp": 20.0},
            },
        ]
    )

    windows = build_analysis_windows(logs, [sensitive], VisionConfig())

    assert len(windows) == 1
    assert (15_000, "clipboard") in windows[0].action_phases
    assert (20_000, "external_session") in windows[0].action_phases


def test_stale_sensitive_window_title_does_not_warrant_external_session_without_action() -> None:
    sensitive = "C:/Users/alice/Desktop/strategy.docx"
    logs = normalize_logs(
        [
            {
                "timestamp": "2026-01-01T12:00:10",
                "event_type": "modified",
                "file_path": "C:/Users/alice/AppData/Roaming/QQ/nt_db/nt_msg.db-wal",
                "window_info": {"window_title": "strategy.docx - Word"},
                "extra": {"relative_timestamp": 10.0},
            },
            {
                "timestamp": "2026-01-01T12:00:20",
                "event_type": "app_switch",
                "app_name": "ChatGPT",
                "window_info": {"window_title": "ChatGPT"},
                "extra": {"relative_timestamp": 20.0},
            },
        ]
    )

    assert build_analysis_windows(logs, [sensitive], VisionConfig()) == []


def test_clipboard_uses_current_document_instead_of_unclosed_sensitive_activity() -> None:
    sensitive = "C:/Users/alice/Desktop/salary.xlsx"
    logs = normalize_logs(
        [
            {
                "timestamp": "2026-01-01T12:00:00",
                "event_type": "modified",
                "file_path": sensitive,
                "process_info": {"process_name": "wps.exe"},
                "window_info": {"window_title": "salary.xlsx - WPS Office"},
                "extra": {"relative_timestamp": 0.0},
            },
            {
                "timestamp": "2026-01-01T12:01:00",
                "event_type": "modified",
                "file_path": "C:/Users/alice/AppData/Roaming/WPS/state.json",
                "process_info": {"process_name": "wps.exe"},
                "window_info": {"window_title": "public_report.pptx - WPS Office"},
                "extra": {"relative_timestamp": 60.0},
            },
            {
                "timestamp": "2026-01-01T12:01:10",
                "event_type": "clipboard_text",
                "process_info": {"process_name": "wps.exe"},
                "extra": {"relative_timestamp": 70.0, "raw_operation": "clipboard_text"},
            },
            {
                "timestamp": "2026-01-01T12:01:20",
                "event_type": "app_switch",
                "app_name": "ChatGPT",
                "window_info": {"window_title": "ChatGPT"},
                "extra": {"relative_timestamp": 80.0},
            },
        ]
    )

    assert build_analysis_windows(logs, [sensitive], VisionConfig()) == []


def test_one_action_is_owned_by_only_one_external_session() -> None:
    sensitive = "C:/Users/alice/Pictures/Screenshots/secret.png"
    logs = normalize_logs(
        [
            {
                "timestamp": "2026-01-01T12:00:30",
                "event_type": "app_switch",
                "app_name": "QQ",
                "window_info": {"window_title": "QQ"},
                "extra": {"relative_timestamp": 30.0},
            },
            {
                "timestamp": "2026-01-01T12:00:40",
                "event_type": "file_selected",
                "file_path": sensitive,
                "app_name": "QQ",
                "window_info": {"window_title": "QQ"},
                "extra": {"relative_timestamp": 40.0, "raw_operation": "file_selected"},
            },
            {
                "timestamp": "2026-01-01T12:00:50",
                "event_type": "app_switch",
                "app_name": "ChatGPT",
                "window_info": {"window_title": "ChatGPT"},
                "extra": {"relative_timestamp": 50.0},
            },
            {
                "timestamp": "2026-01-01T12:01:10",
                "event_type": "app_switch",
                "app_name": "System",
                "window_info": {"window_title": "Settings"},
                "extra": {"relative_timestamp": 70.0},
            },
        ]
    )

    windows = build_analysis_windows(logs, [sensitive], VisionConfig())

    assert len(windows) == 1
    assert windows[0].reason == "strong:external_session:chat"
    assert (40_000, "file_selected") in windows[0].action_phases


def test_distant_external_session_does_not_inherit_capture_without_transfer_evidence() -> None:
    screenshot = "C:/Users/alice/Pictures/Screenshots/secret.png"
    logs = normalize_logs(
        [
            {
                "timestamp": "2026-01-01T12:00:10",
                "event_type": "screenshot",
                "file_path": screenshot,
                "process_info": {"process_name": "SnippingTool.exe"},
                "extra": {"relative_timestamp": 10.0, "raw_operation": "screenshot"},
            },
            {
                "timestamp": "2026-01-01T12:01:30",
                "event_type": "app_switch",
                "app_name": "ChatGPT",
                "window_info": {"window_title": "ChatGPT"},
                "extra": {"relative_timestamp": 90.0},
            },
        ]
    )

    windows = build_analysis_windows(logs, [screenshot], VisionConfig())

    assert any((10_000, "capture") in window.action_phases for window in windows)
    assert not any(
        action == "external_session"
        for window in windows
        for _, action in window.action_phases
    )


def test_sensitive_file_editing_does_not_create_production_window() -> None:
    sensitive = "C:/Users/alice/Documents/strategy.docx"
    logs = normalize_logs(
        [
            {
                "timestamp": "2026-01-01T12:00:00",
                "event_type": "file_open",
                "file_path": sensitive,
                "app_name": "Word",
                "extra": {"relative_timestamp": 0.0},
            },
            {
                "timestamp": "2026-01-01T12:07:00",
                "event_type": "app_switch",
                "app_name": "System",
                "extra": {"relative_timestamp": 420.0},
            },
            {
                "timestamp": "2026-01-01T12:11:00",
                "event_type": "app_switch",
                "app_name": "ChatGPT",
                "extra": {"relative_timestamp": 660.0},
            },
            {
                "timestamp": "2026-01-01T12:23:00",
                "event_type": "file_closed",
                "file_path": sensitive,
                "app_name": "Word",
                "extra": {"relative_timestamp": 1_380.0},
            },
        ]
    )

    windows = build_analysis_windows(logs, [sensitive], VisionConfig())
    assert windows == []


def test_sensitive_editing_and_blank_shell_do_not_create_window() -> None:
    sensitive = "C:/Users/alice/Documents/strategy.docx"
    logs = normalize_logs(
        [
            {
                "timestamp": "2026-01-01T12:00:00",
                "event_type": "file_open",
                "file_path": sensitive,
                "extra": {"relative_timestamp": 0.0},
            },
            {
                "timestamp": "2026-01-01T12:00:02",
                "event_type": "app_switch",
                "app_name": "Word",
                "window_info": {"window_title": "strategy.docx - Word"},
                "extra": {"relative_timestamp": 2.0},
            },
            {
                "timestamp": "2026-01-01T12:00:05",
                "event_type": "app_switch",
                "app_name": "File Explorer",
                "window_info": {"window_title": ""},
                "extra": {"relative_timestamp": 5.0},
            },
            {
                "timestamp": "2026-01-01T12:00:10",
                "event_type": "file_closed",
                "file_path": sensitive,
                "extra": {"relative_timestamp": 10.0},
            },
        ]
    )

    assert build_analysis_windows(logs, [sensitive], VisionConfig()) == []


def test_sensitive_editing_and_monitor_switch_do_not_create_window() -> None:
    sensitive = "C:/Users/alice/Documents/strategy.docx"
    logs = normalize_logs(
        [
            {
                "timestamp": "2026-01-01T12:00:00",
                "event_type": "file_open",
                "file_path": sensitive,
                "extra": {"relative_timestamp": 0.0},
            },
            {
                "timestamp": "2026-01-01T12:00:02",
                "event_type": "app_switch",
                "app_name": "Edge",
                "window_info": {"window_title": "Outlook - Edge"},
                "extra": {"relative_timestamp": 2.0},
            },
            {
                "timestamp": "2026-01-01T12:00:08",
                "event_type": "app_switch",
                "app_name": "python",
                "window_info": {"window_title": "DataLeakDetector Pro"},
                "extra": {"relative_timestamp": 8.0},
            },
            {
                "timestamp": "2026-01-01T12:00:10",
                "event_type": "file_closed",
                "file_path": sensitive,
                "extra": {"relative_timestamp": 10.0},
            },
        ]
    )

    assert build_analysis_windows(logs, [sensitive], VisionConfig()) == []


def test_sensitive_editing_and_wallpaper_switch_do_not_create_window() -> None:
    sensitive = "C:/Users/alice/Documents/strategy.docx"
    logs = normalize_logs(
        [
            {
                "timestamp": "2026-01-01T12:00:00",
                "event_type": "file_open",
                "file_path": sensitive,
                "extra": {"relative_timestamp": 0.0},
            },
            {
                "timestamp": "2026-01-01T12:00:02",
                "event_type": "app_switch",
                "app_name": "Word",
                "window_info": {"window_title": "strategy.docx - Word"},
                "extra": {"relative_timestamp": 2.0},
            },
            {
                "timestamp": "2026-01-01T12:00:08",
                "event_type": "app_switch",
                "app_name": "kwallpaper",
                "window_info": {"window_title": ""},
                "extra": {"relative_timestamp": 8.0},
            },
            {
                "timestamp": "2026-01-01T12:00:10",
                "event_type": "file_closed",
                "file_path": sensitive,
                "extra": {"relative_timestamp": 10.0},
            },
        ]
    )

    assert build_analysis_windows(logs, [sensitive], VisionConfig()) == []


def test_browser_cache_rename_is_not_promoted_to_sensitive_derivation() -> None:
    sensitive = "C:/Users/alice/Documents/strategy.pdf"
    logs = normalize_logs(
        [
            {
                "timestamp": "2026-01-01T12:00:00",
                "event_type": "opened",
                "file_path": sensitive,
                "extra": {"relative_timestamp": 0.0},
            },
            {
                "timestamp": "2026-01-01T12:00:05",
                "event_type": "renamed",
                "file_path": "C:/Users/alice/AppData/Local/Lenovo/SLBrowser/User Data/Default/LOG.old",
                "process_info": {"process_name": "SLBrowser.exe"},
                "window_info": {"window_title": "strategy.pdf - Browser"},
                "extra": {"raw_operation": "renamed", "relative_timestamp": 5.0},
            },
        ]
    )

    windows = build_analysis_windows(logs, [sensitive], VisionConfig())

    assert windows == []


def test_unclosed_sensitive_editing_does_not_create_window() -> None:
    sensitive = "C:/Users/alice/Documents/strategy.docx"
    logs = normalize_logs(
        [
            {
                "timestamp": "2026-01-01T12:00:00",
                "event_type": "file_open",
                "file_path": sensitive,
                "extra": {"relative_timestamp": 0.0},
            },
            {
                "timestamp": "2026-01-01T12:31:00",
                "event_type": "app_switch",
                "app_name": "ChatGPT",
                "extra": {"relative_timestamp": 1_860.0},
            },
        ]
    )

    assert build_analysis_windows(logs, [sensitive], VisionConfig()) == []


def test_unanchored_translation_does_not_create_a_visual_window() -> None:
    logs = normalize_logs(
        [
            {
                "timestamp": "2026-01-01T12:00:30",
                "event_type": "window_changed",
                "app_name": "Chrome",
                "process_info": {"process_name": "chrome.exe"},
                "window_info": {"window_title": "AI translation - online document"},
                "extra": {"raw_operation": "translate", "relative_timestamp": 30.0},
            }
        ]
    )

    windows = build_analysis_windows(logs, [], VisionConfig())

    assert windows == []


def test_isolated_clipboard_copy_does_not_create_visual_window() -> None:
    sensitive = "C:/Users/alice/Desktop/secret.docx"
    logs = normalize_logs(
        [
            {
                "timestamp": "2026-01-01T12:00:00",
                "event_type": "file_open",
                "file_path": sensitive,
                "end_time": "2026-01-01T12:01:00",
                "extra": {"relative_timestamp": 0.0},
                "process_info": {"process_name": "WINWORD.EXE"},
            },
            {
                "timestamp": "2026-01-01T12:00:30",
                "event_type": "clipboard_text",
                "content_preview": "copied confidential paragraph",
                "extra": {"raw_operation": "clipboard_text", "relative_timestamp": 30.0},
                "process_info": {"process_name": "WINWORD.EXE"},
            },
            {
                "timestamp": "2026-01-01T12:01:30",
                "event_type": "clipboard_text",
                "content_preview": "copied unrelated text",
                "extra": {"raw_operation": "clipboard_text", "relative_timestamp": 90.0},
                "process_info": {"process_name": "msedge.exe"},
            },
        ]
    )

    windows = build_analysis_windows(logs, [sensitive], VisionConfig(include_unanchored_medium_windows=True))

    assert windows == []


def test_screenshot_file_anchor_requires_active_sensitive_context() -> None:
    sensitive = "C:/Users/alice/Desktop/secret.docx"
    screenshot = "C:/Users/alice/Pictures/Screenshots/screenshot.png"
    logs = normalize_logs(
        [
            {
                "timestamp": "2026-01-01T12:00:00",
                "event_type": "file_open",
                "file_path": sensitive,
                "end_time": "2026-01-01T12:00:20",
                "extra": {"relative_timestamp": 0.0},
                "process_info": {"process_name": "WINWORD.EXE"},
            },
            {
                "timestamp": "2026-01-01T12:00:10",
                "event_type": "modified",
                "file_path": screenshot,
                "extra": {"raw_operation": "modified", "relative_timestamp": 10.0},
                "process_info": {"process_name": "SnippingTool.exe"},
            },
            {
                "timestamp": "2026-01-01T12:00:40",
                "event_type": "modified",
                "file_path": "C:/Users/alice/Pictures/Screenshots/later.png",
                "extra": {"raw_operation": "modified", "relative_timestamp": 40.0},
                "process_info": {"process_name": "SnippingTool.exe"},
            },
        ]
    )

    windows = build_analysis_windows(logs, [sensitive], VisionConfig(include_unanchored_medium_windows=True))

    assert any(window.priority == "strong" and 10_000 in window.anchor_ms for window in windows)
    assert not any(window.priority == "strong" and 40_000 in window.anchor_ms for window in windows)


def test_screenshot_tool_cache_file_does_not_become_capture_anchor() -> None:
    sensitive = "C:/Users/alice/Desktop/secret.docx"
    logs = normalize_logs(
        [
            {
                "timestamp": "2026-01-01T12:00:00",
                "event_type": "file_open",
                "file_path": sensitive,
                "end_time": "2026-01-01T12:00:20",
                "extra": {"relative_timestamp": 0.0},
                "process_info": {"process_name": "WINWORD.EXE"},
            },
            {
                "timestamp": "2026-01-01T12:00:10",
                "event_type": "modified",
                "file_path": "C:/Users/alice/AppData/Local/SnippingTool/Cache/data_2",
                "window_info": {"window_title": "鎴浘宸ュ叿瑕嗙洊"},
                "extra": {"raw_operation": "modified", "relative_timestamp": 10.0},
                "process_info": {"process_name": "SnippingTool.exe"},
            },
        ]
    )

    windows = build_analysis_windows(logs, [sensitive], VisionConfig(include_unanchored_medium_windows=True))

    assert not any(window.priority == "strong" and 10_000 in window.anchor_ms for window in windows)


def test_sink_file_dialog_screenshot_is_file_selection_not_capture() -> None:
    logs = normalize_logs(
        [
            {
                "timestamp": "2026-01-01T12:00:10",
                "event_type": "modified",
                "file_path": "C:/Users/alice/Pictures/Screenshots/screenshot.png",
                "app_name": "QQ",
                "process_info": {"process_name": "QQ.exe"},
                "window_info": {"window_title": "请选择"},
                "extra": {"source": "watchdog_fs_monitor", "relative_timestamp": 10.0},
            }
        ]
    )

    windows = build_analysis_windows(logs, [], VisionConfig())

    assert len(windows) == 1
    assert windows[0].action_phases == ((10_000, "file_selected"),)
    assert windows[0].end_ms == 40_000


def test_browser_modifying_old_screenshot_is_not_new_capture() -> None:
    logs = normalize_logs(
        [
            {
                "timestamp": "2026-01-01T12:00:10",
                "event_type": "modified",
                "file_path": "C:/Users/alice/Pictures/Screenshots/old.png",
                "app_name": "Edge",
                "process_info": {"process_name": "msedge.exe"},
                "window_info": {"window_title": "AI chat"},
                "extra": {"source": "watchdog_fs_monitor", "relative_timestamp": 10.0},
            }
        ]
    )

    assert build_analysis_windows(logs, [], VisionConfig()) == []


def test_sink_cache_write_does_not_impersonate_a_file_selection_event() -> None:
    logs = normalize_logs(
        [
            {
                "timestamp": "2026-01-01T12:00:34.280",
                "event_type": "modified",
                "file_path": "C:/Users/alice/AppData/Roaming/Tencent/QQ/misc/cache.dat",
                "process_info": {"process_name": "QQ.exe"},
                "window_info": {"window_title": "请选择"},
                "extra": {"source": "etw_file_monitor", "relative_timestamp": 34.28},
            }
        ]
    )

    windows = build_analysis_windows(logs, ["C:/Users/alice/Desktop/secret.docx"], VisionConfig())

    assert windows == []


def test_dense_sink_cache_writes_do_not_create_visual_windows() -> None:
    logs = normalize_logs(
        [
            {
                "timestamp": "2026-01-01T12:00:00",
                "event_type": "modified",
                "file_path": f"C:/Users/alice/AppData/Roaming/Tencent/WeChat/cache_{index}.dat",
                "process_info": {"process_name": "WeChat.exe"},
                "window_info": {"window_title": "打开"},
                "extra": {"source": "etw_file_monitor", "relative_timestamp": timestamp / 1000},
            }
            for index, timestamp in enumerate((13_570, 13_589, 13_624, 13_655, 13_675, 14_609, 15_963, 17_000, 18_600, 24_000))
        ]
    )

    windows = build_analysis_windows(logs, ["C:/Users/alice/Desktop/secret.docx"], VisionConfig())

    assert windows == []


def test_external_file_dialog_switch_is_not_an_action_anchor() -> None:
    logs = normalize_logs(
        [
            {
                "timestamp": "2026-01-01T12:00:17.525",
                "event_type": "app_switch",
                "file_path": "",
                "app_name": "鐧惧害缃戠洏",
                "process_info": {"process_name": "BaiduNetdiskUnite.exe"},
                "window_info": {"window_title": "请选择文件/文件夹"},
                "extra": {"source": "window_monitor", "category": "缃戠洏", "relative_timestamp": 17.525},
            }
        ]
    )

    windows = build_analysis_windows(logs, [], VisionConfig())

    assert windows == []


def test_browser_file_dialog_switch_is_not_an_action_anchor() -> None:
    logs = normalize_logs(
        [
            {
                "timestamp": "2026-01-01T12:00:15.919",
                "event_type": "app_switch",
                "file_path": "",
                "app_name": "Chrome",
                "process_info": {"process_name": "chrome.exe"},
                "window_info": {"window_title": "夸克网盘 - Google Chrome"},
                "extra": {"source": "window_monitor", "category": "浏览器", "relative_timestamp": 15.919},
            },
            {
                "timestamp": "2026-01-01T12:00:18.925",
                "event_type": "app_switch",
                "file_path": "",
                "app_name": "Chrome",
                "process_info": {"process_name": "chrome.exe"},
                "window_info": {"window_title": "打开"},
                "extra": {"source": "window_monitor", "category": "浏览器", "relative_timestamp": 18.925},
            },
        ]
    )

    windows = build_analysis_windows(logs, [], VisionConfig())

    assert windows == []


def test_workspace_open_dialog_switch_is_not_an_action_anchor() -> None:
    logs = normalize_logs(
        [
            {
                "timestamp": "2026-01-01T12:00:10.644",
                "event_type": "app_switch",
                "file_path": "",
                "app_name": "Lark",
                "process_info": {"process_name": "Lark.exe"},
                "window_info": {"window_title": "Open"},
                "extra": {"source": "window_monitor", "category": "workplace", "relative_timestamp": 10.644},
            }
        ]
    )

    windows = build_analysis_windows(logs, [], VisionConfig())

    assert windows == []


def test_meeting_aliases_form_one_packet_and_keep_each_visual_state() -> None:
    sensitive = "D:/DataLeakTest/docx/company_contract.docx"
    logs = normalize_logs(
        [
            {
                "timestamp": "2026-01-01T12:00:03",
                "event_type": "app_switch",
                "app_name": "Edge",
                "window_info": {"window_title": "Google Meet"},
                "extra": {"relative_timestamp": 3.0},
            },
            {
                "timestamp": "2026-01-01T12:00:09",
                "event_type": "app_switch",
                "app_name": "Edge",
                "window_info": {"window_title": "Meet"},
                "extra": {"relative_timestamp": 9.0},
            },
            {
                "timestamp": "2026-01-01T12:00:23",
                "event_type": "app_switch",
                "app_name": "Edge",
                "window_info": {"window_title": "meet.google.com 正在共享你的屏幕。"},
                "extra": {"relative_timestamp": 23.0},
            },
            {
                "timestamp": "2026-01-01T12:00:37",
                "event_type": "modified",
                "file_path": sensitive,
                "process_info": {"process_name": "wps.exe"},
                "extra": {"relative_timestamp": 37.0},
            },
            {
                "timestamp": "2026-01-01T12:01:06",
                "event_type": "app_switch",
                "app_name": "Win Monitor",
                "window_info": {"window_title": "Win Monitor - 数据泄露行为监控"},
                "extra": {"relative_timestamp": 66.0},
            },
        ]
    )

    windows = build_analysis_windows(logs, [sensitive], VisionConfig())
    meeting = next(window for window in windows if window.reason == "strong:external_session:meeting")

    assert meeting.start_ms == 3_000
    assert meeting.end_ms == 53_000
    assert (9_000, "external_state") in meeting.action_phases
    assert (23_000, "external_state") in meeting.action_phases
    assert {3_000, 9_000, 23_000, 53_000}.issubset(_probe_timestamps(meeting, VisionConfig()))


def test_clipboard_with_recent_sensitive_context_scopes_later_external_session() -> None:
    sensitive = "C:/Users/alice/Desktop/strategy.docx"
    logs = normalize_logs(
        [
            {
                "timestamp": "2026-01-01T12:00:10",
                "event_type": "opened",
                "file_path": sensitive,
                "process_info": {"process_name": "WINWORD.EXE"},
                "extra": {"relative_timestamp": 10.0},
            },
            {
                "timestamp": "2026-01-01T12:00:30",
                "event_type": "clipboard_text",
                "content_preview": "sensitive strategy excerpt",
                "process_info": {"process_name": "WINWORD.EXE"},
                "extra": {"source": "clipboard_monitor", "relative_timestamp": 30.0},
            },
            {
                "timestamp": "2026-01-01T12:00:39",
                "event_type": "app_switch",
                "app_name": "Doubao",
                "process_info": {"process_name": "Doubao.exe"},
                "window_info": {"window_title": "Doubao AI chat"},
                "extra": {
                    "source": "window_monitor",
                    "category": "AI",
                    "risk_level": "high",
                    "relative_timestamp": 39.0,
                },
            },
        ]
    )

    windows = build_analysis_windows(logs, [sensitive], VisionConfig())
    anchors = {anchor for window in windows if window.priority == "strong" for anchor in window.action_anchor_ms}

    assert {30_000, 39_000}.issubset(anchors)
    assert any((30_000, "clipboard") in window.action_phases for window in windows)
    assert any((39_000, "external_session") in window.action_phases for window in windows)


def test_external_session_keeps_source_and_full_business_session() -> None:
    sensitive = "D:/work/员工薪资明细表.xlsx"
    logs = normalize_logs(
        [
            {
                "timestamp": "2026-01-01T12:00:30",
                "event_type": "created",
                "file_path": "D:/work/员工薪资明细表/高管薪资.xlsx",
                "extra": {"raw_operation": "created", "relative_timestamp": 30.0},
            },
            {
                "timestamp": "2026-01-01T12:00:52",
                "event_type": "app_switch",
                "app_name": "Edge",
                "window_info": {"window_title": "网易邮箱 - 搜索"},
                "extra": {"relative_timestamp": 52.0},
            },
            {
                "timestamp": "2026-01-01T12:01:17",
                "event_type": "app_switch",
                "app_name": "Edge",
                "window_info": {"window_title": "网易邮箱6.0"},
                "extra": {"relative_timestamp": 77.0},
            },
            {
                "timestamp": "2026-01-01T12:01:46",
                "event_type": "app_switch",
                "app_name": "Edge",
                "window_info": {"window_title": "写邮件 - 网易邮箱"},
                "extra": {"relative_timestamp": 106.0},
            },
        ]
    )

    windows = build_analysis_windows(logs, [sensitive], VisionConfig())
    outbound = next(window for window in windows if any(action == "external_session" for _, action in window.action_phases))

    assert (30_000, "derive") in outbound.action_phases
    assert (52_000, "external_session") in outbound.action_phases
    assert any(action == "session_end" and timestamp >= 106_000 for timestamp, action in outbound.action_phases)
    assert {30_000, 52_000, 106_000}.issubset(_probe_timestamps(outbound, VisionConfig()))


def test_external_mail_session_survives_sensitive_file_explorer_switch() -> None:
    source = "D:/work/员工薪资明细表.xlsx"
    executive = "D:/work/员工薪资明细表/高管薪资.xlsx"
    regular = "D:/work/员工薪资明细表/普通员工薪资.xlsx"
    logs = normalize_logs(
        [
            {
                "timestamp": "2026-01-01T12:00:30",
                "event_type": "created",
                "file_path": executive,
                "app_name": "WPS Excel",
                "extra": {"raw_operation": "created", "relative_timestamp": 30.0},
            },
            {
                "timestamp": "2026-01-01T12:00:31",
                "event_type": "created",
                "file_path": regular,
                "app_name": "WPS Excel",
                "extra": {"raw_operation": "created", "relative_timestamp": 31.0},
            },
            {
                "timestamp": "2026-01-01T12:00:52",
                "event_type": "app_switch",
                "app_name": "Edge",
                "window_info": {"window_title": "网易邮箱 - 搜索"},
                "extra": {"relative_timestamp": 52.0},
            },
            {
                "timestamp": "2026-01-01T12:01:17",
                "event_type": "app_switch",
                "app_name": "Edge",
                "window_info": {"window_title": "网易邮箱6.0"},
                "extra": {"relative_timestamp": 77.0},
            },
            {
                "timestamp": "2026-01-01T12:01:47",
                "event_type": "app_switch",
                "app_name": "explorer",
                "window_info": {"window_title": "员工薪资明细表 - 文件资源管理器"},
                "extra": {"relative_timestamp": 107.0},
            },
            {
                "timestamp": "2026-01-01T12:01:51",
                "event_type": "modified",
                "file_path": executive,
                "app_name": "WPS Excel",
                "extra": {"raw_operation": "modified", "relative_timestamp": 111.0},
            },
            {
                "timestamp": "2026-01-01T12:02:09",
                "event_type": "app_switch",
                "app_name": "explorer",
                "window_info": {"window_title": "员工薪资明细表 - 文件资源管理器"},
                "extra": {"relative_timestamp": 129.0},
            },
            {
                "timestamp": "2026-01-01T12:02:11",
                "event_type": "modified",
                "file_path": regular,
                "app_name": "WPS Excel",
                "extra": {"raw_operation": "modified", "relative_timestamp": 131.0},
            },
        ]
    )

    windows = build_analysis_windows(logs, [source, executive, regular], VisionConfig())
    outbound = next(
        window
        for window in windows
        if any(action == "external_session" for _, action in window.action_phases)
        and (131_000, "file_selected") in window.action_phases
    )

    assert any(action == "external_session" for _, action in outbound.action_phases)
    assert (111_000, "file_selected") in outbound.action_phases
    assert (131_000, "file_selected") in outbound.action_phases


def test_external_session_promotes_later_sensitive_file_accesses() -> None:
    original = "D:/work/employee_salary.xlsx"
    derived = "D:/work/employee_salary/high_salary.xlsx"
    logs = normalize_logs(
        [
            {
                "timestamp": "2026-01-01T12:00:10",
                "event_type": "created",
                "file_path": derived,
                "extra": {"raw_operation": "created", "relative_timestamp": 10.0},
            },
            {
                "timestamp": "2026-01-01T12:00:20",
                "event_type": "app_switch",
                "app_name": "Edge",
                "window_info": {"window_title": "mail compose"},
                "extra": {"relative_timestamp": 20.0},
            },
            {
                "timestamp": "2026-01-01T12:00:30",
                "event_type": "app_switch",
                "app_name": "Edge",
                "window_info": {"window_title": "mail compose"},
                "extra": {"relative_timestamp": 30.0},
            },
            {
                "timestamp": "2026-01-01T12:00:34",
                "event_type": "modified",
                "file_path": derived,
                "app_name": "WPS",
                "extra": {"raw_operation": "modified", "relative_timestamp": 34.0},
            },
        ]
    )

    windows = build_analysis_windows(logs, [original], VisionConfig())

    outbound = next(window for window in windows if any(action == "external_session" for _, action in window.action_phases))
    assert (10_000, "derive") in outbound.action_phases
    assert (20_000, "external_session") in outbound.action_phases
    assert (34_000, "file_selected") in outbound.action_phases


def test_usb_detection_does_not_match_statusbar_appseq_path() -> None:
    logs = normalize_logs(
        [
            {
                "timestamp": "2026-01-01T12:00:00",
                "event_type": "modified",
                "file_path": "C:/Users/alice/AppData/Roaming/Kingsoft/office6/statusbarAppSeq/1706647148/appSeq.json",
                "app_name": "WPS",
                "extra": {"raw_operation": "modified", "relative_timestamp": 0.0},
            }
        ]
    )

    assert _action_kind(logs[0]) == ""


def test_default_log_miner_keeps_in_memory_window_contract() -> None:
    records = [
        {
            "timestamp": "2026-01-01T12:00:00",
            "event_type": "file_selected",
            "file_path": "C:/Users/alice/Documents/secret.docx",
            "window_info": {"window_title": "Upload files - Unknown Cloud"},
            "extra": {"raw_operation": "file_selected", "relative_timestamp": 60.0},
        }
    ]
    logs = normalize_logs(records)

    result = mine_analysis_windows(
        case_id="unit",
        log_file="logs.json",
        records=records,
        logs=logs,
        sensitive_files=["C:/Users/alice/Documents/secret.docx"],
        vision_config=VisionConfig(),
        neo4j_log_miner=False,
    )

    assert result.source == "in_memory"
    assert result.windows[0].priority == "strong"
    assert result.windows[0].start_ms == 55_000
    assert result.metadata["neo4j_enabled"] is False


def test_compact_event_view_keeps_action_evidence_without_filesystem_noise() -> None:
    sensitive = ("c:/users/alice/documents/secret.docx",)
    records = [
        {
            "timestamp": "2026-01-01T00:00:00",
            "event_type": "opened",
            "file_path": "C:/Users/alice/Documents/secret.docx",
            "extra": {"relative_timestamp": 0.0},
        },
        *[
            {
                "timestamp": f"2026-01-01T00:00:{index % 60:02d}",
                "event_type": "modified" if index % 2 else "opened",
                "file_path": f"C:/Users/alice/AppData/Local/Temp/noise-{index}.tmp",
                "extra": {"relative_timestamp": 1.0 + index / 1000},
            }
            for index in range(200)
        ],
        {
            "timestamp": "2026-01-01T00:00:03",
            "event_type": "file_selected",
            "file_path": "C:/Users/alice/Documents/secret.docx",
            "process_info": {"process_name": "chrome.exe"},
            "window_info": {"window_title": "Open"},
            "extra": {"relative_timestamp": 3.0, "raw_operation": "file_selected"},
        },
        {
            "timestamp": "2026-01-01T00:00:04",
            "event_type": "clipboard_text",
            "content_preview": "secret excerpt",
            "extra": {"relative_timestamp": 4.0},
        },
        {
            "timestamp": "2026-01-01T00:00:05",
            "event_type": "send",
            "process_info": {"process_name": "wechat.exe"},
            "window_info": {"window_title": "Send file"},
            "extra": {"relative_timestamp": 5.0},
        },
    ]
    logs = normalize_logs(records)
    candidates = [event for event in logs if _may_need_analysis_window(event, sensitive)]

    compact = _compact_event_view(logs, candidates)
    compact_types = [event.event_type for event in compact]
    compact_paths = {normalize_path(event.file_path).lower() for event in compact}

    assert len(compact) < len(logs)
    assert "file_selected" in compact_types
    assert "clipboard_text" in compact_types
    assert "send" in compact_types
    assert "c:/users/alice/documents/secret.docx" in compact_paths
    assert not any("noise-" in path for path in compact_paths)


def test_neo4j_log_miner_graph_event_payload_is_case_scoped() -> None:
    logs = normalize_logs(
        [
            {
                "timestamp": "2026-01-01T12:00:00",
                "event_type": "opened",
                "file_path": "C:/Users/alice/Documents/secret.docx",
                "process_info": {"process_name": "WINWORD.EXE"},
            }
        ]
    )

    events = records_to_graph_events("case_a", logs, ["C:/Users/alice/Documents/secret.docx"])

    assert events[0]["id"] == "case_a:event:log_0"
    assert events[0]["file_id"].startswith("case_a:file:")
    assert events[0]["file_path_lower"].endswith("secret.docx")
    assert events[0]["is_sensitive_related"] is True
    assert events[0]["is_candidate"] is True
    assert len(fingerprint_records([logs[0].raw])) == 64


def test_neo4j_log_miner_does_not_promote_risky_app_without_activity() -> None:
    logs = normalize_logs(
        [
            {
                "timestamp": "2026-01-01T12:00:00",
                "event_type": "app_switch",
                "file_path": "",
                "app_name": "Edge",
                "window_info": {"window_title": "New tab"},
            }
        ]
    )

    events = records_to_graph_events("case_a", logs, [])

    assert events[0]["is_risky_app"] is True
    assert events[0]["is_candidate"] is False


def test_bluetooth_transfer_window_keeps_separate_strong_budget() -> None:
    logs = normalize_logs(
        [
            {
                "timestamp": "2026-01-01T12:00:00",
                "event_type": "opened",
                "file_path": "C:/Users/alice/Desktop/secret.docx",
                "process_info": {"process_name": "WINWORD.EXE"},
                "extra": {"source": "etw_file_monitor"},
            },
            {
                "timestamp": "2026-01-01T12:01:00",
                "event_type": "app_switch",
                "file_path": "",
                "process_info": {"process_name": "fsquirt.exe"},
                "window_info": {"window_title": "蓝牙文件传送"},
                "extra": {"source": "window_monitor"},
            },
        ]
    )

    windows = build_analysis_windows(
        logs,
        ["C:/Users/alice/Desktop/secret.docx"],
        VisionConfig(include_unanchored_medium_windows=True),
    )

    assert [window.priority for window in windows] == ["strong"]
    assert windows[0].start_ms == 55_000
    assert windows[0].end_ms == 90_000
    assert windows[0].max_keyframes == VisionConfig().max_keyframes_per_strong_window
    assert 60_000 in windows[0].anchor_ms


def test_bluetooth_sensitive_file_access_becomes_keyframe_anchor() -> None:
    logs = normalize_logs(
        [
            {
                "timestamp": "2026-01-01T12:02:21",
                "event_type": "opened",
                "file_path": "C:/Users/alice/Desktop/secret.docx",
                "process_info": {"process_name": "fsquirt.exe"},
                "window_info": {"window_title": "蓝牙文件传送"},
                "extra": {"source": "etw_file_monitor"},
            }
        ]
    )

    windows = build_analysis_windows(logs, ["C:/Users/alice/Desktop/secret.docx"], VisionConfig())

    assert windows[0].priority == "strong"
    assert windows[0].anchor_ms == (0,)
    assert windows[0].action_anchor_ms == (0,)


def test_dense_sensitive_file_events_do_not_create_visual_windows() -> None:
    records = [
        {
            "timestamp": "2026-01-01T12:00:00",
            "event_type": "opened",
            "file_path": "C:/Users/alice/Desktop/customer_salary.xlsx",
            "extra": {"relative_timestamp": float(second)},
        }
        for second in range(0, 1_201, 30)
    ]
    logs = normalize_logs(records)

    windows = build_analysis_windows(
        logs,
        ["C:/Users/alice/Desktop/customer_salary.xlsx"],
        VisionConfig(case_segment_ms=300_000, include_unanchored_medium_windows=True),
    )
    assert windows == []


def test_merged_window_budget_keeps_all_log_anchors() -> None:
    windows = [
        AnalysisWindow(0, 10_000, "a", priority="medium", max_keyframes=2, anchor_ms=(1_000, 2_000)),
        AnalysisWindow(5_000, 15_000, "b", priority="medium", max_keyframes=2, anchor_ms=(6_000, 7_000, 8_000)),
    ]

    merged = merge_analysis_windows(windows)

    assert merged[0].anchor_ms == (1_000, 2_000, 6_000, 7_000, 8_000)
    assert merged[0].max_keyframes == 5






def test_weak_analysis_windows_are_opt_in() -> None:
    logs = normalize_logs(
        [
            {
                "timestamp": "2026-01-01T12:00:00",
                "event_type": "app_switch",
                "app_name": "Edge",
                "extra": {"risk_level": "high", "relative_timestamp": 1.0},
            }
        ]
    )

    assert build_analysis_windows(logs, [], VisionConfig()) == []
    assert build_analysis_windows(logs, [], VisionConfig(include_weak_windows=True)) == []


def test_created_document_under_sensitive_source_folder_is_derivation() -> None:
    sensitive = "D:/work/员工薪资明细表.xlsx"
    logs = normalize_logs(
        [
            {
                "timestamp": "2026-01-01T12:00:00",
                "event_type": "modified",
                "file_path": sensitive,
                "process_info": {"process_name": "et.exe"},
                "extra": {"relative_timestamp": 0.0},
            },
            {
                "timestamp": "2026-01-01T12:00:30",
                "event_type": "created",
                "file_path": "D:/work/员工薪资明细表/高管薪资.xlsx",
                "process_info": {"process_name": "et.exe"},
                "extra": {"raw_operation": "created", "relative_timestamp": 30.0},
            },
        ]
    )

    windows = build_analysis_windows(logs, [sensitive], VisionConfig())

    assert any(window.priority == "strong" and 30_000 in window.action_anchor_ms for window in windows)






def test_app_switch_without_sensitive_or_risk_context_does_not_create_window() -> None:
    logs = normalize_logs(
        [
            {
                "timestamp": "2026-01-01T00:00:30",
                "event_type": "app_switch",
                "process_info": {"process_name": "chrome.exe"},
                "window_info": {"window_title": "mail.163.com"},
                "extra": {"source": "window_monitor", "category": "浏览器"},
            }
        ],
        session_start_ms=1_767_225_570_000,
    )

    windows = build_analysis_windows(logs, [], VisionConfig())

    assert windows == []


def test_risk_window_keeps_external_session_context_around_file_selection() -> None:
    logs = normalize_logs(
        [
            {
                "timestamp": "2026-01-01T00:00:30",
                "event_type": "app_switch",
                "process_info": {"process_name": "QQ.exe"},
                "window_info": {"window_title": "QQ"},
                "extra": {"source": "window_monitor", "category": "鍗虫椂閫氳", "relative_timestamp": 30.0},
            },
            {
                "timestamp": "2026-01-01T00:00:32",
                "event_type": "app_switch",
                "process_info": {"process_name": "QQ.exe"},
                "window_info": {"window_title": "QQ"},
                "extra": {"source": "window_monitor", "category": "鍗虫椂閫氳", "relative_timestamp": 32.0},
            },
            {
                "timestamp": "2026-01-01T00:00:35",
                "event_type": "app_switch",
                "process_info": {"process_name": "QQ.exe"},
                "window_info": {"window_title": "QQ"},
                "extra": {"source": "window_monitor", "category": "鍗虫椂閫氳", "relative_timestamp": 35.0},
            },
            {
                "timestamp": "2026-01-01T00:01:00",
                "event_type": "file_selected",
                "file_path": "C:/Users/alice/Documents/secret.docx",
                "window_info": {"window_title": "Upload files - Unknown Cloud"},
                "extra": {
                    "source": "file_dialog_monitor",
                    "category": "文件上传",
                    "raw_operation": "file_selected",
                    "relative_timestamp": 60.0,
                },
            },
        ]
    )

    windows = build_analysis_windows(logs, [], VisionConfig())

    assert [window.priority for window in windows] == ["strong"]
    assert (30_000, "external_session") in windows[0].action_phases
    assert (60_000, "file_selected") in windows[0].action_phases
    assert windows[0].active_apps == ("qq",)


def test_unanchored_windows_use_three_temporal_coverage_targets() -> None:
    window = AnalysisWindow(0, 156_000, "medium", priority="medium", step_ms=1_000, max_keyframes=18)

    coverage = _coverage_timestamps(window)

    assert coverage == (0, 78_000, 156_000)


def test_sensitive_activity_windows_do_not_force_uniform_coverage_frames() -> None:
    window = AnalysisWindow(0, 156_000, "sensitive_activity:secret.docx", priority="activity", max_keyframes=18)

    assert _coverage_timestamps(window) == ()


def test_anchored_strong_window_prioritizes_risk_anchors_over_coverage() -> None:
    window = AnalysisWindow(0, 60_000, "upload", priority="strong", step_ms=250, anchor_ms=(10_000, 40_000))

    assert _coverage_timestamps(window) == ()
    probes = _probe_timestamps(window, VisionConfig())
    assert {9_750, 10_000, 10_250, 39_750, 40_000, 40_250}.issubset(probes)
    assert {0, 60_000}.issubset(probes)


def test_paste_action_probes_result_state() -> None:
    window = AnalysisWindow(
        5_000,
        40_000,
        "strong:file_selected:file_dialog_monitor",
        priority="strong",
        step_ms=250,
        max_keyframes=8,
        anchor_ms=(10_000,),
        action_anchor_ms=(10_000,),
        requires_post_action_state=True,
    )

    probes = _probe_timestamps(window, VisionConfig())

    assert {8_000, 10_000, 12_000, 15_000, 20_000}.issubset(probes)
    assert 40_000 in probes


def test_paste_phase_keeps_pre_action_submission_and_result_states() -> None:
    np = pytest.importorskip("numpy")
    gray = np.zeros((90, 160), dtype=np.uint8)
    candidates = [
        _FrameCandidate(
            KeyFrame(frame_id, timestamp, f"{frame_id}.jpg", 0.1, "strong:action_state:paste"),
            "strong",
            gray,
            (0, 64),
        )
        for frame_id, timestamp in (
            ("before", 8_000),
            ("at_action", 10_000),
            ("just_pasted", 12_000),
            ("settled", 15_000),
            ("stable", 25_000),
        )
    ]
    window = AnalysisWindow(
        5_000,
        30_000,
        "strong:paste",
        priority="strong",
        action_anchor_ms=(10_000,),
        action_phases=((10_000, "paste"),),
    )

    focused = _focus_semantic_action_phases(candidates, window)

    assert [item.frame.frame_id for item in focused] == ["before", "at_action", "just_pasted", "stable"]


def test_mandatory_budget_keeps_identity_and_result_for_each_repeated_selection() -> None:
    np = pytest.importorskip("numpy")
    gray = np.zeros((90, 160), dtype=np.uint8)
    candidates = [
        _FrameCandidate(
            KeyFrame(frame_id, timestamp, f"{frame_id}.jpg", 0.1, f"strong:action_state:file_selected:{role}"),
            "strong",
            gray,
            (0, 64),
        )
        for frame_id, timestamp, role in (
            ("first_early", 8_000, "pre"),
            ("first_late", 9_900, "pre"),
            ("first_at", 10_000, "at"),
            ("first_post", 12_000, "post"),
            ("first_result", 20_000, "result"),
            ("second_early", 48_000, "pre"),
            ("second_late", 49_900, "pre"),
            ("second_at", 50_000, "at"),
            ("second_post", 52_000, "post"),
            ("second_result", 60_000, "result"),
        )
    ]
    window = AnalysisWindow(
        5_000,
        80_000,
        "strong:external_session:mail",
        priority="strong",
        max_keyframes=8,
        action_phases=((10_000, "file_selected"), (50_000, "file_selected")),
    )

    selected = _trim_mandatory_evidence(candidates, window, 8)

    assert {item.frame.frame_id for item in selected} == {
        "first_early",
        "first_late",
        "first_at",
        "first_result",
        "second_early",
        "second_late",
        "second_at",
        "second_result",
    }


def test_window_budget_reserves_visual_completion_over_repeated_context_states() -> None:
    np = pytest.importorskip("numpy")
    gray = np.zeros((90, 160), dtype=np.uint8)

    def candidate(frame_id: str, timestamp: int, score: float, reason: str) -> _FrameCandidate:
        return _FrameCandidate(KeyFrame(frame_id, timestamp, f"{frame_id}.jpg", score, reason), "strong", gray, (0, 64))

    candidates = [
        candidate("session", 2_000, 0.01, "strong:action_state:external_session:at"),
        candidate("state_1", 7_000, 0.01, "strong:action_state:external_state:at"),
        candidate("selection_pre", 8_000, 0.01, "strong:action_state:file_selected:pre"),
        candidate("selection_at", 10_000, 0.01, "strong:action_state:file_selected:at"),
        candidate("selection_post", 12_000, 0.02, "strong:action_state:file_selected:post"),
        candidate("state_2", 16_000, 0.01, "strong:action_state:external_state:at"),
        candidate("state_3", 20_000, 0.01, "strong:action_state:external_state:at"),
        candidate("upload_complete", 21_000, 0.8, "strong:visual_change"),
        candidate("uploaded_file", 22_000, 0.6, "strong:visual_change"),
        candidate("selection_result", 30_000, 0.01, "strong:action_state:file_selected:result"),
        candidate("state_4", 35_000, 0.01, "strong:action_state:external_state:at"),
        candidate("end", 39_000, 0.01, "strong:action_state:session_end:at"),
    ]
    window = AnalysisWindow(
        0,
        40_000,
        "strong:external_session:cloud_drive",
        priority="strong",
        max_keyframes=8,
        action_phases=(
            (2_000, "external_session"),
            (7_000, "external_state"),
            (10_000, "file_selected"),
            (16_000, "external_state"),
            (20_000, "external_state"),
            (35_000, "external_state"),
            (39_000, "session_end"),
        ),
    )

    selected = _budget_window_candidates(candidates, window, 8)
    selected_ids = {item.frame.frame_id for item in selected}

    assert len(selected) == 8
    assert {"upload_complete", "uploaded_file", "selection_at", "selection_result"}.issubset(selected_ids)
    assert len(selected_ids & {"session", "state_1", "state_2", "state_3", "state_4", "end"}) <= 3


def test_context_budget_prefers_latest_external_state_over_generic_session_end() -> None:
    np = pytest.importorskip("numpy")
    gray = np.zeros((90, 160), dtype=np.uint8)

    def candidate(frame_id: str, timestamp: int, reason: str) -> _FrameCandidate:
        frame = KeyFrame(frame_id, timestamp, f"{frame_id}.jpg", 0.01, reason)
        return _FrameCandidate(frame, "strong", gray, (0, 64))

    candidates = [
        candidate("session", 5_838, "strong:action_state:external_session:at"),
        candidate("early", 7_399, "strong:action_state:external_state:at"),
        candidate("middle", 22_932, "strong:action_state:external_state:at"),
        candidate("sent", 40_392, "strong:action_state:external_state:at"),
        candidate("end", 50_739, "strong:action_state:session_end:at"),
    ]

    selected = _select_context_evidence(candidates, 2)

    assert [item.frame.frame_id for item in selected] == ["session", "sent"]


def test_external_state_keeps_immediate_visual_completion_evidence() -> None:
    np = pytest.importorskip("numpy")
    gray = np.zeros((90, 160), dtype=np.uint8)

    def candidate(frame_id: str, timestamp: int, score: float) -> _FrameCandidate:
        frame = KeyFrame(frame_id, timestamp, f"{frame_id}.jpg", score, "strong:visual_change")
        return _FrameCandidate(frame, "strong", gray, (0, 64))

    candidates = [
        candidate("before_send", 40_142, 0.1),
        candidate("sent_file_card", 41_785, 0.8),
        candidate("unrelated_later", 47_754, 0.9),
    ]
    window = AnalysisWindow(
        0,
        50_739,
        "strong:external_session:chat",
        priority="strong",
        action_phases=((40_392, "external_state"),),
    )

    selected = _post_action_visual_evidence(candidates, window, 2)

    assert [item.frame.frame_id for item in selected] == ["sent_file_card"]


def test_window_budget_keeps_clipboard_paste_and_submitted_visual_states() -> None:
    np = pytest.importorskip("numpy")
    gray = np.zeros((90, 160), dtype=np.uint8)

    def candidate(frame_id: str, timestamp: int, score: float, reason: str) -> _FrameCandidate:
        return _FrameCandidate(KeyFrame(frame_id, timestamp, f"{frame_id}.jpg", score, reason), "strong", gray, (0, 64))

    candidates = [
        candidate("copy", 10_000, 0.1, "strong:action_state:clipboard:at"),
        candidate("post_copy", 12_000, 0.1, "strong:action_state:clipboard:post"),
        candidate("chat", 15_000, 0.5, "strong:visual_change"),
        candidate("paste_menu", 25_000, 0.01, "strong:action_state:clipboard:result"),
        candidate("pasted", 25_400, 0.8, "strong:visual_change"),
        candidate("submitted", 27_000, 0.7, "strong:visual_change"),
        candidate("external", 30_000, 0.01, "strong:action_state:external_state:at"),
        candidate("end", 39_000, 0.01, "strong:action_state:session_end:at"),
        candidate("noise", 35_000, 0.2, "strong:visual_change"),
    ]
    window = AnalysisWindow(
        5_000,
        40_000,
        "strong:action_cluster",
        priority="strong",
        max_keyframes=7,
        action_phases=((10_000, "clipboard"), (30_000, "external_state"), (39_000, "session_end")),
    )

    selected = _budget_window_candidates(candidates, window, 7)

    assert {"paste_menu", "pasted", "submitted"}.issubset({item.frame.frame_id for item in selected})


def test_exact_anchor_reaches_traceable_global_dedupe(tmp_path: Path) -> None:
    cv2 = pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")
    image = np.full((90, 160, 3), 128, dtype=np.uint8)
    window = AnalysisWindow(
        1_000,
        1_250,
        "strong:send",
        priority="strong",
        step_ms=250,
        anchor_ms=(1_250,),
    )

    candidates = _select_window_candidates(
        cv2,
        {1_000: image, 1_250: image.copy()},
        window,
        0,
        tmp_path,
        VisionConfig(),
    )
    kept, duplicates = _dedupe_keyframes_globally(candidates, VisionConfig())

    assert [item.frame.timestamp_ms for item in candidates] == [1_000, 1_250]
    assert [item.timestamp_ms for item in kept] == [1_250]
    assert duplicates[0].frame.timestamp_ms == 1_000
    assert duplicates[0].reason == "lower_evidence_priority"


def test_capture_action_probes_box_selection_before_file_creation() -> None:
    window = AnalysisWindow(
        2_000,
        25_000,
        "strong:file_operation:capture",
        priority="strong",
        step_ms=250,
        max_keyframes=8,
        anchor_ms=(10_000,),
        action_anchor_ms=(10_000,),
        action_phases=((10_000, "capture"),),
    )

    probes = _probe_timestamps(window, VisionConfig())

    assert {5_000, 7_000, 9_000, 10_000}.issubset(probes)
    assert 2_000 in probes


def test_capture_start_and_file_creation_keep_both_derivation_phases() -> None:
    sensitive = "C:/Users/alice/Desktop/secret.docx"
    logs = normalize_logs(
        [
            {
                "timestamp": "2026-01-01T12:00:00",
                "event_type": "file_open",
                "file_path": sensitive,
                "end_time": "2026-01-01T12:01:00",
                "extra": {"relative_timestamp": 0.0},
            },
            {
                "timestamp": "2026-01-01T12:00:10",
                "event_type": "app_switch",
                "app_name": "SnippingTool",
                "process_info": {"process_name": "SnippingTool.exe"},
                "window_info": {"window_title": "截图工具"},
                "extra": {"relative_timestamp": 10.0},
            },
            {
                "timestamp": "2026-01-01T12:00:20",
                "event_type": "created",
                "file_path": "C:/Users/alice/Pictures/Screenshots/screenshot.png",
                "process_info": {"process_name": "SnippingTool.exe"},
                "extra": {"raw_operation": "created", "relative_timestamp": 20.0},
            },
            {
                "timestamp": "2026-01-01T12:00:15",
                "event_type": "clipboard_image",
                "extra": {"raw_operation": "clipboard_image", "relative_timestamp": 15.0},
            },
        ]
    )

    windows = build_analysis_windows(logs, [sensitive], VisionConfig())

    assert len(windows) == 1
    assert windows[0].action_phases == ((10_000, "capture_start"), (20_000, "capture"))
    probes = _probe_timestamps(windows[0], VisionConfig())
    assert {10_000, 12_000, 15_000, 18_000, 20_000}.issubset(probes)


def test_screen_share_start_probes_stable_shared_state() -> None:
    sensitive = "C:/Users/alice/Desktop/secret.txt"
    logs = normalize_logs(
        [
            {
                "timestamp": "2026-01-01T12:00:00",
                "event_type": "file_open",
                "file_path": sensitive,
                "end_time": "2026-01-01T12:02:00",
                "extra": {"relative_timestamp": 0.0},
            },
            {
                "timestamp": "2026-01-01T12:01:04",
                "event_type": "screen_share_start",
                "app_name": "wemeetapp",
                "process_info": {"process_name": "wemeetapp.exe"},
                "extra": {"raw_operation": "screen_share_start", "relative_timestamp": 64.0},
            },
        ]
    )

    windows = build_analysis_windows(logs, [sensitive], VisionConfig())

    assert len(windows) == 1
    assert windows[0].action_phases == ((64_000, "screen_share"),)
    assert windows[0].end_ms == 94_000
    assert {62_000, 64_000, 66_000, 69_000, 74_000, 79_000}.issubset(
        _probe_timestamps(windows[0], VisionConfig())
    )


def test_capture_phase_prefers_high_contrast_selection_overlay() -> None:
    np = pytest.importorskip("numpy")
    normal = np.full((90, 160), 240, dtype=np.uint8)
    selection = np.indices((90, 160)).sum(axis=0).astype(np.uint8) % 2 * 220
    candidates = [
        _FrameCandidate(KeyFrame("selection", 9_000, "selection.jpg", 0.3, "strong:action_state"), "strong", selection, (0, 64)),
        _FrameCandidate(KeyFrame("normal", 9_750, "normal.jpg", 0.3, "strong:visual_change"), "strong", normal, (0, 64)),
    ]
    window = AnalysisWindow(
        5_000,
        20_000,
        "strong:capture",
        priority="strong",
        action_anchor_ms=(10_000,),
        action_phases=((10_000, "capture"),),
    )

    focused = _focus_semantic_action_phases(candidates, window)

    assert [item.frame.frame_id for item in focused] == ["selection", "normal"]


def test_file_selection_probes_attachment_and_send_states() -> None:
    window = AnalysisWindow(
        5_000,
        40_000,
        "strong:file_selected:file_dialog_monitor",
        priority="strong",
        step_ms=250,
        max_keyframes=8,
        anchor_ms=(10_000,),
        action_anchor_ms=(10_000,),
        action_phases=((10_000, "file_selected"),),
    )

    probes = _probe_timestamps(window, VisionConfig())

    assert {8_000, 10_000, 12_000, 15_000}.issubset(probes)
    assert {20_000, 30_000, 40_000}.issubset(probes)


def test_send_phase_keeps_pre_click_click_and_confirmation() -> None:
    np = pytest.importorskip("numpy")
    gray = np.zeros((90, 160), dtype=np.uint8)
    candidates = [
        _FrameCandidate(
            KeyFrame(frame_id, timestamp, f"{frame_id}.jpg", 0.1, "strong:action_state:send"),
            "strong",
            gray,
            (0, 64),
        )
        for frame_id, timestamp in (
            ("early", 8_000),
            ("before", 9_500),
            ("click", 10_000),
            ("confirmation", 12_000),
            ("late", 15_000),
        )
    ]
    window = AnalysisWindow(
        5_000,
        20_000,
        "strong:send",
        priority="strong",
        action_anchor_ms=(10_000,),
        action_phases=((10_000, "send"),),
    )

    focused = _focus_semantic_action_phases(candidates, window)

    assert [item.frame.frame_id for item in focused] == ["before", "click", "confirmation"]


def test_sensitive_activity_window_only_probes_source_anchor_context() -> None:
    window = AnalysisWindow(
        0,
        3_600_000,
        "sensitive_activity:secret.docx",
        priority="activity",
        step_ms=1_000,
        anchor_ms=(120_000, 3_200_000),
    )

    probes = _probe_timestamps(window, VisionConfig())

    assert probes == [119_000, 120_000, 121_000, 3_199_000, 3_200_000, 3_201_000]


def test_merged_non_activity_window_with_active_ranges_does_not_use_activity_probes() -> None:
    window = AnalysisWindow(
        0,
        60_000,
        "strong:app_switch",
        priority="strong",
        step_ms=250,
        anchor_ms=(10_000, 50_000),
        active_ranges=((10_000, 50_000),),
    )

    probes = _probe_timestamps(window, VisionConfig())

    assert {9_750, 10_000, 10_250, 49_750, 50_000, 50_250}.issubset(probes)
    assert {0, 60_000}.issubset(probes)


def test_ffmpeg_cuda_frame_command_uses_nvdec_decoder() -> None:
    command = _ffmpeg_cuda_frame_command("ffmpeg", Path("recording.mp4"), 1_234, "h264_cuvid")

    assert command[:8] == ["ffmpeg", "-v", "error", "-hwaccel", "cuda", "-c:v", "h264_cuvid", "-ss"]
    assert "1.234" in command
    assert command[-2:] == ["mjpeg", "pipe:1"]


def test_window_coverage_clamps_to_video_duration() -> None:
    window = AnalysisWindow(0, 156_000, "medium", priority="medium", step_ms=1_000, max_keyframes=18)

    clamped = _clamp_window_to_duration(window, 40_000)
    coverage = _coverage_timestamps(clamped)

    assert clamped.end_ms == 40_000
    assert coverage == (0, 20_000, 40_000)


def test_video_coverage_fallback_produces_nonempty_keyframes(tmp_path: Path) -> None:
    cv2 = pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")
    video = tmp_path / "coverage.avi"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (160, 90))
    assert writer.isOpened()
    for index in range(30):
        frame = np.full((90, 160, 3), index * 8, dtype=np.uint8)
        writer.write(frame)
    writer.release()

    windows = build_video_coverage_windows(video, VisionConfig())
    selection = select_keyframes_detailed(video, windows, VisionConfig())

    assert len(windows) == 1
    assert windows[0].reason == "medium:video_coverage_fallback"
    assert len(windows[0].anchor_ms) == 6
    assert selection.keyframes


def test_frame_hash_distance_can_detect_near_duplicates() -> None:
    assert _hamming((0b101010, 64), (0b101011, 64)) == 1


def test_keyframe_filter_does_not_reject_changed_frame_by_hash_alone() -> None:
    config = VisionConfig(frame_diff_threshold=0.08, frame_hash_distance_threshold=3, frame_min_keep_gap_ms=0)

    keep_duplicate_after_long_gap = _should_keep_frame(
        timestamp_ms=120_000,
        score=0.2,
        diff_threshold=config.frame_diff_threshold,
        force_keep=False,
        exact_duplicate=False,
        frame_hash=(0b101010, 64),
        retained_hashes=[(0b101011, 64)],
        previous_small=object(),
        last_kept_ms=0,
        config=config,
    )
    keep_different_frame = _should_keep_frame(
        timestamp_ms=500,
        score=0.2,
        diff_threshold=config.frame_diff_threshold,
        force_keep=False,
        exact_duplicate=False,
        frame_hash=(0b11110000, 64),
        retained_hashes=[(0b00001111, 64)],
        previous_small=object(),
        last_kept_ms=0,
        config=config,
    )
    keep_duplicate_log_anchor = _should_keep_frame(
        timestamp_ms=145_000,
        score=0.0,
        diff_threshold=config.frame_diff_threshold,
        force_keep=True,
        exact_duplicate=True,
        frame_hash=(0b101010, 64),
        retained_hashes=[(0b101010, 64)],
        previous_small=object(),
        last_kept_ms=0,
        config=config,
    )
    keep_changed_log_anchor = _should_keep_frame(
        timestamp_ms=145_000,
        score=0.002,
        diff_threshold=config.frame_diff_threshold,
        force_keep=True,
        exact_duplicate=False,
        frame_hash=(0b101010, 64),
        retained_hashes=[(0b101010, 64)],
        previous_small=object(),
        last_kept_ms=0,
        config=config,
    )

    assert keep_duplicate_after_long_gap is True
    assert keep_different_frame is True
    assert keep_duplicate_log_anchor is False
    assert keep_changed_log_anchor is True


def test_entropy_change_can_keep_low_pixel_delta_frame() -> None:
    config = VisionConfig(frame_diff_threshold=0.08, frame_entropy_change_threshold=0.04)

    keep = _should_keep_frame(
        timestamp_ms=1_000,
        score=0.05,
        diff_threshold=config.frame_diff_threshold,
        force_keep=False,
        exact_duplicate=False,
        frame_hash=(0, 64),
        retained_hashes=[(0, 64)],
        previous_small=object(),
        last_kept_ms=0,
        config=config,
        entropy_delta=0.05,
    )

    assert keep is True


def test_frame_entropy_measures_information_distribution() -> None:
    cv2 = pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")
    uniform = np.zeros((90, 160), dtype=np.uint8)
    checkerboard = np.indices((90, 160)).sum(axis=0).astype(np.uint8) % 2 * 255

    assert _frame_entropy(cv2, uniform) == 0.0
    assert _frame_entropy(cv2, checkerboard) == pytest.approx(1.0, abs=0.01)


def test_near_duplicate_requires_two_independent_similarity_signals() -> None:
    np = pytest.importorskip("numpy")
    gray_a = np.zeros((90, 160), dtype=np.uint8)
    gray_b = np.full((90, 160), 4, dtype=np.uint8)
    frame_hash = (0, 64)
    candidates = [
        _FrameCandidate(KeyFrame("a", 1_000, "a.jpg", 0.0, "medium:coverage"), "medium", gray_a, frame_hash, 0.0),
        _FrameCandidate(KeyFrame("b", 2_000, "b.jpg", 0.0, "medium:coverage"), "medium", gray_b, frame_hash, 1.0),
    ]

    retained, duplicates = _dedupe_keyframes_globally(candidates, VisionConfig())

    assert [frame.frame_id for frame in retained] == ["a", "b"]
    assert duplicates == []


def test_global_dedupe_removes_near_duplicate_coverage_frames() -> None:
    pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")
    gray_a = np.full((90, 160), 128, dtype=np.uint8)
    gray_b = np.full((90, 160), 129, dtype=np.uint8)
    frame_hash = ((1 << 64) - 1, 64)
    candidates = [
        _FrameCandidate(KeyFrame("a", 1_000, "a.jpg", 0.0, "medium:coverage"), "medium", gray_a, frame_hash),
        _FrameCandidate(KeyFrame("b", 2_000, "b.jpg", 0.0, "medium:coverage"), "medium", gray_b, frame_hash),
    ]

    kept, duplicates = _dedupe_keyframes_globally(candidates, VisionConfig())

    assert [item.frame_id for item in kept] == ["a"]
    assert duplicates[0].frame.frame_id == "b"


def test_global_dedupe_keeps_near_duplicate_visual_change_frames() -> None:
    pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")
    gray_a = np.full((90, 160), 128, dtype=np.uint8)
    gray_b = np.full((90, 160), 129, dtype=np.uint8)
    frame_hash = ((1 << 64) - 1, 64)
    candidates = [
        _FrameCandidate(KeyFrame("a", 1_000, "a.jpg", 0.0, "medium:coverage"), "medium", gray_a, frame_hash),
        _FrameCandidate(KeyFrame("b", 2_000, "b.jpg", 0.0, "medium:visual_change"), "medium", gray_b, frame_hash),
    ]

    kept, duplicates = _dedupe_keyframes_globally(candidates, VisionConfig())

    assert [item.frame_id for item in kept] == ["a", "b"]
    assert duplicates == []


def test_global_dedupe_keeps_exact_visual_states_from_distant_actions() -> None:
    np = pytest.importorskip("numpy")
    gray = np.full((90, 160), 128, dtype=np.uint8)
    frame_hash = ((1 << 64) - 1, 64)
    candidates = [
        _FrameCandidate(KeyFrame("copy", 1_000, "copy.jpg", 0.0, "strong:visual_change"), "strong", gray, frame_hash),
        _FrameCandidate(KeyFrame("send", 31_000, "send.jpg", 0.0, "strong:visual_change"), "strong", gray, frame_hash),
    ]

    kept, duplicates = _dedupe_keyframes_globally(candidates, VisionConfig())

    assert [item.frame_id for item in kept] == ["copy", "send"]
    assert duplicates == []


def test_global_dedupe_keeps_planned_action_states_one_second_apart() -> None:
    np = pytest.importorskip("numpy")
    gray_a = np.full((90, 160), 128, dtype=np.uint8)
    gray_b = np.full((90, 160), 129, dtype=np.uint8)
    frame_hash = ((1 << 64) - 1, 64)
    candidates = [
        _FrameCandidate(KeyFrame("attachment", 10_000, "attachment.jpg", 0.1, "strong:action_state:outbound_context"), "strong", gray_a, frame_hash),
        _FrameCandidate(KeyFrame("success", 12_000, "success.jpg", 0.2, "strong:action_state:outbound_context"), "strong", gray_b, frame_hash),
    ]

    kept, duplicates = _dedupe_keyframes_globally(candidates, VisionConfig())

    assert [item.frame_id for item in kept] == ["attachment", "success"]
    assert duplicates == []


def test_global_dedupe_keeps_earliest_equivalent_action_state() -> None:
    pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")
    gray_a = np.full((90, 160), 128, dtype=np.uint8)
    gray_b = np.full((90, 160), 128, dtype=np.uint8)
    frame_hash = ((1 << 64) - 1, 64)
    candidates = [
        _FrameCandidate(KeyFrame("early", 84_228, "early.jpg", 0.0, "activity:activity_gap"), "activity", gray_a, frame_hash),
        _FrameCandidate(KeyFrame("late", 86_161, "late.jpg", 0.0, "activity:activity_gap"), "activity", gray_b, frame_hash),
    ]

    kept, duplicates = _dedupe_keyframes_globally(candidates, VisionConfig())

    assert [item.frame_id for item in kept] == ["early"]
    assert duplicates[0].frame.frame_id == "late"


def test_normalize_path_repairs_gbk_text_decoded_as_latin1() -> None:
    garbled_name = "公司合同".encode("gb18030").decode("latin1")

    assert normalize_path(f"C:/Users/alice/Desktop/{garbled_name}.docx") == "C:/Users/alice/Desktop/公司合同.docx"
    assert same_file(f"C:/Users/alice/Desktop/{garbled_name}.docx", "C:/Users/alice/Desktop/公司合同.docx")


def test_file_dialog_compatibility_boundary_does_not_filter_frames() -> None:
    pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")

    def candidate(frame_id: str, timestamp_ms: int, reason: str, window_id: str, image: object) -> _FrameCandidate:
        frame = KeyFrame(frame_id, timestamp_ms, f"{frame_id}.jpg", 0.0, reason, window_id=window_id)
        return _FrameCandidate(frame, reason.split(":", 1)[0], image, (0, 64))

    blank = np.zeros((90, 160), dtype=np.uint8)
    dialog_initial = blank.copy()
    dialog_initial[10:30, 10:40] = 255
    dialog_final = dialog_initial.copy()
    dialog_final[50:60, 20:120] = 255
    loading = blank.copy()
    loading[5:15, 5:15] = 255
    result = blank.copy()
    for row in range(20, 75, 10):
        result[row:row + 2, 20:140] = 255
    saved = blank.copy()
    for column in range(10, 150, 12):
        saved[10:80, column:column + 2] = 255
    candidates = [
        candidate("dialog_initial", 1_000, "strong:anchor", "window_0", dialog_initial),
        candidate("dialog_final", 4_000, "strong:visual_change", "window_0", dialog_final),
        candidate("loading", 5_200, "activity:anchor", "window_1", loading),
        candidate("result_start", 6_000, "activity:anchor", "window_1", result),
        candidate("result_final", 8_000, "activity:anchor", "window_1", result),
        candidate("saved", 11_000, "activity:anchor", "window_1", saved),
    ]
    windows = [
        AnalysisWindow(
            0,
            12_000,
            "strong:file_selected:file_dialog_monitor",
            priority="strong",
            anchor_ms=(1_000, 4_000),
            active_ranges=((1_000, 4_999), (5_000, 12_000)),
        ),
        AnalysisWindow(
            0,
            12_000,
            "sensitive_activity:secret.docx",
            priority="activity",
            active_ranges=((1_000, 4_999), (5_000, 12_000)),
        ),
    ]

    focused = _focus_file_dialog_flows([item.frame for item in candidates], candidates, windows)

    assert [frame.frame_id for frame in focused] == [
        "dialog_initial", "dialog_final", "loading", "result_start", "result_final", "saved"
    ]


def test_file_dialog_compatibility_boundary_preserves_available_evidence() -> None:
    pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")
    dialog = np.zeros((90, 160), dtype=np.uint8)
    result = dialog.copy()
    for row in range(20, 75, 10):
        result[row:row + 2, 20:140] = 255
    frames = [
        KeyFrame("dialog_open", 1_000, "dialog.jpg", 0.0, "strong:anchor", window_id="window_0"),
        KeyFrame("result_start", 5_000, "result.jpg", 0.0, "activity:anchor", window_id="window_1"),
        KeyFrame("result_final", 7_000, "result.jpg", 0.0, "activity:anchor", window_id="window_1"),
    ]
    candidates = [
        _FrameCandidate(frames[0], "strong", dialog, (0, 64)),
        _FrameCandidate(frames[1], "activity", result, (0, 64)),
        _FrameCandidate(frames[2], "activity", result, (0, 64)),
    ]
    windows = [
        AnalysisWindow(
            0,
            8_000,
            "strong:file_selected:file_dialog_monitor",
            priority="strong",
            anchor_ms=(1_000,),
            active_ranges=((1_000, 3_999), (4_000, 8_000)),
        ),
        AnalysisWindow(0, 8_000, "sensitive_activity:secret.docx", priority="activity"),
    ]

    focused = _focus_file_dialog_flows(frames, candidates, windows)

    assert [frame.frame_id for frame in focused] == ["dialog_open", "result_start", "result_final"]


def test_actionable_compatibility_boundary_does_not_apply_a_second_filter() -> None:
    frames = [
        KeyFrame("reading", 1_000, "reading.jpg", 0.1, "activity:anchor", window_id="window_1"),
        KeyFrame("uploading", 1_500, "uploading.jpg", 0.4, "activity:activity_gap", window_id="window_1"),
        KeyFrame("save_pdf", 2_000, "save.jpg", 0.4, "strong:anchor", window_id="window_0"),
        KeyFrame("confirmation", 2_500, "confirmation.jpg", 0.3, "activity:anchor", window_id="window_1"),
    ]
    windows = [
        AnalysisWindow(0, 4_000, "strong:print_to_pdf", priority="strong", anchor_ms=(2_000,)),
        AnalysisWindow(0, 4_000, "sensitive_activity:secret.docx", priority="activity", anchor_ms=(1_000,)),
    ]

    focused = _focus_actionable_keyframes(frames, [], windows)

    assert [frame.frame_id for frame in focused] == ["reading", "uploading", "save_pdf", "confirmation"]


def test_actionable_compatibility_boundary_preserves_temporal_context() -> None:
    frames = [
        KeyFrame("source_early", 1_000, "early.jpg", 0.1, "activity:anchor", window_id="window_1"),
        KeyFrame("source_near", 4_000, "near.jpg", 0.2, "activity:anchor", window_id="window_1"),
        KeyFrame("copy", 5_000, "copy.jpg", 0.4, "strong:anchor", window_id="window_0"),
        KeyFrame("result", 7_000, "result.jpg", 0.3, "activity:anchor", window_id="window_1"),
    ]
    windows = [
        AnalysisWindow(
            0,
            8_000,
            "strong:clipboard_text",
            priority="strong",
            anchor_ms=(5_000,),
            action_anchor_ms=(5_000,),
        ),
        AnalysisWindow(0, 8_000, "sensitive_activity:secret.docx", priority="activity"),
    ]

    focused = _focus_actionable_keyframes(frames, [], windows)

    assert [frame.frame_id for frame in focused] == ["source_early", "source_near", "copy", "result"]


def test_actionable_compatibility_boundary_only_orders_existing_frames() -> None:
    frames = [
        KeyFrame("opened", 1_000, "opened.jpg", 0.1, "activity:anchor", window_id="window_0"),
        KeyFrame("closed", 9_000, "closed.jpg", 0.2, "activity:anchor", window_id="window_0"),
    ]
    windows = [
        AnalysisWindow(0, 10_000, "sensitive_activity:secret.docx", priority="activity", anchor_ms=(1_000, 9_000)),
    ]

    assert _focus_actionable_keyframes(frames, [], windows) == frames


def test_actionable_compatibility_boundary_preserves_external_context() -> None:
    frames = [
        KeyFrame(f"frame_{index}", index * 1_000, f"frame_{index}.jpg", 0.2, "activity:anchor", window_id="window_0")
        for index in range(5)
    ]
    windows = [
        AnalysisWindow(
            0,
            5_000,
            "sensitive_activity:secret.docx",
            priority="activity",
            active_apps=("Edge", "Explorer"),
        ),
    ]

    focused = _focus_actionable_keyframes(frames, [], windows)

    assert [frame.frame_id for frame in focused] == ["frame_0", "frame_1", "frame_2", "frame_3", "frame_4"]


def test_actionable_compatibility_boundary_does_not_reclassify_pixels() -> None:
    frame = KeyFrame("desktop", 1_000, "desktop.jpg", 1.0, "strong:anchor", window_id="window_0")
    windows = [AnalysisWindow(0, 2_000, "strong:app_switch:window_monitor", priority="strong")]

    assert _focus_actionable_keyframes([frame], [], windows) == [frame]


def test_unresolved_sensitive_sink_context_keeps_sparse_visual_evidence() -> None:
    frames = [
        KeyFrame(f"sink_{index}", index * 1_000, f"sink_{index}.jpg", 0.2, "strong:anchor", window_id="window_0")
        for index in range(6)
    ] + [
        KeyFrame(f"source_{index}", index * 1_000 + 500, f"source_{index}.jpg", 0.2, "activity:anchor", window_id="window_1")
        for index in range(4)
    ]
    windows = [
        AnalysisWindow(0, 10_000, "strong:app_switch:window_monitor:meeting", priority="strong"),
        AnalysisWindow(0, 10_000, "sensitive_activity:secret.docx", priority="activity"),
    ]

    focused = _focus_actionable_keyframes(frames, [], windows)

    assert [frame.frame_id for frame in focused] == [
        "sink_0", "source_0", "sink_1", "source_1", "sink_2",
        "source_2", "sink_3", "source_3", "sink_4", "sink_5",
    ]


def test_unresolved_medium_clipboard_window_keeps_visual_risk_phases() -> None:
    frames = [
        KeyFrame("compose", 2_000, "compose.jpg", 0.2, "medium:anchor", window_id="window_0"),
        KeyFrame("attached", 5_000, "attached.jpg", 0.4, "medium:activity_gap", window_id="window_0"),
    ]
    windows = [
        AnalysisWindow(0, 8_000, "medium:clipboard_text:browser", priority="medium"),
    ]

    focused = _focus_actionable_keyframes(frames, [], windows)

    assert [frame.frame_id for frame in focused] == ["compose", "attached"]


def test_structured_recent_folder_upload_builds_reportable_strong_window() -> None:
    logs = normalize_logs(
        [
            {
                "timestamp": "2026-03-10T09:27:04.380",
                "event_type": "opened",
                "file_path": "C:/Users/alice/Desktop/customer-data.zip",
                "process_info": {"process_name": "chrome.exe"},
                "window_info": {"window_title": "Blog upload - Chrome"},
                "upload_detection": {
                    "is_upload": True,
                    "upload_type": "File Access",
                    "original_file": "C:/Users/alice/Desktop/customer-data.zip",
                },
                "extra": {"source": "recent_folder_monitor", "relative_timestamp": 33.38},
            }
        ]
    )

    windows = build_analysis_windows(logs, ["C:/unrelated/secret.docx"], VisionConfig())

    assert len(windows) == 1
    assert windows[0].priority == "strong"
    assert windows[0].action_anchor_ms == (33_380,)
    assert windows[0].reason.endswith(":upload")


def test_medium_window_compatibility_boundary_does_not_use_app_special_cases() -> None:
    frames = [
        KeyFrame("before", 5_000, "before.jpg", 0.2, "medium:anchor", window_id="window_0"),
        KeyFrame("action", 12_000, "action.jpg", 0.4, "medium:activity_gap", window_id="window_0"),
        KeyFrame("sent", 15_000, "sent.jpg", 0.4, "medium:anchor", window_id="window_0"),
    ]
    windows = [
        AnalysisWindow(
            0,
            20_000,
            "medium:app_switch:window_monitor",
            priority="medium",
            active_apps=("Androws",),
        ),
    ]

    focused = _focus_actionable_keyframes(frames, [], windows)

    assert [frame.frame_id for frame in focused] == ["before", "action", "sent"]


def test_pipeline_records_prefer_all_keyevents_over_raw_logs(tmp_path: Path) -> None:
    logs_dir = tmp_path / "logs"
    logs_dir.mkdir()
    logs_file = logs_dir / "logs.json"
    keyevents_file = logs_dir / "keyevents.json"
    logs_file.write_text(json.dumps([{"timestamp": "t0", "event_type": "app_switch"}]), encoding="utf-8")
    keyevents_file.write_text(
        json.dumps(
            [
                {
                    "timestamp": "t1",
                    "event_type": "file_upload",
                    "file_path": "C:/secret.docx",
                    "app_name": "TIM",
                    "process_info": {"process_name": "Androws.exe"},
                },
                {"timestamp": "t2", "event_type": "file_upload", "app_name": "Chrome"},
                {"timestamp": "t3", "event_type": "app_switch", "app_name": "TIM"},
            ]
        ),
        encoding="utf-8",
    )

    records = _load_pipeline_records(logs_file)

    assert [(item["timestamp"], item["event_type"]) for item in records] == [
        ("t1", "file_upload"),
        ("t2", "file_upload"),
        ("t3", "app_switch"),
    ]


def test_unresolved_sink_context_survives_when_dedupe_kept_activity_frames() -> None:
    frames = [
        KeyFrame("uploading", 2_000, "uploading.jpg", 0.2, "activity:anchor", window_id="window_1"),
        KeyFrame("uploaded", 4_000, "uploaded.jpg", 0.3, "activity:anchor", window_id="window_1"),
    ]
    windows = [
        AnalysisWindow(0, 5_000, "strong:app_switch:window_monitor:browser", priority="strong"),
        AnalysisWindow(0, 5_000, "sensitive_activity:secret.docx", priority="activity"),
    ]

    focused = _focus_actionable_keyframes(frames, [], windows)

    assert [frame.frame_id for frame in focused] == ["uploading", "uploaded"]


def test_single_action_frame_adds_available_deduplicated_result_state() -> None:
    frames = [
        KeyFrame("rename", 2_000, "rename.jpg", 0.3, "strong:anchor", window_id="window_0"),
        KeyFrame("audit", 8_000, "audit.jpg", 0.4, "activity:visual_change", window_id="window_1"),
    ]
    windows = [
        AnalysisWindow(0, 9_000, "strong:renamed:derivation", priority="strong", action_anchor_ms=(2_000,)),
        AnalysisWindow(0, 9_000, "sensitive_activity:secret.docx", priority="activity"),
    ]

    focused = _focus_actionable_keyframes(frames, [], windows)

    assert [frame.frame_id for frame in focused] == ["rename", "audit"]


def test_strong_derivation_is_not_filtered_after_raw_selection() -> None:
    frames = [
        KeyFrame("before", 1_000, "before.jpg", 0.1, "strong:anchor", window_id="window_0"),
        KeyFrame("export", 5_000, "export.jpg", 0.2, "strong:anchor", window_id="window_0"),
        KeyFrame("result", 6_000, "result.jpg", 0.2, "strong:anchor", window_id="window_0"),
        KeyFrame("after", 9_000, "after.jpg", 0.2, "strong:anchor", window_id="window_0"),
    ]
    windows = [
        AnalysisWindow(
            0,
            10_000,
            "strong:app_switch:derivation",
            priority="strong",
            action_anchor_ms=(5_000,),
        )
    ]

    focused = _focus_actionable_keyframes(frames, [], windows)

    assert [frame.frame_id for frame in focused] == ["before", "export", "result", "after"]


def test_strong_action_flow_keeps_small_complete_temporal_sequence() -> None:
    frames = [
        KeyFrame(f"frame_{index}", (index + 1) * 1_000, f"{index}.jpg", 0.2, "strong:anchor", window_id="window_0")
        for index in range(5)
    ]
    windows = [
        AnalysisWindow(0, 10_000, "strong:capture", priority="strong", action_anchor_ms=(1_000, 2_000, 3_000, 4_000, 5_000))
    ]

    focused = _focus_actionable_keyframes(frames, [], windows)

    assert [frame.frame_id for frame in focused] == ["frame_0", "frame_1", "frame_2", "frame_3", "frame_4"]


def test_clipboard_frames_are_not_reclassified_after_raw_selection() -> None:
    frames = [
        KeyFrame("selected_text", 9_000, "word.jpg", 0.2, "strong:anchor", window_id="window_0"),
        KeyFrame("desktop", 10_000, "desktop.jpg", 0.3, "strong:anchor", window_id="window_0"),
    ]
    windows = [
        AnalysisWindow(0, 12_000, "strong:clipboard_text", priority="strong", action_anchor_ms=(10_000,))
    ]

    focused = _focus_actionable_keyframes(frames, [], windows)

    assert [frame.frame_id for frame in focused] == ["selected_text", "desktop"]


def test_global_dedupe_prefers_file_dialog_selection_over_activity_context() -> None:
    pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")
    gray = np.full((90, 160), 128, dtype=np.uint8)
    frame_hash = (0, 64)
    candidates = [
        _FrameCandidate(KeyFrame("dialog", 10_000, "dialog.jpg", 0.0, "strong:anchor", window_id="window_0"), "strong", gray, frame_hash),
        _FrameCandidate(KeyFrame("activity", 10_100, "activity.jpg", 0.0, "activity:anchor", window_id="window_1"), "activity", gray, frame_hash),
    ]
    windows = [
        AnalysisWindow(0, 12_000, "strong:file_selected:file_dialog_monitor", priority="strong"),
        AnalysisWindow(0, 12_000, "sensitive_activity:secret.docx", priority="activity"),
    ]

    kept, duplicates = _dedupe_keyframes_globally(candidates, VisionConfig(), windows=windows)

    assert [item.frame_id for item in kept] == ["dialog"]
    assert duplicates[0].frame.frame_id == "activity"


def test_global_dedupe_prefers_explicit_derivation_over_activity_context() -> None:
    pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")
    gray = np.full((90, 160), 128, dtype=np.uint8)
    candidates = [
        _FrameCandidate(KeyFrame("export", 10_000, "export.jpg", 0.0, "strong:anchor", window_id="window_0"), "strong", gray, (0, 64)),
        _FrameCandidate(KeyFrame("activity", 10_100, "activity.jpg", 0.0, "activity:anchor", window_id="window_1"), "activity", gray, (0, 64)),
    ]
    windows = [
        AnalysisWindow(0, 12_000, "strong:app_switch:derivation", priority="strong", action_anchor_ms=(10_000,)),
        AnalysisWindow(0, 12_000, "sensitive_activity:secret.docx", priority="activity"),
    ]

    kept, _ = _dedupe_keyframes_globally(candidates, VisionConfig(), windows=windows)

    assert [item.frame_id for item in kept] == ["export"]


def test_derivation_candidate_is_strong_when_sensitive_activity_exists() -> None:
    sensitive = "C:/Users/alice/Documents/secret.docx"
    logs = normalize_logs(
        [
            {
                "timestamp": "2026-01-01T12:00:00",
                "event_type": "opened",
                "file_path": sensitive,
                "extra": {"relative_timestamp": 0.0},
            },
            {
                "timestamp": "2026-01-01T12:00:10",
                "event_type": "print_to_pdf",
                "file_path": "C:/Users/alice/Desktop/secret.pdf",
                "process_info": {"process_name": "WINWORD.EXE"},
                "extra": {
                    "relative_timestamp": 10.0,
                    "raw_operation": "print_to_pdf",
                    "source_path": sensitive,
                },
            },
        ]
    )

    windows = build_analysis_windows(logs, [sensitive], VisionConfig())

    assert [window.priority for window in windows] == ["strong"]
    assert windows[0].anchor_ms == (10_000,)


def test_clipboard_with_downstream_paste_uses_recent_sensitive_signal() -> None:
    sensitive = "C:/Users/alice/Documents/strategy.docx"
    logs = normalize_logs(
        [
            {
                "timestamp": "2026-01-01T12:00:00",
                "event_type": "modified",
                "file_path": sensitive,
                "extra": {"relative_timestamp": 0.0},
            },
            {
                "timestamp": "2026-01-01T12:00:10",
                "event_type": "clipboard_text",
                "app_name": "WINWORD",
                "content_preview": "A copied sensitive paragraph",
                "extra": {"relative_timestamp": 10.0, "raw_operation": "clipboard_text"},
            },
            {
                "timestamp": "2026-01-01T12:00:20",
                "event_type": "clipboard_operation",
                "app_name": "Edge",
                "extra": {"relative_timestamp": 20.0, "raw_operation": "clipboard_paste"},
            },
        ]
    )

    windows = build_analysis_windows(logs, [sensitive], VisionConfig())
    action = next(window for window in windows if window.priority == "strong")

    assert action.action_anchor_ms == (10_000, 20_000)
    assert action.action_phases == ((10_000, "clipboard"), (20_000, "paste"))


def test_clipboard_window_uses_full_semantic_horizon_and_keeps_prior_derivation_state() -> None:
    sensitive = "C:/Users/alice/Desktop/merger_notes.docx"
    logs = normalize_logs(
        [
            {
                "timestamp": "2026-01-01T12:00:00",
                "event_type": "modified",
                "file_path": sensitive,
                "extra": {"relative_timestamp": 0.0},
            },
            {
                "timestamp": "2026-01-01T12:00:25",
                "event_type": "renamed",
                "file_path": "C:/Users/alice/Desktop/New Text Document.txt",
                "destination_path": "C:/Users/alice/Desktop/draft.txt",
                "extra": {"relative_timestamp": 25.0, "raw_operation": "renamed"},
            },
            {
                "timestamp": "2026-01-01T12:00:47",
                "event_type": "clipboard_text",
                "app_name": "WINWORD",
                "content_preview": "copied merger terms",
                "extra": {"relative_timestamp": 47.0, "raw_operation": "clipboard_text"},
            },
            {
                "timestamp": "2026-01-01T12:00:48",
                "event_type": "modified",
                "file_path": "C:/Users/alice/Desktop/draft.txt",
                "window_info": {"window_title": "draft.txt - Notepad"},
                "extra": {"relative_timestamp": 48.0, "raw_operation": "modified"},
            },
            {
                "timestamp": "2026-01-01T12:01:29",
                "event_type": "created",
                "file_path": "C:/Users/alice/Pictures/Screenshots/Screenshot.png",
                "extra": {"relative_timestamp": 89.0, "raw_operation": "created"},
            },
        ]
    )

    windows = build_analysis_windows(logs, [sensitive], VisionConfig())
    clipboard = next(
        window
        for window in windows
        if (47_000, "clipboard") in window.action_phases
    )

    assert clipboard.start_ms <= 17_000
    assert clipboard.end_ms >= 77_000


def test_global_dedupe_keeps_distant_anchor_frames_with_small_visual_changes() -> None:
    pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")
    gray_a = np.full((90, 160), 128, dtype=np.uint8)
    gray_b = np.full((90, 160), 129, dtype=np.uint8)
    frame_hash = ((1 << 64) - 1, 64)
    candidates = [
        _FrameCandidate(KeyFrame("a", 10_000, "a.jpg", 0.0, "strong:anchor"), "strong", gray_a, frame_hash),
        _FrameCandidate(KeyFrame("b", 50_000, "b.jpg", 0.0, "strong:anchor"), "strong", gray_b, frame_hash),
    ]

    kept, duplicates = _dedupe_keyframes_globally(candidates, VisionConfig())

    assert [item.frame_id for item in kept] == ["a", "b"]
    assert duplicates == []


def test_global_dedupe_keeps_distant_strong_and_activity_anchor_context() -> None:
    pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")
    gray_a = np.full((90, 160), 128, dtype=np.uint8)
    gray_b = np.full((90, 160), 129, dtype=np.uint8)
    frame_hash = (0, 64)
    candidates = [
        _FrameCandidate(KeyFrame("strong", 5_424, "strong.jpg", 0.0, "strong:anchor"), "strong", gray_a, frame_hash),
        _FrameCandidate(KeyFrame("activity", 14_600, "activity.jpg", 0.0, "activity:anchor"), "activity", gray_b, frame_hash),
    ]

    kept, duplicates = _dedupe_keyframes_globally(candidates, VisionConfig())

    assert [item.frame_id for item in kept] == ["strong", "activity"]
    assert duplicates == []


def test_global_dedupe_removes_near_duplicate_anchor_frames_from_same_event() -> None:
    pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")
    gray_a = np.full((90, 160), 128, dtype=np.uint8)
    gray_b = np.full((90, 160), 129, dtype=np.uint8)
    frame_hash = (0, 64)
    candidates = [
        _FrameCandidate(KeyFrame("a", 10_000, "a.jpg", 0.0, "strong:anchor"), "strong", gray_a, frame_hash),
        _FrameCandidate(KeyFrame("b", 10_250, "b.jpg", 0.0, "strong:anchor"), "strong", gray_b, frame_hash),
    ]

    kept, duplicates = _dedupe_keyframes_globally(candidates, VisionConfig())

    assert [item.frame_id for item in kept] == ["a"]
    assert duplicates[0].frame.frame_id == "b"


def test_keyframe_probe_timestamps_are_grouped_for_sequential_decode() -> None:
    groups = _timestamp_groups([0, 250, 1_000, 8_000, 8_300, 20_000], max_gap_ms=1_000)

    assert groups == [[0, 250, 1_000], [8_000, 8_300], [20_000]]


def test_action_probe_includes_frame_immediately_before_click() -> None:
    window = AnalysisWindow(
        5_000,
        20_000,
        "strong:send",
        priority="strong",
        step_ms=250,
        max_keyframes=8,
        action_anchor_ms=(10_000,),
        action_phases=((10_000, "send"),),
    )

    assert 9_900 in _probe_timestamps(window, VisionConfig())


def test_global_dedupe_preserves_pre_action_payload_frame() -> None:
    pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")
    gray = np.full((90, 160), 128, dtype=np.uint8)
    frame_hash = (0, 64)
    candidates = [
        _FrameCandidate(
            KeyFrame("before", 9_900, "before.jpg", 0.0, "strong:action_state:send:pre_action"),
            "strong",
            gray,
            frame_hash,
        ),
        _FrameCandidate(
            KeyFrame("after", 10_000, "after.jpg", 0.0, "strong:action_state:send"),
            "strong",
            gray,
            frame_hash,
        ),
    ]

    kept, duplicates = _dedupe_keyframes_globally(candidates, VisionConfig())

    assert [item.frame_id for item in kept] == ["before", "after"]
    assert duplicates == []


def test_close_derivation_phases_keep_latest_pre_action_frame() -> None:
    np = pytest.importorskip("numpy")
    gray = np.zeros((90, 160), dtype=np.uint8)
    candidates = [
        _FrameCandidate(
            KeyFrame(frame_id, timestamp, f"{frame_id}.jpg", 0.1, reason),
            "strong",
            gray,
            (0, 64),
        )
        for frame_id, timestamp, reason in (
            ("first_before", 9_900, "strong:action_state:derive:pre_action"),
            ("first", 10_000, "strong:action_state:derive"),
            ("second_before", 10_100, "strong:action_state:derive:pre_action"),
            ("second", 10_200, "strong:action_state:derive"),
        )
    ]
    window = AnalysisWindow(
        5_000,
        15_000,
        "strong:action_cluster",
        priority="strong",
        action_anchor_ms=(10_000, 10_200),
        action_phases=((10_000, "derive"), (10_200, "derive")),
    )

    focused = _focus_semantic_action_phases(candidates, window)

    assert [item.frame.frame_id for item in focused] == ["second_before", "second"]


def test_missing_sequential_frames_retry_with_direct_seek(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "data_leak_detector.frame_analyzer.frames._read_timestamp_group_sequentially",
        lambda cv2, capture, timestamps, fps: {},
    )
    monkeypatch.setattr(
        "data_leak_detector.frame_analyzer.frames._seek_read_frame",
        lambda cv2, capture, timestamp: f"frame-{timestamp}",
    )

    frames = _read_frames_for_timestamps(
        object(),
        object(),
        [1_000, 1_250],
        30.0,
        VisionConfig(frame_sequential_gap_ms=5_000),
    )

    assert frames == {1_000: "frame-1000", 1_250: "frame-1250"}


















def test_direct_keyframe_precompute_removes_legacy_vision_outputs(tmp_path: Path) -> None:
    image = tmp_path / "frame.jpg"
    image.write_bytes(b"fake image")
    frame = KeyFrame("raw", 1000, str(image), 0.9, "strong:visual_change", window_id="window_0")

    manifest = export_vision_artifacts(
        artifact_dir=tmp_path / "vision",
        keyframes=[frame],
        raw_all_keyframes=[frame],
        duplicate_keyframes=[],
    )

    root = Path(manifest["root_dir"])

    assert Path(manifest["keyframes_raw_all_dir"]).exists()
    assert Path(manifest["keyframes_raw_dir"]).exists()
    assert manifest["counts"]["keyframes_raw_files"] == 1


def test_direct_keyframe_precompute_skips_vlm(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    image = tmp_path / "frame.jpg"
    image.write_bytes(b"fake image")
    frame = KeyFrame("raw", 1000, str(image), 0.9, "strong:anchor", window_id="window_0")
    selection = KeyFrameSelection(keyframes=[frame], raw_keyframes=[frame], duplicates=[], warnings=[])

    monkeypatch.setattr(
        "data_leak_detector.frame_analyzer.analyzer.select_keyframes_detailed",
        lambda *_args, **_kwargs: selection,
    )
    monkeypatch.setattr(
        "data_leak_detector.frame_analyzer.vlm_dispatch.run_vlm_batches",
        lambda *_args, **_kwargs: pytest.fail("direct precompute must not call VLM"),
    )

    result = analyze_video_behavior(
        video_path="missing.mp4",
        logs=[],
        sensitive_files=["salary.xlsx"],
        vision_enabled=True,
        max_vlm_frames=0,
        analysis_windows=[AnalysisWindow(0, 2000, "unit", priority="strong")],
        artifact_dir=tmp_path / "vision",
    )

    vision = result["statistics"]["vision"]
    artifacts = vision["artifacts"]
    assert result["observations"] == []
    assert vision["vlm_enabled_for_run"] is False
    assert vision["vlm_frames"] == 0
    assert Path(artifacts["vision_precompute_file"]).exists()


def test_event_correlator_links_derived_upload_to_original() -> None:
    original = "C:/Users/alice/Documents/customer_salary.xlsx"
    derived = "C:/Users/alice/Desktop/customer_salary_part1.xlsx"
    bundle = EventCorrelator().run(
        {
            "session_id": "unit",
            "log_events": _records(),
            "frame_segments": [],
            "sensitive_files": [original],
        }
    )

    assert bundle["analysis_status"] == "success"
    assert bundle["upload_candidates"]
    assert bundle["file_lineage"]["direct_file_mappings"][derived] == original
    leak_fact = next(fact for fact in bundle["datalog_facts"] if fact["relation"] == "LeakFile")
    assert leak_fact["args"][2] == derived

    engine = DatalogEngine()
    for fact in bundle["datalog_facts"]:
        engine.add_fact(fact["relation"], *fact["args"])
    leaks = engine.query_leak()
    assert len(leaks) == 1
    assert leaks[0].leaked_file == derived
    assert ":transfer" in leaks[0].full_path


def test_event_correlator_builds_two_hop_clipboard_lineage_from_bounded_writes() -> None:
    original = "C:/Users/alice/Desktop/merger_notes.docx"
    draft = "C:/Users/alice/Desktop/draft.txt"
    screenshot = "C:/Users/alice/Pictures/Screenshots/Screenshot 2026-06-05.png"
    records = [
        {
            "timestamp": "2026-06-05T22:41:00",
            "event_type": "modified",
            "file_path": "C:/Users/alice/AppData/Roaming/Office/state.json",
            "process_info": {"process_name": "wps.exe"},
            "window_info": {"window_title": "merger_notes.docx - WPS Office"},
            "extra": {"relative_timestamp": 0.0, "raw_operation": "modified"},
        },
        {
            "timestamp": "2026-06-05T22:41:47",
            "event_type": "clipboard_text",
            "process_info": {"process_name": "wps.exe"},
            "content_preview": "copied merger terms",
            "extra": {"relative_timestamp": 47.0, "raw_operation": "clipboard_text"},
        },
        {
            "timestamp": "2026-06-05T22:41:48.250",
            "event_type": "modified",
            "file_path": draft,
            "process_info": {"process_name": "Notepad.exe"},
            "window_info": {"window_title": "*draft.txt - Notepad"},
            "extra": {"relative_timestamp": 48.25, "raw_operation": "modified"},
        },
        {
            "timestamp": "2026-06-05T22:42:20",
            "event_type": "clipboard_image",
            "process_info": {"process_name": "Notepad.exe"},
            "extra": {"relative_timestamp": 80.0, "raw_operation": "clipboard_image"},
        },
        {
            "timestamp": "2026-06-05T22:42:29",
            "event_type": "created",
            "file_path": screenshot,
            "process_info": {"process_name": "explorer.exe"},
            "window_info": {"window_title": "Program Manager"},
            "extra": {"relative_timestamp": 89.0, "raw_operation": "created"},
        },
    ]

    observations = [
        {
            "observation_id": "vlm_upload",
            "start_ms": 120_000,
            "end_ms": 120_000,
            "app_name": "QQ",
            "operation_type": "external_sink_interaction",
            "resource": Path(screenshot).name,
            "related_resources": [Path(screenshot).name],
            "description": "direct_leak: file_send. sink_type=chat_upload. action_status=submitted.",
            "confidence": 0.95,
            "source": "vlm",
        }
    ]
    correlator = EventCorrelator()
    bundle = correlator.run(
        {
            "case_id": "clipboard-lineage",
            "log_events": records,
            "frame_segments": observations,
            "sensitive_files": [original],
            "non_vlm_enabled": False,
        }
    )
    derived = correlator.derived_sensitive_files(normalize_logs(records), [original])

    assert bundle["file_lineage"]["direct_file_mappings"] == {
        draft: original,
        screenshot: draft,
    }
    assert derived == [draft, screenshot]
    assert bundle["upload_candidates"][0]["current_file"] == screenshot
    engine = DatalogEngine(case_id="clipboard-lineage")
    for fact in bundle["datalog_facts"]:
        engine.add_fact(fact["relation"], *fact["args"], case_id=fact["case_id"])
    leak = engine.query_leak()[0]
    assert leak.file_chain == (original, draft, screenshot)
    assert [
        (item.get("source_file"), item.get("derived_file"))
        for item in leak.flow_steps
        if item["relation"] == "TransferFile"
    ] == [(original, draft), (draft, screenshot)]


def test_event_correlator_does_not_infer_same_app_family_clipboard_lineage() -> None:
    original = "C:/Users/alice/Desktop/salary.xlsx"
    presentation = "C:/Users/alice/Desktop/business_review.pptx"
    records = [
        {
            "timestamp": "2026-06-05T22:41:00",
            "event_type": "modified",
            "file_path": original,
            "process_info": {"process_name": "wps.exe"},
        },
        {
            "timestamp": "2026-06-05T22:41:10",
            "event_type": "clipboard_text",
            "process_info": {"process_name": "wps.exe"},
            "extra": {"raw_operation": "clipboard_text"},
        },
        {
            "timestamp": "2026-06-05T22:41:11",
            "event_type": "modified",
            "file_path": presentation,
            "process_info": {"process_name": "wpp.exe"},
            "window_info": {"window_title": "business_review.pptx - WPS Office"},
            "extra": {"raw_operation": "modified"},
        },
    ]

    bundle = EventCorrelator().run(
        {"log_events": records, "frame_segments": [], "sensitive_files": [original]}
    )

    assert bundle["file_lineage"]["direct_file_mappings"] == {}


def test_event_correlator_uses_recent_document_context_for_save_as_sink_path() -> None:
    original = "C:/Users/alice/Documents/confidential_terms.docx"
    derived = "C:/Users/alice/Desktop/ordinary_terms.docx"
    records = [
        {
            "timestamp": "2026-06-05T22:41:00",
            "event_type": "modified",
            "file_path": original,
            "process_info": {"process_name": "wps.exe"},
            "window_info": {"window_title": "confidential_terms.docx - WPS Office"},
        },
        {
            "timestamp": "2026-06-05T22:41:10",
            "event_type": "created",
            "file_path": derived,
            "process_info": {"process_name": "wps.exe"},
            "extra": {"raw_operation": "created"},
        },
        {
            "timestamp": "2026-06-05T22:41:20",
            "event_type": "file_selected",
            "file_path": "ordinary_terms",
            "process_info": {"process_name": "quark.exe"},
            "window_info": {"window_title": "Quark Netdisk"},
            "extra": {"raw_operation": "file_selected"},
        },
    ]
    observations = [
        {
            "observation_id": "vlm_upload",
            "start_ms": 20_000,
            "end_ms": 20_000,
            "app_name": "Quark Netdisk",
            "operation_type": "external_sink_interaction",
            "resource": "confidential_terms.docx",
            "related_resources": ["confidential_terms.docx"],
            "description": "direct_leak: cloud_upload. sink_type=cloud_sync. action_status=submitted.",
            "confidence": 0.95,
            "source": "vlm",
        }
    ]

    bundle = EventCorrelator().run(
        {
            "log_events": records,
            "frame_segments": observations,
            "sensitive_files": [original],
        }
    )

    assert bundle["file_lineage"]["direct_file_mappings"] == {derived: original}
    assert bundle["upload_candidates"][0]["original_file"] == original
    assert bundle["upload_candidates"][0]["current_file"] == derived


def test_event_correlator_normalizes_known_windows_monitor_path_typo() -> None:
    canonical = (
        "D:/DataLeakDetector/DataLeakDetector-main/ScreenMonitor/"
        "windows_monitor/test_files/confidential_terms.docx"
    )
    logged = canonical.replace("/windows_monitor/", "/winows_monitor/")

    bundle = EventCorrelator().run(
        {
            "log_events": [
                {
                    "timestamp": "2026-06-05T22:41:00",
                    "event_type": "modified",
                    "file_path": logged,
                    "process_info": {"process_name": "wps.exe"},
                }
            ],
            "frame_segments": [],
            "sensitive_files": [canonical],
        }
    )

    assert bundle["correlated_events"][0]["original_file"] == canonical
    assert bundle["correlated_events"][0]["current_file"] == logged


def test_visual_send_uses_derived_file_created_before_observation() -> None:
    original = "D:/test_files/confidential_terms.docx"
    derived = "C:/Users/alice/Desktop/ordinary_terms.docx"
    records = [
        {
            "timestamp": "2026-06-05T22:41:00",
            "event_type": "modified",
            "file_path": original,
            "process_info": {"process_name": "wps.exe"},
        },
        {
            "timestamp": "2026-06-05T22:41:10",
            "event_type": "created",
            "file_path": derived,
            "process_info": {"process_name": "wps.exe"},
        },
    ]
    observations = [
        {
            "observation_id": "vlm_send",
            "start_ms": parse_timestamp_ms("2026-06-05T22:41:30"),
            "end_ms": parse_timestamp_ms("2026-06-05T22:41:30"),
            "app_name": "WeChat",
            "operation_type": "external_sink_interaction",
            "resource": "confidential_terms.docx",
            "related_resources": ["confidential_terms.docx"],
            "description": (
                "direct_leak: file_send. sink_type=chat_upload. action_status=completed. "
                "The sensitive file confidential_terms.docx was sent."
            ),
            "confidence": 0.95,
            "source": "vlm",
        }
    ]

    bundle = EventCorrelator().run(
        {
            "log_events": records,
            "frame_segments": observations,
            "sensitive_files": [original],
        }
    )

    assert bundle["upload_candidates"][0]["original_file"] == original
    assert bundle["upload_candidates"][0]["current_file"] == derived


def test_event_correlator_allows_same_process_screenshot_lineage() -> None:
    original = "C:/Users/alice/Desktop/salary.xlsx"
    screenshot = "C:/Users/alice/Pictures/Screenshots/Screenshot.png"
    records = [
        {
            "timestamp": "2026-06-05T22:41:00",
            "event_type": "modified",
            "file_path": original,
            "process_info": {"process_name": "wps.exe"},
        },
        {
            "timestamp": "2026-06-05T22:41:10",
            "event_type": "clipboard_image",
            "process_info": {"process_name": "wps.exe"},
            "extra": {"raw_operation": "clipboard_image"},
        },
        {
            "timestamp": "2026-06-05T22:41:11",
            "event_type": "created",
            "file_path": screenshot,
            "process_info": {"process_name": "wps.exe"},
            "extra": {"raw_operation": "created"},
        },
    ]

    bundle = EventCorrelator().run(
        {"log_events": records, "frame_segments": [], "sensitive_files": [original]}
    )

    assert bundle["file_lineage"]["direct_file_mappings"] == {screenshot: original}


@pytest.mark.parametrize(
    ("source_to_clipboard_seconds", "target_delay_seconds", "target", "window_title"),
    [
        (121.0, 1.0, "C:/Users/alice/Desktop/draft.txt", "draft.txt - Notepad"),
        (10.0, 16.0, "C:/Users/alice/Desktop/draft.txt", "draft.txt - Notepad"),
        (10.0, 1.0, "C:/Users/alice/Desktop/draft.txt", "notes.txt - Notepad"),
        (
            10.0,
            1.0,
            "C:/Users/alice/AppData/Local/Packages/Notepad/draft.txt",
            "draft.txt - Notepad",
        ),
    ],
)
def test_event_correlator_rejects_unbounded_or_unverified_clipboard_targets(
    source_to_clipboard_seconds: float,
    target_delay_seconds: float,
    target: str,
    window_title: str,
) -> None:
    original = "C:/Users/alice/Desktop/merger_notes.docx"
    target_time = source_to_clipboard_seconds + target_delay_seconds
    records = [
        {
            "timestamp": "2026-06-05T22:41:00",
            "event_type": "modified",
            "file_path": "C:/Users/alice/AppData/Roaming/Office/state.json",
            "process_info": {"process_name": "wps.exe"},
            "window_info": {"window_title": "merger_notes.docx - WPS Office"},
            "extra": {"relative_timestamp": 0.0, "raw_operation": "modified"},
        },
        {
            "timestamp": "2026-06-05T22:43:01",
            "event_type": "clipboard_text",
            "process_info": {"process_name": "wps.exe"},
            "extra": {
                "relative_timestamp": source_to_clipboard_seconds,
                "raw_operation": "clipboard_text",
            },
        },
        {
            "timestamp": "2026-06-05T22:43:02",
            "event_type": "modified",
            "file_path": target,
            "process_info": {"process_name": "Notepad.exe"},
            "window_info": {"window_title": window_title},
            "extra": {"relative_timestamp": target_time, "raw_operation": "modified"},
        },
    ]

    bundle = EventCorrelator().run(
        {"log_events": records, "frame_segments": [], "sensitive_files": [original]}
    )

    assert bundle["file_lineage"]["direct_file_mappings"] == {}


def test_office_backup_with_sensitive_stem_is_not_a_derived_artifact() -> None:
    original = "C:/Users/alice/Desktop/employee_salary_q4.xlsx"
    visible_copy = "C:/Users/alice/Desktop/employee_salary_q4_part1.xlsx"
    office_backup = (
        "C:/Users/alice/AppData/Roaming/kingsoft/office6/backup/"
        "employee_salary_q4_part1.xlsx.ABC123.20260603002354808.et"
    )
    records = [
        {
            "timestamp": "2026-06-03T00:00:00",
            "event_type": "created",
            "file_path": visible_copy,
            "process_info": {"process_name": "wps.exe"},
            "window_info": {"window_title": "employee_salary_q4_part1.xlsx - WPS Office"},
            "extra": {"raw_operation": "created"},
        },
        {
            "timestamp": "2026-06-03T00:01:00",
            "event_type": "created",
            "file_path": office_backup,
            "process_info": {"process_name": "wps.exe"},
            "window_info": {"window_title": "employee_salary_q4_part1.xlsx - WPS Office"},
            "extra": {"raw_operation": "created"},
            "upload_detection": {
                "is_upload": True,
                "upload_type": "File Access",
                "original_file": office_backup,
            },
        },
    ]

    bundle = EventCorrelator().run(
        {"log_events": records, "frame_segments": [], "sensitive_files": [original]}
    )

    assert bundle["file_lineage"]["direct_file_mappings"] == {visible_copy: original}


def test_datalog_path_preserves_every_multi_hop_derivation() -> None:
    original = "C:/Users/alice/Documents/customer_salary.xlsx"
    derived_pdf = "C:/Users/alice/Desktop/customer_salary.pdf"
    derived_zip = "C:/Users/alice/Desktop/customer_salary.zip"
    records = [
        {
            "timestamp": "2026-06-28T10:00:00",
            "event_type": "file_open",
            "file_path": original,
            "process_info": {"process_name": "excel.exe"},
        },
        {
            "timestamp": "2026-06-28T10:00:10",
            "event_type": "print_to_pdf",
            "file_path": derived_pdf,
            "source_file": original,
            "destination_path": derived_pdf,
            "extra": {"raw_operation": "print_to_pdf"},
            "process_info": {"process_name": "wps.exe"},
        },
        {
            "timestamp": "2026-06-28T10:00:20",
            "event_type": "compressed",
            "file_path": derived_zip,
            "source_file": derived_pdf,
            "destination_path": derived_zip,
            "extra": {"raw_operation": "compress"},
            "process_info": {"process_name": "explorer.exe"},
        },
        {
            "timestamp": "2026-06-28T10:00:30",
            "event_type": "file_upload",
            "file_path": derived_zip,
            "process_info": {"process_name": "msedge.exe"},
            "window_info": {"window_title": "ChatGPT upload"},
        },
    ]

    bundle = EventCorrelator().run(
        {"log_events": records, "frame_segments": [], "sensitive_files": [original]}
    )
    engine = DatalogEngine()
    for fact in bundle["datalog_facts"]:
        engine.add_fact(fact["relation"], *fact["args"])
    leaks = engine.query_leak()

    assert bundle["file_lineage"]["direct_file_mappings"] == {
        derived_pdf: original,
        derived_zip: derived_pdf,
    }
    assert len(leaks) == 1
    assert leaks[0].leaked_file == derived_zip
    assert leaks[0].full_path.count(":transfer:") >= 2


def test_event_correlator_uses_real_source_and_destination_fields() -> None:
    original = "C:/Users/alice/Documents/salary.xlsx"
    derived = "C:/Users/alice/Desktop/exported_salary.xlsx"
    records = [
        {
            "timestamp": "2026-01-01T00:00:00",
            "event_type": "created",
            "file_path": original,
            "destination_path": derived,
            "source_file": original,
            "extra": {"raw_operation": "export", "source": "watchdog_fs_monitor"},
            "process_info": {"process_name": "excel.exe"},
        },
        {
            "timestamp": "2026-01-01T00:00:10",
            "event_type": "file_selected",
            "file_path": derived,
            "extra": {"category": "文件上传", "source": "file_dialog_monitor"},
            "process_info": {"process_name": "msedge.exe"},
            "window_info": {"window_title": "Gmail attach"},
        },
    ]

    bundle = EventCorrelator().run({"log_events": records, "frame_segments": [], "sensitive_files": [original]})

    assert bundle["file_lineage"]["direct_file_mappings"][derived] == original
    assert any(item["current_file"] == derived for item in bundle["correlated_events"])


def test_removable_media_temp_copy_derives_from_unique_sensitive_stem() -> None:
    original = "C:/Users/alice/Documents/strategy.docx"
    derived = "C:/Users/alice/AppData/Local/Temp/strategy_{a1b2c3}.docx"
    records = [
        {
            "timestamp": "2026-01-01T00:00:46",
            "event_type": "created",
            "file_path": derived,
            "extra": {"raw_operation": "created", "relative_timestamp": 46.0},
            "process_info": {"process_name": "explorer.exe"},
            "window_info": {"window_title": "USB drive (F:) - File Explorer"},
        }
    ]
    logs = normalize_logs(records)

    bundle = EventCorrelator().run({"log_events": records, "frame_segments": [], "sensitive_files": [original]})
    windows = build_analysis_windows(logs, [original], VisionConfig())

    assert bundle["file_lineage"]["direct_file_mappings"][derived] == original
    strong = next(window for window in windows if window.priority == "strong")
    assert strong.anchor_ms == (46_000,)
    assert strong.action_anchor_ms == (46_000,)


def test_recent_sensitive_process_context_does_not_create_unrelated_lineage_edges() -> None:
    original = "C:/Users/alice/Documents/strategy.docx"
    derived = "C:/Users/alice/AppData/Local/Temp/strategy_{a1b2c3}.docx"
    unrelated = "C:/Users/alice/AppData/Local/Temp/browser_cache.bin"
    records = [
        {
            "timestamp": "2026-01-01T00:00:46",
            "event_type": "created",
            "file_path": derived,
            "extra": {"raw_operation": "created"},
            "process_info": {"process_name": "explorer.exe"},
            "window_info": {"window_title": "USB drive (F:) - File Explorer"},
        },
        {
            "timestamp": "2026-01-01T00:00:47",
            "event_type": "created",
            "file_path": unrelated,
            "extra": {"raw_operation": "copy"},
            "process_info": {"process_name": "explorer.exe"},
            "window_info": {"window_title": "USB drive (F:) - File Explorer"},
        },
    ]

    bundle = EventCorrelator().run({"log_events": records, "frame_segments": [], "sensitive_files": [original]})

    assert bundle["file_lineage"]["direct_file_mappings"] == {derived: original}


def test_event_correlator_keeps_log_lineage_in_vlm_only_mode() -> None:
    original = "C:/Users/alice/Documents/salary.xlsx"
    derived = "C:/Users/alice/Desktop/salary_export.pdf"
    records = [
        {
            "timestamp": "2026-01-01T00:00:00",
            "event_type": "print_to_pdf",
            "file_path": derived,
            "extra": {"source_path": original, "output_path": derived, "raw_operation": "print_to_pdf"},
            "process_info": {"process_name": "excel.exe"},
        }
    ]
    observations = [
        {
            "observation_id": "vlm_0",
            "start_ms": 0,
            "end_ms": 0,
            "app_name": "ChatGPT",
            "operation_type": "external_sink_interaction",
            "resource": derived,
            "description": "Uploaded salary_export.pdf to an external AI chat.",
            "confidence": 0.93,
            "source": "vlm",
        }
    ]

    bundle = EventCorrelator().run(
        {
            "log_events": records,
            "frame_segments": observations,
            "sensitive_files": [original],
            "non_vlm_enabled": False,
        }
    )

    assert bundle["file_lineage"]["direct_file_mappings"][derived] == original
    assert bundle["correlated_events"][0]["original_file"] == original
    assert bundle["correlated_events"][0]["current_file"] == derived


def test_lineage_prefers_unique_full_path_over_duplicate_basename_mapping() -> None:
    original = "C:/Users/alice/Documents/salary.xlsx"
    derived = "C:/Users/alice/Pictures/Screenshots/salary.png"
    lineage = Lineage()
    lineage.add(derived, original)
    lineage.add("salary.png", original)

    assert lineage.resolve_artifact("salary.png") == derived


def test_visual_upload_keeps_unique_full_derived_artifact_path() -> None:
    original = "C:/Users/alice/Documents/salary.xlsx"
    derived = "C:/Users/alice/Pictures/Screenshots/salary.png"
    observations = [
        {
            "observation_id": "vlm_derived",
            "start_ms": 10_000,
            "end_ms": 10_000,
            "app_name": "Snipping Tool",
            "operation_type": "file_or_content_transfer",
            "resource": derived,
            "related_resources": [original, derived],
            "description": "hidden_transfer: screenshot. action_status=completed.",
            "confidence": 0.95,
            "source": "vlm",
        },
        {
            "observation_id": "vlm_upload",
            "start_ms": 20_000,
            "end_ms": 20_000,
            "app_name": "Doubao",
            "operation_type": "external_sink_interaction",
            "resource": "salary.png",
            "related_resources": ["salary.png"],
            "description": "direct_leak: file_upload. sink_type=ai_chat. action_status=submitted.",
            "confidence": 0.95,
            "source": "vlm",
        },
    ]

    bundle = EventCorrelator().run(
        {
            "log_events": [],
            "frame_segments": observations,
            "sensitive_files": [original],
            "non_vlm_enabled": False,
        }
    )

    assert len(bundle["upload_candidates"]) == 1
    assert bundle["upload_candidates"][0]["current_file"] == derived
    assert "salary.png" not in bundle["file_lineage"]["direct_file_mappings"]


def test_event_correlator_resolves_extensionless_upload_selection() -> None:
    original = "C:/Users/alice/Desktop/company_contract.docx"
    records = [
        {
            "timestamp": "2026-01-01T00:01:00",
            "event_type": "file_selected",
            "file_path": "company_contract",
            "file_name": "company_contract",
            "file_extension": "",
            "extra": {"raw_operation": "file_selected", "category": "文件上传", "source": "file_dialog_monitor"},
            "process_info": {"process_name": "outlook.exe"},
        }
    ]

    bundle = EventCorrelator().run({"log_events": records, "frame_segments": [], "sensitive_files": [original]})
    engine = DatalogEngine()
    for fact in bundle["datalog_facts"]:
        engine.add_fact(fact["relation"], *fact["args"])

    assert bundle["upload_candidates"]
    assert bundle["upload_candidates"][0]["risk_level"] == "selected_or_attached"
    leaks = engine.query_leak()
    assert leaks
    assert leaks[0].leaked_file == original


def test_event_correlator_confirms_file_upload_events() -> None:
    original = "C:/Users/alice/Desktop/company_contract.docx"
    records = [
        {
            "timestamp": "2026-01-01T00:01:00",
            "event_type": "file_upload",
            "file_path": "company_contract.docx",
            "extra": {"raw_operation": "file_upload", "category": "文件上传", "source": "file_dialog_monitor"},
            "process_info": {"process_name": "outlook.exe"},
        }
    ]

    bundle = EventCorrelator().run({"log_events": records, "frame_segments": [], "sensitive_files": [original]})
    engine = DatalogEngine()
    for fact in bundle["datalog_facts"]:
        engine.add_fact(fact["relation"], *fact["args"])

    leaks = engine.query_leak()

    assert bundle["upload_candidates"]
    assert bundle["upload_candidates"][0]["risk_level"] == "completed"
    assert leaks
    assert leaks[0].leaked_file == original


def test_event_correlator_fuses_visual_file_identity_with_later_send_result() -> None:
    original = "C:/Users/alice/Documents/company_contract.docx"
    parsed = parse_vlm_response_detailed(
        json.dumps(
            {
                "events": [
                    {
                        "evidence_frame_ids": ["frame_identity"],
                        "timestamp_ms": 10_000,
                        "app_name": "Outlook Mail",
                        "behavior_category": "normal",
                        "operation_type": "file_selected",
                        "original_filename": "company_contract.docx",
                        "description": "The exact filename is visible in the chooser.",
                        "action_status": "selected",
                    },
                    {
                        "evidence_frame_ids": ["frame_result"],
                        "timestamp_ms": 25_000,
                        "app_name": "Outlook Mail",
                        "behavior_category": "direct_leak",
                        "operation_type": "email_send",
                        "original_filename": "unknown",
                        "sink_type": "mail_attachment",
                        "description": "The sent-message confirmation is visible.",
                        "action_status": "completed",
                    },
                ]
            }
        ),
        keywords=[original],
    )
    assert len(parsed.events) == 2

    bundle = EventCorrelator().run(
        {
            "log_events": [],
            "frame_segments": vision_events_to_observations(parsed.events),
            "sensitive_files": [original],
            "non_vlm_enabled": False,
        }
    )
    assert len(bundle["upload_candidates"]) == 1
    upload = bundle["upload_candidates"][0]
    assert upload["original_file"] == original
    assert upload["current_file"] == original
    assert upload["sink_type"] == "mail_attachment"
    assert {"frame:vlm_0", "frame:vlm_1"}.issubset(upload["evidence_refs"])

    engine = DatalogEngine()
    for fact in bundle["datalog_facts"]:
        engine.add_fact(fact["relation"], *fact["args"])
    leaks = engine.query_leak()
    assert len(leaks) == 1
    assert leaks[0].leaked_file == original
    assert ":open" in leaks[0].full_path
    assert ":leak" in leaks[0].full_path


def test_event_correlator_preserves_distinct_sink_channels_for_same_file() -> None:
    original = "C:/Users/alice/Documents/company_contract.docx"
    observations = [
        {
            "observation_id": "vlm_chat",
            "start_ms": 10_000,
            "end_ms": 10_000,
            "app_name": "Tencent Meeting",
            "operation_type": "external_sink_interaction",
            "resource": original,
            "description": (
                "direct_leak: file_send. sink_type=chat_upload. "
                "action_status=completed. The file was sent to the meeting chat."
            ),
            "confidence": 0.95,
            "source": "vlm",
        },
        {
            "observation_id": "vlm_share",
            "start_ms": 20_000,
            "end_ms": 20_000,
            "app_name": "Tencent Meeting",
            "operation_type": "external_sink_interaction",
            "resource": original,
            "description": (
                "direct_leak: screen_share. sink_type=screen_share. "
                "action_status=in_progress. The document is visible while sharing."
            ),
            "confidence": 0.9,
            "source": "vlm",
        },
    ]

    bundle = EventCorrelator().run(
        {
            "log_events": [],
            "frame_segments": observations,
            "sensitive_files": [original],
            "non_vlm_enabled": False,
        }
    )

    assert {item["sink_type"] for item in bundle["upload_candidates"]} == {"chat_upload", "screen_share"}
    engine = DatalogEngine()
    for fact in bundle["datalog_facts"]:
        engine.add_fact(fact["relation"], *fact["args"])
    assert {item.leak_channel for item in engine.query_leak()} == {"chat_upload", "screen_share"}


def test_event_correlator_uses_log_identity_for_filename_free_vlm_sink() -> None:
    original = "C:/Users/alice/Documents/company_contract.docx"
    observations = [
        {
            "observation_id": "vlm_0",
            "start_ms": 20_000,
            "end_ms": 20_000,
            "app_name": "ChatGPT",
            "operation_type": "external_sink_interaction",
            "resource": "",
            "description": (
                "direct_leak: ai_chat_upload. evidence_frame_ids=frame_result. "
                "sink_type=ai_chat. action_status=completed. The submitted prompt is visible."
            ),
            "confidence": 0.93,
            "source": "vlm",
        }
    ]
    records = [
        {
            "event_id": "identity_log",
            "timestamp": "2026-01-01T00:00:10",
            "event_type": "file_open",
            "file_path": original,
            "process_info": {"process_name": "winword.exe"},
            "extra": {"relative_timestamp": 10.0},
        }
    ]

    bundle = EventCorrelator().run(
        {
            "log_events": records,
            "frame_segments": observations,
            "sensitive_files": [original],
            "non_vlm_enabled": False,
        }
    )
    assert len(bundle["upload_candidates"]) == 1
    upload = bundle["upload_candidates"][0]
    assert upload["original_file"] == original
    assert {"log:identity_log", "frame:vlm_0"}.issubset(upload["evidence_refs"])

    engine = DatalogEngine()
    for fact in bundle["datalog_facts"]:
        engine.add_fact(fact["relation"], *fact["args"])
    leaks = engine.query_leak()
    assert len(leaks) == 1
    assert leaks[0].leaking_proc == "ChatGPT"
    assert leaks[0].leaked_file == original


def test_failed_visual_upload_still_creates_leak_fact_for_attempted_upload() -> None:
    original = "C:/Users/alice/Documents/company_contract.docx"
    parsed = parse_vlm_response_detailed(
        json.dumps(
            {
                "events": [
                    {
                        "evidence_frame_ids": ["frame_failed"],
                        "timestamp_ms": 30_000,
                        "app_name": "Gemini",
                        "behavior_category": "direct_leak",
                        "operation_type": "file_upload",
                        "original_filename": "company_contract.docx",
                        "sink_type": "ai_chat",
                        "action_status": "failed",
                        "description": "Gemini displays: File types are not supported.",
                    }
                ]
            }
        ),
        keywords=[original],
    )
    assert parsed.events[0].action_status == "failed"
    observations = vision_events_to_observations(parsed.events)
    assert "action_status=failed" in observations[0].description

    bundle = EventCorrelator().run(
        {
            "log_events": [],
            "frame_segments": observations,
            "sensitive_files": [original],
            "non_vlm_enabled": False,
        }
    )
    assert bundle["correlated_events"][0]["behavior_category"] == "failed_external_attempt"
    assert bundle["upload_candidates"][0]["risk_level"] == "selected_or_attached"
    relations = [fact["relation"] for fact in bundle["datalog_facts"]]
    assert "SuspiciousBehavior" in relations
    assert "LeakFile" in relations

    engine = DatalogEngine()
    for fact in bundle["datalog_facts"]:
        engine.add_fact(fact["relation"], *fact["args"])
    leaks = engine.query_leak()
    assert len(leaks) == 1
    assert leaks[0].leaked_file == original


def test_event_correlator_treats_removable_media_copy_as_leak() -> None:
    original = "C:/Users/alice/Desktop/company_contract.docx"
    target = "E:/USB/company_contract.docx"
    records = [
        {
            "timestamp": "2026-01-01T00:00:00",
            "event_type": "file_open",
            "file_path": original,
            "process_info": {"process_name": "winword.exe"},
        },
        {
            "timestamp": "2026-01-01T00:00:10",
            "event_type": "created",
            "file_path": target,
            "source_file": original,
            "destination_path": target,
            "extra": {"raw_operation": "copy", "category": "copy to USB removable drive"},
            "process_info": {"process_name": "explorer.exe"},
            "window_info": {"window_title": "Copying to USB Drive (E:)"},
        },
    ]

    bundle = EventCorrelator().run({"log_events": records, "frame_segments": [], "sensitive_files": [original]})
    engine = DatalogEngine()
    for fact in bundle["datalog_facts"]:
        engine.add_fact(fact["relation"], *fact["args"])

    leaks = engine.query_leak()

    assert bundle["upload_candidates"]
    assert bundle["upload_candidates"][0]["sink_type"] == "removable_media"
    assert bundle["upload_candidates"][0]["risk_level"] == "completed"
    assert leaks
    assert leaks[0].leak_channel == "removable_media"


def test_event_correlator_does_not_infer_initial_sensitive_files_from_logs() -> None:
    bundle = EventCorrelator().run(
        {
            "session_id": "unit",
            "log_events": _records(),
            "frame_segments": [],
            "sensitive_files": [],
        }
    )

    assert bundle["statistics"]["sensitive_files"] == 0
    assert bundle["datalog_facts"] == []


def test_event_correlator_ignores_placeholder_sensitive_sources() -> None:
    records = [
        {
            "timestamp": "2026-01-01T00:00:00",
            "event_type": "created",
            "file_path": "C:/Users/alice/AppData/Local/Google/Chrome/User Data/Default/Cache/Cache_Data/f_000123",
            "process_info": {"process_name": "chrome.exe"},
        },
        {
            "timestamp": "2026-01-01T00:00:01",
            "event_type": "file_upload",
            "file_path": "N/A",
            "process_info": {"process_name": "chrome.exe"},
            "extra": {"category": "upload"},
        },
    ]

    bundle = EventCorrelator().run({"log_events": records, "frame_segments": [], "sensitive_files": ["N/A"]})

    assert same_file("N/A", "N/A") is False
    assert bundle["upload_candidates"] == []
    assert bundle["datalog_facts"] == []


def test_event_correlator_does_not_promote_browser_cache_noise_to_upload() -> None:
    original = "C:/Users/alice/Desktop/secret.docx"
    cache_file = "C:/Users/alice/AppData/Local/Google/Chrome/User Data/Default/Cache/Cache_Data/f_000123"
    records = [
        {
            "timestamp": "2026-01-01T00:00:00",
            "event_type": "file_open",
            "file_path": original,
            "process_info": {"process_name": "winword.exe"},
        },
        {
            "timestamp": "2026-01-01T00:00:01",
            "event_type": "created",
            "file_path": cache_file,
            "source_file": original,
            "process_info": {"process_name": "chrome.exe"},
            "extra": {"raw_operation": "upload"},
        },
        {
            "timestamp": "2026-01-01T00:00:02",
            "event_type": "file_upload",
            "file_path": cache_file,
            "process_info": {"process_name": "chrome.exe"},
            "extra": {"category": "upload"},
        },
    ]

    bundle = EventCorrelator().run({"log_events": records, "frame_segments": [], "sensitive_files": [original]})

    assert bundle["upload_candidates"] == []
    assert not any(fact["relation"] == "LeakFile" for fact in bundle["datalog_facts"])


def test_event_correlator_keeps_real_file_inside_screenmonitor_tree_but_filters_capture_artifacts() -> None:
    original = "D:/workspace/ScreenMonitor/windows_monitor/test_files/company_secret.docx"
    capture_log = "D:/workspace/ScreenMonitor/windows_monitor/recordings/session_20260101/logs/logs.json"
    observations = [
        {
            "observation_id": "vlm_real",
            "start_ms": 10_000,
            "end_ms": 10_000,
            "app_name": "Cloud Drive",
            "operation_type": "external_sink_interaction",
            "resource": original,
            "related_resources": [original],
            "description": "direct_leak: upload completed. action_status=completed. sink_type=cloud_sync.",
            "confidence": 0.95,
            "source": "vlm",
        },
        {
            "observation_id": "vlm_capture_log",
            "start_ms": 20_000,
            "end_ms": 20_000,
            "app_name": "Cloud Drive",
            "operation_type": "external_sink_interaction",
            "resource": capture_log,
            "related_resources": [original],
            "description": "direct_leak: collector log upload completed. action_status=completed. sink_type=cloud_sync.",
            "confidence": 0.95,
            "source": "vlm",
        },
    ]

    bundle = EventCorrelator().run(
        {
            "log_events": [],
            "frame_segments": observations,
            "sensitive_files": [original],
            "non_vlm_enabled": False,
        }
    )

    assert [item["current_file"] for item in bundle["upload_candidates"]] == [original]


def test_visual_sensitive_mention_prefers_longest_specific_filename() -> None:
    shorter = "C:/Users/alice/Desktop/公司机密.docx"
    specific = "D:/workspace/ScreenMonitor/windows_monitor/test_files/公司机密条款.docx"
    derived = "公司机密条款.zip"
    observations = [
        {
            "observation_id": "vlm_specific",
            "start_ms": 10_000,
            "end_ms": 10_000,
            "app_name": "File Explorer",
            "operation_type": "file_or_content_transfer",
            "resource": derived,
            "related_resources": ["公司机密条款.docx", derived],
            "description": "hidden_transfer: 公司机密条款.docx was compressed to 公司机密条款.zip.",
            "confidence": 0.9,
            "source": "vlm",
        }
    ]

    bundle = EventCorrelator().run(
        {
            "log_events": [],
            "frame_segments": observations,
            "sensitive_files": [shorter, specific],
            "non_vlm_enabled": False,
        }
    )

    assert bundle["correlated_events"][0]["original_file"] == specific
    assert bundle["file_lineage"]["direct_file_mappings"][derived] == specific


def test_visual_upload_candidate_leaks_original_when_frame_only_has_basename() -> None:
    original = "C:/Users/alice/Desktop/secret.docx"
    observations = [
        {
            "observation_id": "vlm_0",
            "start_ms": 10_000,
            "end_ms": 10_000,
            "app_name": "Gmail",
            "operation_type": "external_sink_interaction",
            "resource": "secret.docx",
            "related_resources": ["secret.docx"],
            "description": "direct_leak: email attachment upload",
            "confidence": 0.95,
            "source": "vlm",
        }
    ]

    bundle = EventCorrelator().run({"log_events": [], "frame_segments": observations, "sensitive_files": [original]})
    engine = DatalogEngine()
    for fact in bundle["datalog_facts"]:
        engine.add_fact(fact["relation"], *fact["args"])

    leaks = engine.query_leak()

    assert bundle["upload_candidates"]
    assert leaks
    assert leaks[0].leaked_file == original


def test_visual_upload_candidate_uses_sensitive_file_when_context_resource_differs() -> None:
    original = "C:/Users/alice/OneDrive/product_plan.docx"
    observations = [
        {
            "observation_id": "vlm_0",
            "start_ms": 10_000,
            "end_ms": 10_000,
            "app_name": "VS Code",
            "operation_type": "external_sink_interaction",
            "resource": "config.yaml",
            "related_resources": ["config.yaml"],
            "description": "direct_leak: AI chat upload. The chat references product_plan.docx as an attached sensitive file.",
            "confidence": 0.95,
            "source": "vlm",
        }
    ]

    bundle = EventCorrelator().run({"log_events": [], "frame_segments": observations, "sensitive_files": [original]})

    assert bundle["upload_candidates"]
    assert bundle["upload_candidates"][0]["current_file"] == original


def test_unbound_visual_direct_leak_becomes_suspicious_behavior() -> None:
    observations = [
        {
            "observation_id": "vlm_0",
            "start_ms": 10_000,
            "end_ms": 10_000,
            "app_name": "Outlook",
            "operation_type": "external_sink_interaction",
            "resource": "unknown",
            "related_resources": [],
            "description": "direct_leak: email_send. Send confirmation is visible.",
            "confidence": 0.9,
            "source": "vlm",
        }
    ]

    bundle = EventCorrelator().run(
        {"log_events": [], "frame_segments": observations, "sensitive_files": []}
    )

    assert bundle["upload_candidates"] == []
    suspicious = [fact for fact in bundle["datalog_facts"] if fact["relation"] == "SuspiciousBehavior"]
    assert len(suspicious) == 1


def test_actionless_unknown_risk_does_not_inherit_prior_sensitive_identity() -> None:
    original = "C:/Users/alice/Documents/salary.xlsx"
    observations = [
        {
            "observation_id": "vlm_identity",
            "start_ms": 10_000,
            "end_ms": 10_000,
            "app_name": "Excel",
            "operation_type": "file_or_content_transfer",
            "resource": original,
            "related_resources": [original],
            "description": "hidden_transfer: copy. action_status=completed.",
            "confidence": 0.9,
            "source": "vlm",
        },
        {
            "observation_id": "vlm_unknown_app",
            "start_ms": 20_000,
            "end_ms": 20_000,
            "app_name": "Doubao",
            "operation_type": "external_sink_interaction",
            "resource": "unknown",
            "related_resources": [],
            "description": "unknown_risk: ai_chat. action_status=unknown. No outbound action is visible.",
            "confidence": 0.6,
            "source": "vlm",
        },
    ]

    bundle = EventCorrelator().run(
        {
            "log_events": [],
            "frame_segments": observations,
            "sensitive_files": [original],
            "non_vlm_enabled": False,
        }
    )

    assert bundle["upload_candidates"] == []
    assert not any(item["app_name"] == "Doubao" for item in bundle["correlated_events"])


def test_recent_sensitive_clipboard_then_ai_app_switch_confirms_visual_sink() -> None:
    original = "C:/Users/alice/Desktop/strategy.docx"
    records = [
        {
            "timestamp": "2026-01-01T00:00:10",
            "event_type": "clipboard_text",
            "content_preview": "confidential strategy content",
            "process_info": {"process_name": "WINWORD.EXE"},
            "extra": {"raw_operation": "clipboard_text", "source": "clipboard_monitor"},
        },
        {
            "timestamp": "2026-01-01T00:00:15",
            "event_type": "app_switch",
            "file_path": original,
            "process_info": {"process_name": "msedge.exe"},
            "window_info": {"window_title": "Poe - AI Chat Platform"},
            "extra": {"raw_operation": "app", "source": "window_monitor"},
        },
    ]
    observations = [
        {
            "observation_id": "vlm_poe",
            "start_ms": 15_000,
            "end_ms": 15_000,
            "app_name": "Microsoft Edge",
            "operation_type": "external_sink_interaction",
            "resource": "strategy.docx",
            "related_resources": ["strategy.docx"],
            "description": (
                "unknown_risk: ai_chat_navigation. sink_type=ai_chat. "
                "action_status=unknown. No paste is visible in the selected keyframes."
            ),
            "confidence": 0.6,
            "source": "vlm",
        }
    ]

    bundle = EventCorrelator().run(
        {"log_events": records, "frame_segments": observations, "sensitive_files": [original]}
    )

    upload = bundle["upload_candidates"][0]
    assert upload["original_file"] == original
    assert upload["current_file"] == original
    assert upload["sink_type"] == "ai_chat"


def test_recent_sensitive_clipboard_then_generic_browser_switch_is_not_sink() -> None:
    original = "C:/Users/alice/Desktop/strategy.docx"
    records = [
        {
            "timestamp": "2026-01-01T00:00:10",
            "event_type": "clipboard_text",
            "content_preview": "confidential strategy content",
            "process_info": {"process_name": "WINWORD.EXE"},
            "extra": {"raw_operation": "clipboard_text", "source": "clipboard_monitor"},
        },
        {
            "timestamp": "2026-01-01T00:00:15",
            "event_type": "app_switch",
            "file_path": original,
            "process_info": {"process_name": "msedge.exe"},
            "window_info": {"window_title": "Data Leak Monitoring System"},
            "extra": {"raw_operation": "app", "source": "window_monitor"},
        },
    ]

    bundle = EventCorrelator().run(
        {"log_events": records, "frame_segments": [], "sensitive_files": [original]}
    )

    assert bundle["upload_candidates"] == []
    assert not any(fact["relation"] == "LeakFile" for fact in bundle["datalog_facts"])


def test_visible_cloud_menu_is_preparation_not_confirmed_upload() -> None:
    original = "C:/Users/alice/Documents/salary.xlsx"
    parsed = parse_vlm_response_detailed(
        json.dumps(
            {
                "events": [
                    {
                        "evidence_frame_ids": ["frame_menu"],
                        "timestamp_ms": 10_000,
                        "app_name": "WPS Office",
                        "behavior_category": "direct_leak",
                        "operation_type": "cloud_upload",
                        "original_filename": "salary.xlsx",
                        "sink_type": "cloud_sync",
                        "action_status": "selected",
                        "description": (
                            "The Upload to cloud document menu was opened, indicating an intent "
                            "to sync the file. No upload progress is visible."
                        ),
                    }
                ]
            }
        ),
        keywords=[original],
    )

    assert len(parsed.events) == 1
    assert parsed.events[0].behavior_category == "unknown_risk"
    assert parsed.events[0].action_status == "unknown"
    bundle = EventCorrelator().run(
        {
            "log_events": [],
            "frame_segments": vision_events_to_observations(parsed.events),
            "sensitive_files": [original],
            "non_vlm_enabled": False,
        }
    )
    assert bundle["upload_candidates"] == []


def test_visual_unknown_risk_with_progress_does_not_become_upload() -> None:
    original = "C:/Users/alice/Documents/secret.docx"
    observations = [
        {
            "observation_id": "vlm_monitor",
            "start_ms": 10_000,
            "end_ms": 10_000,
            "app_name": "Data Leak Monitoring System",
            "operation_type": "external_sink_interaction",
            "resource": "secret.docx",
            "related_resources": ["secret.docx"],
            "description": (
                "unknown_risk: monitoring progress. sink_type=unknown. "
                "action_status=in_progress. No outbound transfer is confirmed."
            ),
            "confidence": 0.8,
            "source": "vlm",
        }
    ]

    bundle = EventCorrelator().run(
        {"log_events": [], "frame_segments": observations, "sensitive_files": [original]}
    )

    assert bundle["upload_candidates"] == []
    assert not any(fact["relation"] == "LeakFile" for fact in bundle["datalog_facts"])


def test_cloud_folder_membership_without_file_sync_status_is_not_upload() -> None:
    original = "C:/Users/alice/OneDrive/Desktop/secret.docx"
    observations = [
        {
            "observation_id": "vlm_cloud_folder",
            "start_ms": 10_000,
            "end_ms": 10_000,
            "app_name": "File Explorer",
            "operation_type": "external_sink_interaction",
            "resource": "secret.docx",
            "related_resources": ["secret.docx"],
            "description": (
                "direct_leak: cloud_sync. sink_type=cloud_sync. action_status=completed. "
                "Files in the OneDrive folder are automatically synced to the cloud, "
                "but no file-specific sync status is visible."
            ),
            "confidence": 0.8,
            "source": "vlm",
        }
    ]

    bundle = EventCorrelator().run(
        {"log_events": [], "frame_segments": observations, "sensitive_files": [original]}
    )

    assert bundle["upload_candidates"] == []
    assert not any(fact["relation"] == "LeakFile" for fact in bundle["datalog_facts"])


def test_declared_ai_chat_sink_is_not_reclassified_as_generic_chat() -> None:
    original = "C:/Users/alice/Documents/secret.docx"
    observations = [
        {
            "observation_id": "vlm_ai",
            "start_ms": 10_000,
            "end_ms": 10_000,
            "app_name": "WPS Office",
            "operation_type": "external_sink_interaction",
            "resource": "secret.docx",
            "related_resources": ["secret.docx"],
            "description": (
                "direct_leak: copy_paste_to_ai. sink_type=ai_chat. "
                "action_status=submitted. Sensitive content was submitted."
            ),
            "confidence": 0.95,
            "source": "vlm",
        }
    ]

    bundle = EventCorrelator().run(
        {"log_events": [], "frame_segments": observations, "sensitive_files": [original]}
    )

    assert bundle["upload_candidates"][0]["sink_type"] == "ai_chat"


def test_explicit_ai_prompt_paste_remains_external_when_response_translates_content() -> None:
    event = ParsedVisionEvent(
        start_ms=10_000,
        end_ms=10_000,
        app_name="Doubao",
        behavior_category="direct_leak",
        operation_type="ai_prompt_paste",
        original_resource="strategy.docx",
        modified_resource="unknown",
        description="The content was submitted and the response translated it.",
        confidence=0.95,
        sink_type="ai_chat",
        action_status="submitted",
    )

    observations = vision_events_to_observations([event])

    assert observations[0].operation_type == "external_sink_interaction"


def test_clipboard_text_containing_usb_case_name_is_not_removable_transfer() -> None:
    original = "D:/sensitive/company_contract.docx"
    records = [
        {
            "timestamp": "2026-01-01T00:00:00",
            "event_type": "clipboard_text",
            "file_path": original,
            "content_preview": "1-transfer-USB flash drive-3",
            "process_info": {"process_name": "QQ.exe"},
            "extra": {"raw_operation": "clipboard_text", "source": "clipboard_monitor"},
        }
    ]

    bundle = EventCorrelator().run(
        {"log_events": records, "frame_segments": [], "sensitive_files": [original]}
    )

    assert bundle["upload_candidates"] == []
    assert not any(fact["relation"] == "LeakFile" for fact in bundle["datalog_facts"])


def test_same_named_sensitive_sources_keep_exact_full_path_identity() -> None:
    first = "C:/Users/alice/Desktop/product_plan.docx"
    second = "D:/gdata/documents_1/product_plan.docx"
    records = [
        {
            "timestamp": "2026-01-01T00:00:00",
            "event_type": "file_selected",
            "file_path": second,
            "process_info": {"process_name": "msedge.exe"},
            "window_info": {"window_title": "CSDN upload"},
            "extra": {"raw_operation": "file_selected", "category": "file upload"},
        }
    ]

    bundle = EventCorrelator().run(
        {"log_events": records, "frame_segments": [], "sensitive_files": [first, second]}
    )

    assert bundle["upload_candidates"][0]["original_file"] == second


def test_vlm_removable_media_event_becomes_external_sink() -> None:
    response = json.dumps(
        {
            "events": [
                {
                    "evidence_frame_ids": ["frame_0"],
                    "timestamp_ms": 10_000,
                    "app_name": "Windows Explorer",
                    "behavior_category": "direct_leak",
                    "operation_type": "copy_to_removable_media",
                    "original_filename": "company_contract.docx",
                    "modified_filename": "E:/USB/company_contract.docx",
                    "sink_type": "removable_media",
                    "description": "Sensitive file is being copied to a USB removable drive.",
                    "confidence": 0.94,
                }
            ]
        }
    )

    observations = vision_events_to_observations(parse_vlm_response(response), source="vlm")

    assert observations[0].operation_type == "external_sink_interaction"
    assert "sink_type=removable_media" in observations[0].description


def test_vlm_hidden_transfer_does_not_become_external_sink_from_menu_text() -> None:
    response = json.dumps(
        {
            "events": [
                {
                    "evidence_frame_ids": ["frame_0"],
                    "timestamp_ms": 10_000,
                    "app_name": "Explorer",
                    "behavior_category": "hidden_transfer",
                    "operation_type": "copy_file",
                    "original_filename": "customer_contacts.pdf",
                    "modified_filename": "customer_contacts.pdf",
                    "sink_type": "unknown",
                    "description": "An unselected context menu merely shows Send to my phone.",
                    "confidence": 0.8,
                }
            ]
        }
    )

    observations = vision_events_to_observations(parse_vlm_response(response), source="vlm")

    assert observations[0].operation_type == "file_or_content_transfer"


@pytest.mark.parametrize(
    ("app_name", "operation_type", "action_status"),
    [
        ("Doubao AI", "copy_paste_to_ai", "completed"),
        ("DeepSeek", "ai_prompt_paste", "submitted"),
    ],
)
def test_vlm_structured_ai_chat_direct_leak_becomes_external_sink(
    app_name: str,
    operation_type: str,
    action_status: str,
) -> None:
    event = ParsedVisionEvent(
        start_ms=10_000,
        end_ms=10_000,
        app_name=app_name,
        behavior_category="direct_leak",
        operation_type=operation_type,
        original_resource="company_strategy.txt",
        modified_resource="unknown",
        description="The model produced an answer to the submitted content.",
        confidence=0.95,
        sink_type="ai_chat",
        action_status=action_status,
    )

    observations = vision_events_to_observations([event])

    assert observations[0].operation_type == "external_sink_interaction"


def test_vlm_direct_leak_with_explicit_outbound_action_becomes_external_sink_without_sink_type() -> None:
    event = ParsedVisionEvent(
        start_ms=10_000,
        end_ms=10_000,
        app_name="External service",
        behavior_category="direct_leak",
        operation_type="ai_prompt_paste",
        original_resource="company_strategy.txt",
        modified_resource="unknown",
        description="The request was accepted.",
        confidence=0.9,
        sink_type="unknown",
        action_status="submitted",
    )

    observations = vision_events_to_observations([event])

    assert observations[0].operation_type == "external_sink_interaction"


def test_vlm_hidden_selected_generic_send_panel_stays_content_transfer() -> None:
    event = ParsedVisionEvent(
        start_ms=10_000,
        end_ms=10_000,
        app_name="Generic transfer panel",
        behavior_category="hidden_transfer",
        operation_type="send_to_phone",
        original_resource="company_strategy.txt",
        modified_resource="unknown",
        description="The panel is visible, but no destination or transfer queue is shown.",
        confidence=0.8,
        sink_type="chat_upload",
        action_status="selected",
    )

    observations = vision_events_to_observations([event])

    assert observations[0].operation_type == "file_or_content_transfer"


def test_vlm_derived_lineage_links_later_upload_log_to_original() -> None:
    original = "C:/Users/alice/Documents/secret.docx"
    derived = "C:/Users/alice/Desktop/secret_screen.png"
    observations = [
        {
            "observation_id": "vlm_0",
            "start_ms": 10_000,
            "end_ms": 10_000,
            "app_name": "Snipping Tool",
            "operation_type": "file_or_content_transfer",
            "resource": derived,
            "related_resources": ["secret.docx", derived],
            "description": "hidden_transfer: screenshot derived from secret.docx",
            "confidence": 0.92,
            "source": "vlm",
        }
    ]
    records = [
        {
            "timestamp": "2026-01-01T00:06:00",
            "event_type": "file_selected",
            "file_path": derived,
            "extra": {"category": "upload", "raw_operation": "file_selected"},
            "process_info": {"process_name": "msedge.exe"},
            "window_info": {"window_title": "ChatGPT upload"},
        }
    ]

    bundle = EventCorrelator().run({"log_events": records, "frame_segments": observations, "sensitive_files": [original]})
    engine = DatalogEngine()
    for fact in bundle["datalog_facts"]:
        engine.add_fact(fact["relation"], *fact["args"])

    leaks = engine.query_leak()

    assert bundle["file_lineage"]["direct_file_mappings"][derived] == original
    assert bundle["upload_candidates"]
    assert bundle["upload_candidates"][0]["current_file"] == derived
    assert leaks


def test_source_named_split_directory_links_unique_wrapped_alias_to_later_upload() -> None:
    original = "C:/Users/alice/WPSDrive/team/产品设计机密.ksheet"
    derived = "C:/Users/alice/WPSDrive/team/产品设计机密/2.ksheet.wpsonline"
    unrelated_parent = "C:/Users/alice/Downloads/产品设计机密/3.ksheet.wpsonline"
    wrong_inner_type = "C:/Users/alice/WPSDrive/team/产品设计机密/notes.docx.wpsonline"
    records = [
        {
            "timestamp": "2026-01-01T00:00:00",
            "event_type": "opened",
            "file_path": original,
            "process_info": {"process_name": "wps.exe"},
        },
        {
            "timestamp": "2026-01-01T00:01:00",
            "event_type": "created",
            "file_path": derived,
            "process_info": {"process_name": "mailmaster.exe"},
            "window_info": {"window_title": "New mail"},
            "extra": {"raw_operation": "created"},
        },
        {
            "timestamp": "2026-01-01T00:01:01",
            "event_type": "created",
            "file_path": unrelated_parent,
            "process_info": {"process_name": "mailmaster.exe"},
            "extra": {"raw_operation": "created"},
        },
        {
            "timestamp": "2026-01-01T00:01:02",
            "event_type": "created",
            "file_path": wrong_inner_type,
            "process_info": {"process_name": "mailmaster.exe"},
            "extra": {"raw_operation": "created"},
        },
    ]
    observations = [
        {
            "observation_id": "vlm_mail",
            "start_ms": 1_767_225_720_000,
            "end_ms": 1_767_225_720_000,
            "app_name": "Mail Master",
            "operation_type": "external_sink_interaction",
            "resource": "2.ksheet",
            "related_resources": ["2.ksheet"],
            "description": "direct_leak: email attachment sent. action_status=completed. sink_type=mail_attachment.",
            "confidence": 0.95,
            "source": "vlm",
        }
    ]

    bundle = EventCorrelator().run(
        {
            "log_events": records,
            "frame_segments": observations,
            "sensitive_files": [original],
            "non_vlm_enabled": False,
        }
    )
    mappings = bundle["file_lineage"]["direct_file_mappings"]
    upload = bundle["upload_candidates"][0]
    engine = DatalogEngine()
    for fact in bundle["datalog_facts"]:
        engine.add_fact(fact["relation"], *fact["args"])

    assert mappings[derived] == original
    assert unrelated_parent not in mappings
    assert wrong_inner_type not in mappings
    assert upload["original_file"] == original
    assert upload["current_file"] == derived
    leaks = engine.query_leak()
    assert len(leaks) == 1
    assert leaks[0].leaked_file == derived


def test_datalog_engine_finds_derived_file_leak() -> None:
    engine = DatalogEngine(case_id="unit-case")
    engine.add_fact("OpenFile", "open_1", "excel.exe", "secret.xlsx", 1)
    engine.add_fact("TransferFile", "copy_1", "excel.exe", "secret.xlsx", "secret_copy.xlsx", 2)
    engine.add_fact("LeakFile", "upload_1", "excel.exe", "secret_copy.xlsx", "network", 3)

    leaks = engine.query_leak()

    assert len(leaks) == 1
    assert leaks[0].full_path == "open_1 -> copy_1 -> upload_1"
    assert {fact.case_id for facts in engine.facts.values() for fact in facts} == {"unit-case"}
    assert leaks[0].case_id == "unit-case"
    assert leaks[0].source_file == "secret.xlsx"
    assert leaks[0].file_chain == ("secret.xlsx", "secret_copy.xlsx")
    assert [item["relation"] for item in leaks[0].flow_steps] == [
        "OpenFile",
        "TransferFile",
        "LeakFile",
    ]


def test_datalog_engine_prefers_complete_lineage_over_earlier_shortcut() -> None:
    engine = DatalogEngine(case_id="case-with-lineage")
    engine.add_fact("OpenFile", "open_source", "case_lineage", "secret.docx", 100)
    engine.add_fact("TransferFile", "to_draft", "case_lineage", "secret.docx", "draft.txt", 110)
    engine.add_fact("TransferFile", "shortcut", "case_lineage", "secret.docx", "shot.png", 120)
    engine.add_fact("TransferFile", "to_shot", "case_lineage", "draft.txt", "shot.png", 130)
    engine.add_fact("CrossProcessTransfer", "sink_access", "case_lineage", "qq", "shot.png", 200)
    engine.add_fact("LeakFile", "send_shot", "qq", "shot.png", "chat_upload", 210)

    leak = engine.query_leak()[0]

    assert leak.full_path == "open_source -> to_draft -> to_shot -> sink_access -> send_shot"
    assert leak.file_chain == ("secret.docx", "draft.txt", "shot.png")
    assert [
        (item.get("source_file"), item.get("derived_file"))
        for item in leak.flow_steps
        if item["relation"] == "TransferFile"
    ] == [("secret.docx", "draft.txt"), ("draft.txt", "shot.png")]


def test_datalog_engine_enforces_forward_time_and_preserves_leak_timestamp() -> None:
    engine = DatalogEngine()
    engine.add_fact("OpenFile", "day1_open", "lineage", "secret.docx", 100)
    engine.add_fact("TransferFile", "day2_convert", "lineage", "secret.docx", "secret.pdf", 200)
    engine.add_fact("LeakFile", "day3_upload", "lineage", "secret.pdf", "network", 300)

    leaks = engine.query_leak()

    assert len(leaks) == 1
    assert leaks[0].full_path == "day1_open -> day2_convert -> day3_upload"
    assert leaks[0].leak_timestamp == 300

    reverse = DatalogEngine()
    reverse.add_fact("OpenFile", "day3_open", "lineage", "secret.docx", 300)
    reverse.add_fact("TransferFile", "day2_convert", "lineage", "secret.docx", "secret.pdf", 200)
    reverse.add_fact("LeakFile", "day1_upload", "lineage", "secret.pdf", "network", 100)

    assert reverse.query_leak() == []


def test_datalog_engine_binds_first_case_and_rejects_cross_case_facts() -> None:
    engine = DatalogEngine()
    engine.add_fact("OpenFile", "open_a", "proc", "secret.txt", 1, case_id="case-a")

    assert engine.case_id == "case-a"
    assert engine.facts["OpenFile"][0].case_id == "case-a"
    with pytest.raises(ValueError, match="fact_case_mismatch"):
        engine.add_fact("LeakFile", "leak_b", "proc", "secret.txt", "network", 2, case_id="case-b")


def test_datalog_engine_derives_clipboard_cross_process_transfer() -> None:
    engine = DatalogEngine()
    engine.add_fact("OpenFile", "open_1", "excel.exe", "secret.xlsx", 1)
    engine.add_fact("TransferFile", "copy_1", "excel.exe", "secret.xlsx", "Clipboard", 2)
    engine.add_clipboard_operation("clip_write", "excel.exe", "clip_read", "browser.exe", "Clipboard", 3, 4)
    engine.add_fact("LeakFile", "send_1", "browser.exe", "Clipboard", "chat_upload", 5)

    leaks = engine.query_leak()

    assert len(leaks) == 1
    assert leaks[0].leaking_proc == "browser.exe"


def test_build_datalog_facts_rejects_upload_before_later_canonical_derivation() -> None:
    source = "C:/Users/alice/Documents/secret.docx"
    derived = "C:/Users/alice/Desktop/secret.pdf"
    derive_timestamp = "2026-06-28T10:01:00.000"
    upload_timestamp = "2026-06-28T10:00:00.000"
    lineage = Lineage()
    lineage.add(derived, source)
    facts = build_datalog_facts(
        [
            CorrelatedEvent(
                event_id="upload_before_derivation",
                timestamp=upload_timestamp,
                event_type="file_upload",
                app_name="browser.exe",
                original_file=source,
                current_file=derived,
                operation_type="external_sink_interaction",
                behavior_category="data_exfiltration_candidate",
                confidence=0.95,
            ),
            CorrelatedEvent(
                event_id="derive_after_upload",
                timestamp=derive_timestamp,
                event_type="print_to_pdf",
                app_name="wps.exe",
                original_file=source,
                current_file=derived,
                operation_type="file_or_content_transfer",
                behavior_category="hidden_transformation_candidate",
                confidence=0.95,
            )
        ],
        [
            UploadCandidate(
                candidate_id="early_upload",
                timestamp=upload_timestamp,
                app_name="browser.exe",
                original_file=source,
                current_file=derived,
                sink_type="chat_upload",
                risk_level="completed",
                confidence=0.95,
            )
        ],
        lineage,
        case_id="time-order",
    )

    canonical_transfer = next(
        fact
        for fact in facts
        if fact.relation == "TransferFile"
        and fact.args[1] == "case_lineage"
        and same_file(str(fact.args[2]), source)
        and same_file(str(fact.args[3]), derived)
    )
    assert canonical_transfer.args[4] == parse_timestamp_ms(derive_timestamp)

    engine = DatalogEngine(case_id="time-order")
    for fact in facts:
        engine.add_fact(fact.relation, *fact.args, case_id=fact.case_id)
    assert engine.query_leak() == []


def test_lineage_resolves_two_hops_across_windows_case_and_slash_variants() -> None:
    source = r"C:\Users\Alice\Documents\Secret.DOCX"
    draft = r"C:\Users\Alice\Desktop\Draft.TXT"
    screenshot = r"C:\Users\Alice\Pictures\Screenshot.PNG"
    lineage = Lineage()
    lineage.add(draft, source)
    lineage.add(
        "c:/users/ALICE/pictures/screenshot.png",
        "c:/users/ALICE/desktop/draft.txt",
    )

    chain = lineage.chain(r"c:\USERS\alice\PICTURES\SCREENSHOT.png")

    assert len(chain) == 3
    assert all(same_file(actual, expected) for actual, expected in zip(chain, [screenshot, draft, source], strict=True))


def test_datalog_engine_upload_binding_selects_declared_source_at_merge() -> None:
    source_a = "C:/source/declared-a.docx"
    source_b = "C:/source/other-b.docx"
    merged = "C:/derived/shared-screenshot.png"
    engine = DatalogEngine(case_id="merged-sources")
    # Lexical path ranking prefers B unless UploadBinding constrains this leak to A.
    engine.add_fact("OpenFile", "z_open_a", "case_lineage", source_a, 10)
    engine.add_fact("TransferFile", "z_merge_a", "case_lineage", source_a, merged, 20)
    engine.add_fact("OpenFile", "a_open_b", "case_lineage", source_b, 10)
    engine.add_fact("TransferFile", "a_merge_b", "case_lineage", source_b, merged, 20)
    engine.add_fact("CrossProcessTransfer", "sink_access", "case_lineage", "qq.exe", merged, 30)
    engine.add_fact("UploadBinding", "bind_a", "send_shared", source_a, merged, 30)
    engine.add_fact("LeakFile", "send_shared", "qq.exe", merged, "chat_upload", 31)

    leaks = engine.query_leak()

    assert len(leaks) == 1
    assert leaks[0].source_file == source_a
    assert leaks[0].file_chain == (source_a, merged)
    assert source_b not in leaks[0].file_chain


def test_build_datalog_facts_does_not_emit_leak_for_unbound_upload() -> None:
    upload = UploadCandidate(
        candidate_id="unbound_upload",
        timestamp="2026-06-28T10:00:30.000",
        app_name="teams.exe",
        original_file="",
        current_file="sensitive file",
        sink_type="chat_upload",
        risk_level="completed",
        confidence=0.95,
    )

    facts = build_datalog_facts([], [upload], Lineage(), case_id="unbound-upload")

    assert not any(fact.relation == "LeakFile" for fact in facts)
    assert not any(fact.relation == "UploadBinding" for fact in facts)
    suspicious = [fact for fact in facts if fact.relation == "SuspiciousBehavior"]
    assert len(suspicious) == 1
    assert suspicious[0].args[5] == "unbound_completed"


def test_build_datalog_facts_assigns_unique_operation_ids() -> None:
    source = "C:/Users/alice/Documents/secret.docx"
    draft = "C:/Users/alice/Desktop/draft.txt"
    screenshot = "C:/Users/alice/Pictures/screenshot.png"
    lineage = Lineage()
    lineage.add(draft, source)
    lineage.add(screenshot, draft)
    correlated = [
        CorrelatedEvent(
            event_id="derive_draft",
            timestamp="2026-06-28T10:00:10.000",
            event_type="modified",
            app_name="notepad.exe",
            original_file=source,
            current_file=draft,
            operation_type="file_or_content_transfer",
            behavior_category="hidden_transformation_candidate",
            confidence=0.9,
        ),
        CorrelatedEvent(
            event_id="derive_screenshot",
            timestamp="2026-06-28T10:00:20.000",
            event_type="created",
            app_name="screen_capture.exe",
            original_file=draft,
            current_file=screenshot,
            operation_type="file_or_content_transfer",
            behavior_category="hidden_transformation_candidate",
            confidence=0.9,
        ),
    ]
    uploads = [
        UploadCandidate(
            candidate_id="send_screenshot",
            timestamp="2026-06-28T10:00:30.000",
            app_name="qq.exe",
            original_file=source,
            current_file=screenshot,
            sink_type="chat_upload",
            risk_level="completed",
            confidence=0.95,
        )
    ]

    facts = build_datalog_facts(correlated, uploads, lineage, case_id="unique-ops")
    operation_ids = [str(fact.args[0]) for fact in facts]

    assert operation_ids
    assert len(operation_ids) == len(set(operation_ids))


def test_leak_path_with_structured_flow_steps_is_hashable() -> None:
    leak = LeakPath(
        start_op="open_secret",
        end_op="send_secret",
        leaking_proc="qq.exe",
        leaked_file="C:/derived/secret.png",
        leak_channel="chat_upload",
        leak_timestamp=30,
        full_path="open_secret -> derive_secret -> send_secret",
        case_id="hashable-path",
        source_file="C:/source/secret.docx",
        file_chain=("C:/source/secret.docx", "C:/derived/secret.png"),
        flow_steps=(
            {
                "relation": "TransferFile",
                "source_file": "C:/source/secret.docx",
                "derived_file": "C:/derived/secret.png",
            },
        ),
    )

    assert isinstance(hash(leak), int)
    assert leak in {leak}


def test_qwen_response_parser_keeps_risky_aliases_and_drops_normal_events() -> None:
    response = (
        "prefix ```json\n"
        "{\"events\":["
        "{\"time_range\":\"2026-06-28 10:00:18 - 2026-06-28 10:00:20\","
        "\"app_name\":\"ChatGPT\",\"behavior_category\":\"direct_leak\",\"operation\":\"paste_exfiltration\","
        "\"file_name\":\"payroll-q4\",\"description\":\"employee salary data pasted into an AI chat\"},"
        "{\"time_range\":\"2026-06-28 10:00:05 - 2026-06-28 10:00:06\","
        "\"app_name\":\"Excel\",\"behavior_category\":\"normal\",\"operation_type\":\"read\","
        "\"original_filename\":\"payroll.xlsx\",\"description\":\"only reading the file\"}"
        "]}\n```"
    )

    events = parse_vlm_response(response, keywords=["salary.xlsx"])

    assert len(events) == 1
    assert events[0].app_name == "ChatGPT"
    assert events[0].operation_type == "paste_exfiltration"


def test_vlm_parser_preserves_evidence_frames_and_relative_timestamp() -> None:
    result = parse_vlm_response_detailed(
        json.dumps(
            {
                "events": [
                    {
                        "evidence_frame_ids": ["frame_0_0", "frame_0_1"],
                        "timestamp_ms": 33000,
                        "app_name": "璞嗗寘AI",
                        "behavior_category": "direct_leak",
                        "operation_type": "ai_chat_upload",
                        "original_filename": "员工薪资明细表R4.xlsx",
                        "modified_filename": "灞忓箷鎴浘 2026-06-03 003300.png",
                        "sink_type": "ai_chat",
                        "description": "截图文件上传到 AI 网站",
                        "confidence": 0.91,
                    },
                    {
                        "evidence_frame_ids": ["frame_0_2"],
                        "timestamp_ms": 1000,
                        "app_name": "Excel",
                        "behavior_category": "normal",
                        "operation_type": "read",
                        "original_filename": "员工薪资明细表R4.xlsx",
                        "description": "正常查看表格",
                    },
                ]
            },
            ensure_ascii=False,
        ),
        keywords=["员工薪资明细表R4.xlsx"],
    )

    assert len(result.events) == 1
    assert result.events[0].start_ms == 33000
    assert result.events[0].evidence_frame_ids == ("frame_0_0", "frame_0_1")
    assert result.events[0].sink_type == "ai_chat"
    assert result.dropped_events[0]["reason"] == "not_relevant"


def test_vlm_parser_normalizes_tencent_meeting_document_import_channel() -> None:
    result = parse_vlm_response_detailed(
        json.dumps(
            {
                "events": [
                    {
                        "evidence_frame_ids": ["frame_import"],
                        "timestamp_ms": 33_000,
                        "app_name": "腾讯会议",
                        "behavior_category": "direct_leak",
                        "operation_type": "import_document",
                        "original_filename": "公司合作合同.docx",
                        "sink_type": "screen_share",
                        "action_status": "in_progress",
                        "description": "Import Local Document progress is visible.",
                        "confidence": 0.95,
                    }
                ]
            },
            ensure_ascii=False,
        ),
        keywords=["公司合作合同.docx"],
    )

    assert result.events[0].sink_type == "chat_upload"


def test_vlm_parser_accepts_top_level_array_and_keeps_distinct_resources() -> None:
    result = parse_vlm_response_detailed(
        json.dumps(
            [
                {
                    "evidence_frame_ids": ["frame_a"],
                    "timestamp_ms": 10_000,
                    "app_name": "Outlook Mail",
                    "behavior_category": "direct_leak",
                    "operation_type": "email_send",
                    "original_filename": "secret_a.xlsx",
                    "sink_type": "mail_attachment",
                    "description": "Attachment submitted.",
                },
                {
                    "evidence_frame_ids": ["frame_b"],
                    "timestamp_ms": 10_000,
                    "app_name": "Outlook Mail",
                    "behavior_category": "direct_leak",
                    "operation_type": "email_send",
                    "original_filename": "secret_b.xlsx",
                    "sink_type": "mail_attachment",
                    "description": "Attachment submitted.",
                },
            ]
        )
    )

    assert [event.original_resource for event in result.events] == ["secret_a.xlsx", "secret_b.xlsx"]


@pytest.mark.parametrize(
    ("status", "description", "expected"),
    [
        ("unsupported", "", "failed"),
        ("timed_out", "", "failed"),
        ("unknown", "Upload completed with no errors.", "completed"),
    ],
)
def test_vlm_parser_normalizes_action_status_without_error_substring_false_positive(
    status: str,
    description: str,
    expected: str,
) -> None:
    result = parse_vlm_response_detailed(
        json.dumps(
            {
                "events": [
                    {
                        "evidence_frame_ids": ["frame_0"],
                        "timestamp_ms": 1_000,
                        "app_name": "Gemini",
                        "behavior_category": "direct_leak",
                        "operation_type": "file_upload",
                        "original_filename": "secret.docx",
                        "sink_type": "ai_chat",
                        "action_status": status,
                        "description": description,
                    }
                ]
            }
        )
    )

    assert result.events[0].action_status == expected


def test_vlm_evidence_validation_rejects_hallucinated_frame_ids() -> None:
    frame = VlmRequestFrame(
        KeyFrame("frame_0", 1_000, "unused.jpg", 0.5, "strong:anchor", window_id="window_0"),
        "",
        0.0,
    )
    parsed = parse_vlm_response_detailed(
        json.dumps(
            {
                "events": [
                    {
                        "evidence_frame_ids": ["frame_0", "frame_from_another_window"],
                        "timestamp_ms": 1_000,
                        "app_name": "ChatGPT",
                        "behavior_category": "direct_leak",
                        "operation_type": "file_upload",
                        "original_filename": "secret.docx",
                        "sink_type": "ai_chat",
                    }
                ]
            }
        )
    )

    validated = _validate_vlm_evidence(parsed, [frame])

    assert validated.events[0].evidence_frame_ids == ("frame_0",)
    assert "frame_from_another_window" in validated.parse_errors[0]


def test_vlm_parser_prefers_frame_timestamp_over_absolute_time_range() -> None:
    result = parse_vlm_response_detailed(
        json.dumps(
            {
                "events": [
                    {
                        "evidence_frame_ids": ["frame_0_0"],
                        "timestamp_ms": 16_143,
                        "time_range": "2026-04-20 20:11:16 - 2026-04-20 20:11:26",
                        "app_name": "ChatGPT",
                        "behavior_category": "direct_leak",
                        "operation_type": "file_upload",
                        "sink_type": "ai_chat",
                        "description": "A file is attached to an AI prompt.",
                    }
                ]
            }
        )
    )

    assert result.events[0].start_ms == 16_143
    assert result.events[0].end_ms == 16_143


def test_vlm_content_transform_observation_is_not_external_sink() -> None:
    event = ParsedVisionEvent(
        start_ms=12000,
        end_ms=15000,
        app_name="ChatGPT",
        behavior_category="direct_leak",
        operation_type="AI translation",
        original_resource="product_design.docx",
        modified_resource="unknown",
        description="Sensitive document content is translated in an online service",
        confidence=0.92,
        sink_type="ai_chat",
        evidence_frame_ids=("frame_0_1",),
    )

    observations = vision_events_to_observations([event])

    assert observations[0].operation_type == "file_or_content_transfer"
    assert "AI translation" in observations[0].description


def test_vlm_content_transform_is_recorded_without_leakfile() -> None:
    original = "C:/Users/alice/Documents/product_design.docx"
    event = ParsedVisionEvent(
        start_ms=12000,
        end_ms=15000,
        app_name="ChatGPT",
        behavior_category="direct_leak",
        operation_type="AI translation",
        original_resource="product_design.docx",
        modified_resource="unknown",
        description="Sensitive document content is translated in an online service",
        confidence=0.92,
        sink_type="ai_chat",
        evidence_frame_ids=("frame_0_1",),
    )
    observations = [item.to_dict() for item in vision_events_to_observations([event])]

    bundle = EventCorrelator().run(
        {
            "session_id": "vision",
            "log_events": [],
            "frame_segments": observations,
            "sensitive_files": [original],
        }
    )

    assert bundle["correlated_events"]
    assert bundle["upload_candidates"] == []
    assert any(fact["relation"] == "SuspiciousBehavior" for fact in bundle["datalog_facts"])
    assert not any(fact["relation"] == "LeakFile" for fact in bundle["datalog_facts"])


def test_pipeline_reports_suspicious_detector_state_for_hidden_behavior(tmp_path: Path) -> None:
    original = "C:/Users/alice/Documents/product_design.docx"
    log_file = tmp_path / "logs.json"
    groundtruth = tmp_path / "groundtruth.json"
    observations = tmp_path / "observations.json"
    log_file.write_text("[]", encoding="utf-8")
    groundtruth.write_text(
        json.dumps(
            {"operations": [{"operation": "潜在隐藏行为-内容提取-Base64在线转换", "sensitive_file_path": original}]},
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    observations.write_text(
        json.dumps(
            [
                {
                    "observation_id": "vlm_0",
                    "start_ms": 1000,
                    "end_ms": 1000,
                    "app_name": "ChatGPT",
                    "operation_type": "file_or_content_transfer",
                    "resource": "product_design.docx",
                    "description": "Sensitive document content is transformed with Base64 in an online service",
                    "confidence": 0.91,
                    "source": "vlm",
                }
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    sensitive_config = tmp_path / "sensitive_files.json"
    sensitive_config.write_text(json.dumps({"sensitive_files": [original]}), encoding="utf-8")

    report = run_pipeline(
        log_file=log_file,
        output_dir=tmp_path / "out",
        sensitive_files_config=sensitive_config,
        observations_file=observations,
        groundtruth_file=groundtruth,
    )

    verdict = json.loads(Path(report["detail_files"]["verdict_check"]).read_text(encoding="utf-8"))
    assert report["conclusion"] == "suspicious_behavior_detected"
    assert report["leak_reasoner"]["detector_conclusion"] == "suspicious_behavior_detected"
    assert report["summary"]["suspicious_behaviors"] == 1
    assert any(fact["relation"] == "SuspiciousBehavior" for fact in report["event_correlator"]["datalog_facts"])
    assert verdict["expected_conclusion"] == "suspicious_behavior_detected"
    assert verdict["score_status"] == "scored"
    assert verdict["detector_correct"] is True




def test_visual_observation_can_create_datalog_fact_without_file_path_log() -> None:
    original = "C:/Users/alice/Documents/customer_salary.xlsx"
    records = [
        {
            "timestamp": "2026-06-28T10:00:00.000",
            "event_type": "file_open",
            "file_path": original,
            "process_info": {"process_name": "excel.exe"},
        },
        {
            "timestamp": "2026-06-28T10:00:20.000",
            "event_type": "clipboard_paste",
            "process_info": {"process_name": "chrome.exe"},
            "window_info": {"window_title": "ChatGPT"},
        },
    ]
    events = parse_vlm_response(
        "{\"events\":[{\"time_range\":\"2026-06-28 10:00:20 - 2026-06-28 10:00:21\","
        "\"app_name\":\"ChatGPT\",\"behavior_category\":\"direct_leak\",\"operation_type\":\"paste_exfiltration\","
        "\"original_filename\":\"customer_salary.xlsx\",\"description\":\"salary content pasted to ChatGPT\"}]}",
        keywords=[original],
    )
    observations = [item.to_dict() for item in vision_events_to_observations(events)]

    bundle = EventCorrelator().run(
        {
            "session_id": "vision",
            "log_events": records,
            "frame_segments": observations,
            "sensitive_files": [original],
        }
    )

    assert bundle["upload_candidates"]
    assert any(fact["relation"] == "LeakFile" for fact in bundle["datalog_facts"])


def test_non_vlm_can_be_disabled_while_retaining_log_lineage_for_vlm_evaluation() -> None:
    bundle = EventCorrelator().run(
        {
            "session_id": "unit",
            "log_events": _records(),
            "frame_segments": [],
            "sensitive_files": ["C:/Users/alice/Documents/customer_salary.xlsx"],
            "non_vlm_enabled": False,
        }
    )

    assert bundle["statistics"]["non_vlm_enabled"] is False
    assert bundle["correlated_events"] == []
    assert bundle["upload_candidates"] == []
    assert [item["relation"] for item in bundle["datalog_facts"]] == ["TransferFile"]


def test_vlm_observations_still_work_when_non_vlm_is_disabled() -> None:
    original = "C:/Users/alice/Documents/customer_salary.xlsx"
    records = [
        {
            "timestamp": "2026-06-28T10:00:00.000",
            "event_type": "file_open",
            "file_path": original,
            "process_info": {"process_name": "excel.exe"},
        },
        {
            "timestamp": "2026-06-28T10:00:20.000",
            "event_type": "clipboard_paste",
            "process_info": {"process_name": "chrome.exe"},
            "window_info": {"window_title": "ChatGPT"},
        },
    ]
    events = parse_vlm_response(
        "{\"events\":[{\"time_range\":\"2026-06-28 10:00:20 - 2026-06-28 10:00:21\","
        "\"app_name\":\"ChatGPT\",\"behavior_category\":\"direct_leak\",\"operation_type\":\"paste_exfiltration\","
        "\"original_filename\":\"customer_salary.xlsx\",\"description\":\"salary content pasted to ChatGPT\"}]}",
        keywords=[original],
    )
    observations = [item.to_dict() for item in vision_events_to_observations(events)]

    bundle = EventCorrelator().run(
        {
            "session_id": "vision",
            "log_events": records,
            "frame_segments": observations,
            "sensitive_files": [original],
            "non_vlm_enabled": False,
        }
    )

    assert bundle["upload_candidates"]
    assert any(fact["relation"] == "LeakFile" for fact in bundle["datalog_facts"])




def test_ai_chat_sink_is_classified_without_vlm() -> None:
    assert classify_sink("Cherry Studio 榛樿鍔╂墜 gpt-3.5-turbo 涓婁紶") == "ai_chat"


def test_lineage_uses_extra_source_and_output_paths() -> None:
    original = "C:/Users/alice/Documents/strategy.docx"
    derived = "C:/Users/alice/Desktop/strategy.pdf"
    records = [
        {
            "timestamp": "2026-01-01T00:00:00",
            "event_type": "print_to_pdf",
            "file_path": derived,
            "extra": {"source_path": original, "output_path": derived, "raw_operation": "print_to_pdf"},
            "process_info": {"process_name": "wps.exe"},
        },
        {
            "timestamp": "2026-01-01T00:00:10",
            "event_type": "file_selected",
            "file_path": derived,
            "extra": {"category": "文件上传", "source": "file_dialog_monitor"},
            "process_info": {"process_name": "msedge.exe"},
            "window_info": {"window_title": "ChatGPT upload"},
        },
    ]

    bundle = EventCorrelator().run({"log_events": records, "frame_segments": [], "sensitive_files": [original]})

    assert bundle["file_lineage"]["direct_file_mappings"][derived] == original
    assert any(item["current_file"] == derived for item in bundle["correlated_events"])


def test_pipeline_passes_recursive_log_derived_context_to_log_and_vision(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    original = "C:/Users/alice/Documents/strategy.docx"
    first_derived = "C:/Users/alice/Desktop/strategy.pdf"
    second_derived = "C:/Users/alice/Desktop/strategy.zip"
    records = [
        {
            "timestamp": "2026-01-01T00:00:00",
            "event_type": "print_to_pdf",
            "file_path": first_derived,
            "extra": {"source_path": original, "output_path": first_derived, "raw_operation": "print_to_pdf"},
            "process_info": {"process_name": "wps.exe"},
        },
        {
            "timestamp": "2026-01-01T00:00:10",
            "event_type": "archive_created",
            "file_path": second_derived,
            "extra": {"source_path": first_derived, "output_path": second_derived, "raw_operation": "compress"},
            "process_info": {"process_name": "7z.exe"},
        },
    ]
    log_file = tmp_path / "logs.json"
    log_file.write_text(json.dumps(records), encoding="utf-8")
    sensitive_config = tmp_path / "sensitive_files.json"
    sensitive_config.write_text(json.dumps({"sensitive_files": [original]}), encoding="utf-8")
    observed: dict[str, list[str]] = {}

    class FakeMining:
        windows: list[object] = []
        source = "unit"
        metadata: dict[str, object] = {}

    def fake_analyze(*_: object, **kwargs: object) -> dict:
        observed["vision"] = list(kwargs["sensitive_files"])
        observed["vlm"] = list(kwargs["vlm_sensitive_files"])
        return {"observations": [], "statistics": {}, "warnings": [], "errors": []}

    def fake_mine(**kwargs: object) -> FakeMining:
        observed["log_mining"] = list(kwargs["sensitive_files"])
        return FakeMining()

    monkeypatch.setattr("data_leak_detector.pipeline.mine_analysis_windows", fake_mine)
    monkeypatch.setattr("data_leak_detector.pipeline.analyze_video_behavior", fake_analyze)
    report = run_pipeline(log_file=log_file, sensitive_files_config=sensitive_config, vision_enabled=True)

    assert observed["log_mining"] == [original, first_derived, second_derived]
    assert observed["vision"] == [original, first_derived, second_derived]
    assert observed["vlm"] == [first_derived, second_derived]
    assert report["input"]["sensitive_source_files"] == [original]
    assert report["input"]["derived_sensitive_context"] == [first_derived, second_derived]
    assert report["event_correlator"]["file_lineage"]["direct_file_mappings"][first_derived] == original
    assert report["event_correlator"]["file_lineage"]["direct_file_mappings"][second_derived] == first_derived


def _write_composite_session(
    case_dir: Path,
    session_id: str,
    recording_time: str,
    records: list[dict],
) -> Path:
    session_dir = case_dir / session_id
    (session_dir / "logs").mkdir(parents=True)
    (session_dir / "video").mkdir()
    (session_dir / "logs" / "keyevents.json").write_text(
        json.dumps(records, ensure_ascii=False),
        encoding="utf-8",
    )
    (session_dir / "video" / f"recording_{session_id.removeprefix('session_')}.mp4").write_bytes(b"video")
    (session_dir / "INDEX.md").write_text(
        f"**Session ID**: {session_id.removeprefix('session_')}\n**Recording Time**: {recording_time}\n",
        encoding="utf-8",
    )
    return session_dir


def test_dataset_discovery_groups_direct_sessions_as_one_case(tmp_path: Path) -> None:
    root = tmp_path / "stage4"
    case_dir = root / "e2e-1"
    case_dir.mkdir(parents=True)
    (case_dir / "groundtruth.json").write_text(json.dumps({"operations": []}), encoding="utf-8")
    first = _write_composite_session(case_dir, "session_20260101_090000", "2026-01-01 09:00:00", [])
    second = _write_composite_session(case_dir, "session_20260103_110000", "2026-01-03 11:00:00", [])

    discovered = discover_data_case_directories(root)
    case = discover_data_case(case_dir, case_root=root)
    promoted = discover_data_case(first, case_root=root)

    assert discovered == [case_dir]
    assert first not in discovered and second not in discovered
    assert case.case_id == "e2e-1"
    assert promoted.case_dir == case_dir
    assert promoted.case_id == "e2e-1"
    assert [item.session_id for item in case.sessions] == [
        "session_20260101_090000",
        "session_20260103_110000",
    ]
    assert case.to_input_metadata()["session_count"] == 2


def test_composite_case_merges_sessions_on_absolute_timeline_and_reasons_once(tmp_path: Path) -> None:
    root = tmp_path / "stage4"
    case_dir = root / "e2e-7"
    case_dir.mkdir(parents=True)
    original = "C:/work/company_strategy.docx"
    derived = "C:/work/company_strategy.pdf"
    _write_composite_session(
        case_dir,
        "session_20260101_090000",
        "2026-01-01 09:00:00",
        [
            {
                "timestamp": "2026-01-01T09:00:10",
                "event_type": "print_to_pdf",
                "file_path": derived,
                "source_file": original,
                "destination_path": derived,
                "extra": {"raw_operation": "print_to_pdf", "relative_timestamp": 10.0},
                "process_info": {"process_name": "wps.exe"},
            }
        ],
    )
    _write_composite_session(
        case_dir,
        "session_20260103_110000",
        "2026-01-03 11:00:00",
        [
            {
                "timestamp": "2026-01-03T11:00:10",
                "event_type": "file_upload",
                "file_path": derived,
                "extra": {"raw_operation": "file_upload", "category": "upload", "relative_timestamp": 10.0},
                "process_info": {"process_name": "mail.exe"},
            }
        ],
    )
    (case_dir / "groundtruth.json").write_text(
        json.dumps({"operations": [{"operation": "direct leak", "sensitive_file_path": derived}]}),
        encoding="utf-8",
    )
    sensitive_config = tmp_path / "sensitive_files.json"
    sensitive_config.write_text(json.dumps({"sensitive_files": [original]}), encoding="utf-8")

    report = run_data_case(
        case_dir,
        case_root=root,
        sensitive_files_config=sensitive_config,
        vision_enabled=False,
        neo4j_log_miner=False,
        output_dir=tmp_path / "out",
    )

    assert report["input"]["case_id"] == "e2e-7"
    assert report["input"]["session_count"] == 2
    assert report["summary"]["sessions"] == 2
    assert report["summary"]["logs"] == 2
    assert report["event_correlator"]["session_id"] == "e2e-7"
    assert report["event_correlator"]["case_id"] == "e2e-7"
    assert {item["_dld_session_id"] for item in report["event_correlator"]["raw_log_events"]} == {
        "session_20260101_090000",
        "session_20260103_110000",
    }
    assert {
        item["event_id"].split(":", 1)[0]
        for item in report["event_correlator"]["raw_log_events"]
    } == {
        "session_20260101_090000",
        "session_20260103_110000",
    }
    assert {item["case_id"] for item in report["event_correlator"]["datalog_facts"]} == {"e2e-7"}
    assert report["event_correlator"]["file_lineage"]["direct_file_mappings"][derived] == original
    observations = report["frame_analyzer"]["observations"]
    assert len({item["observation_id"] for item in observations}) == len(observations)
    assert max(item["start_ms"] for item in observations) - min(item["start_ms"] for item in observations) > 2 * 86_400_000
    leak = report["leak_reasoner"]["leak_paths"][0]
    assert report["leak_reasoner"]["case_id"] == "e2e-7"
    assert leak["case_id"] == "e2e-7"
    assert leak["start_op"] == "corr_0:open"
    assert ":transfer:lineage" in leak["full_path"]
    assert ":access" in leak["full_path"]
    assert leak["leak_timestamp"] == 1_767_438_010_000
    event_details = json.loads(Path(report["detail_files"]["event_correlator_details"]).read_text(encoding="utf-8"))
    assert len(event_details["raw_log_events_sources"]) == 2
    assert {item["case_id"] for item in event_details["datalog_facts"]} == {"e2e-7"}
    persisted_leaks = json.loads(Path(report["detail_files"]["leak_paths"]).read_text(encoding="utf-8"))
    assert {item["case_id"] for item in persisted_leaks} == {"e2e-7"}
    readable_report = json.loads(Path(report["report_file"]).read_text(encoding="utf-8"))
    assert readable_report["event_correlator"]["case_id"] == "e2e-7"
    assert readable_report["leak_reasoner"]["case_id"] == "e2e-7"


def test_release_full_flow_keeps_facts_and_leak_paths_scoped_to_composite_cases(tmp_path: Path) -> None:
    root = tmp_path / "stage4"
    cases = {
        "case-a": {
            "original": "C:/work/case_a_secret.docx",
            "derived": "C:/work/case_a_secret.pdf",
            "sessions": ("session_20260101_090000", "session_20260103_110000"),
            "times": ("2026-01-01 09:00:00", "2026-01-03 11:00:00"),
        },
        "case-b": {
            "original": "C:/work/case_b_secret.xlsx",
            "derived": "C:/work/case_b_secret.csv",
            "sessions": ("session_20260201_090000", "session_20260203_110000"),
            "times": ("2026-02-01 09:00:00", "2026-02-03 11:00:00"),
        },
    }
    for case_id, expected in cases.items():
        case_dir = root / case_id
        case_dir.mkdir(parents=True)
        first_session, second_session = expected["sessions"]
        first_time, second_time = expected["times"]
        original = expected["original"]
        derived = expected["derived"]
        _write_composite_session(
            case_dir,
            first_session,
            first_time,
            [
                {
                    "event_id": f"{case_id}:derive",
                    "timestamp": first_time.replace(" ", "T"),
                    "event_type": "print_to_pdf",
                    "file_path": derived,
                    "source_file": original,
                    "destination_path": derived,
                    "extra": {"raw_operation": "print_to_pdf", "relative_timestamp": 0.0},
                    "process_info": {"process_name": "office.exe"},
                }
            ],
        )
        _write_composite_session(
            case_dir,
            second_session,
            second_time,
            [
                {
                    "event_id": f"{case_id}:upload",
                    "timestamp": second_time.replace(" ", "T"),
                    "event_type": "file_upload",
                    "file_path": derived,
                    "extra": {"raw_operation": "file_upload", "category": "upload", "relative_timestamp": 0.0},
                    "process_info": {"process_name": "browser.exe"},
                }
            ],
        )
        (case_dir / "groundtruth.json").write_text(
            json.dumps({"operations": [{"operation": "direct leak", "sensitive_file_path": derived}]}),
            encoding="utf-8",
        )

    sensitive_config = tmp_path / "sensitive_files.json"
    sensitive_config.write_text(
        json.dumps({"sensitive_files": [expected["original"] for expected in cases.values()]}),
        encoding="utf-8",
    )

    result = run_e2e_module._run_case_root(
        str(root),
        common_args={
            "sensitive_files_config": sensitive_config,
            "vision_enabled": False,
            "neo4j_log_miner": False,
        },
        output_dir=str(tmp_path / "release"),
        workers=2,
        release=True,
    )

    release_report = result["release_report"]
    assert release_report["batch"]["case_count"] == 2
    assert release_report["batch"]["completed_cases"] == 2
    reports_by_case = {item["case_id"]: item for item in release_report["cases"]}
    assert set(reports_by_case) == set(cases)

    for case_id, expected in cases.items():
        case_report = reports_by_case[case_id]
        facts = case_report["event_correlator"]["datalog_facts"]
        leak_paths = case_report["leak_reasoner"]["leak_paths"]
        first_time, second_time = (parse_timestamp_ms(item) for item in expected["times"])
        other_case = next(item for item in cases.values() if item is not expected)

        assert case_report["event_correlator"]["case_id"] == case_id
        assert case_report["event_correlator"]["session_id"] == case_id
        assert case_report["leak_reasoner"]["case_id"] == case_id
        assert case_report["summary"]["sessions"] == 2
        assert case_report["summary"]["datalog_facts"] == len(facts)
        assert case_report["summary"]["leak_paths"] == len(leak_paths) == 1
        assert {item["relation"] for item in facts} >= {
            "OpenFile",
            "TransferFile",
            "CrossProcessTransfer",
            "LeakFile",
        }
        assert {item["case_id"] for item in facts} == {case_id}

        serialized_facts = json.dumps(facts, ensure_ascii=False)
        assert expected["original"] in serialized_facts
        assert expected["derived"] in serialized_facts
        assert other_case["original"] not in serialized_facts
        assert other_case["derived"] not in serialized_facts

        source_fact = next(
            item
            for item in facts
            if item["relation"] == "OpenFile" and expected["original"] in item["args"]
        )
        leak_fact = next(
            item
            for item in facts
            if item["relation"] == "LeakFile" and expected["derived"] in item["args"]
        )
        assert source_fact["args"][-1] == first_time
        assert leak_fact["args"][-1] == second_time

        leak = leak_paths[0]
        assert leak["case_id"] == case_id
        assert leak["leaked_file"] == expected["derived"]
        assert source_fact["args"][0] in leak["full_path"]
        assert leak_fact["args"][0] in leak["full_path"]
        assert leak["leak_timestamp"] == second_time
        assert other_case["derived"] not in json.dumps(leak, ensure_ascii=False)


def test_dataset_case_discovery_uses_real_data_layout(tmp_path: Path) -> None:
    case_dir = tmp_path / "case"
    logs_dir = case_dir / "logs"
    video_dir = case_dir / "video"
    logs_dir.mkdir(parents=True)
    video_dir.mkdir()
    (logs_dir / "keyevents.json").write_text(json.dumps(_records()), encoding="utf-8")
    (video_dir / "recording.mp4").write_bytes(b"not a real video")
    (case_dir / "groundtruth.json").write_text(
        json.dumps({"operations": [{"sensitive_file_path": "C:/from-groundtruth.xlsx"}]}),
        encoding="utf-8",
    )

    sensitive_config = tmp_path / "sensitive_files.json"
    sensitive_config.write_text(json.dumps({"sensitive_files": ["C:/from-config.xlsx"]}), encoding="utf-8")
    case = discover_data_case(case_dir, sensitive_files_config=sensitive_config)

    assert case.log_file.name == "keyevents.json"
    assert case.video_file and case.video_file.name == "recording.mp4"
    assert case.sensitive_files == ("C:/from-config.xlsx",)


def test_dataset_case_discovery_accepts_misspelled_groundtruth_name(tmp_path: Path) -> None:
    case_dir = tmp_path / "case"
    logs_dir = case_dir / "logs"
    video_dir = case_dir / "video"
    logs_dir.mkdir(parents=True)
    video_dir.mkdir()
    (logs_dir / "keyevents.json").write_text(json.dumps(_records()), encoding="utf-8")
    (video_dir / "recording.mp4").write_bytes(b"not a real video")
    (case_dir / "groundtrutn.json").write_text(
        json.dumps({"operations": [{"sensitive_file_path": "C:/Users/alice/Documents/customer_salary.xlsx"}]}),
        encoding="utf-8",
    )

    sensitive_config = tmp_path / "sensitive_files.json"
    sensitive_config.write_text(json.dumps({"sensitive_files": ["C:/Users/alice/Documents/customer_salary.xlsx"]}), encoding="utf-8")
    case = discover_data_case(case_dir, sensitive_files_config=sensitive_config)

    assert case.groundtruth_file == case_dir / "groundtrutn.json"
    assert case.groundtruth_status == "available"
    assert case.sensitive_files == ("C:/Users/alice/Documents/customer_salary.xlsx",)


def test_dataset_case_discovery_prefers_indexed_video(tmp_path: Path) -> None:
    case_dir = tmp_path / "case"
    logs_dir = case_dir / "logs"
    video_dir = case_dir / "video"
    logs_dir.mkdir(parents=True)
    video_dir.mkdir()
    (logs_dir / "logs.json").write_text(json.dumps(_records()), encoding="utf-8")
    (video_dir / "recording_20240101_000000.mp4").write_bytes(b"old")
    (video_dir / "recording_20240102_000000.mp4").write_bytes(b"new")
    (case_dir / "INDEX.md").write_text(
        "# Recording Session Index\n\n**Session ID**: 20240102_000000\n\n- `video/recording_20240102_000000.mp4`\n",
        encoding="utf-8",
    )

    case = discover_data_case(case_dir)

    assert case.video_file and case.video_file.name == "recording_20240102_000000.mp4"


def test_dataset_case_discovery_uses_timestamped_video_when_index_is_missing(tmp_path: Path) -> None:
    case_dir = tmp_path / "case"
    logs_dir = case_dir / "logs"
    video_dir = case_dir / "video"
    logs_dir.mkdir(parents=True)
    video_dir.mkdir()
    (logs_dir / "keyevents.json").write_text(json.dumps(_records()), encoding="utf-8")
    (video_dir / "recording_20260602_234649.mp4").write_bytes(b"not a real video")

    case = discover_data_case(case_dir)

    assert case.recording_start_ms == parse_timestamp_ms("2026-06-02 23:46:49")


def test_dataset_case_discovery_uses_relative_case_id_and_marks_missing_child_groundtruth(tmp_path: Path) -> None:
    root = tmp_path / "stage1"
    parent = root / "1-email-QQemail-1"
    child = parent / "1-email-Outlook-2"
    (child / "logs").mkdir(parents=True)
    (child / "video").mkdir()
    (child / "logs" / "keyevents.json").write_text(json.dumps(_records()), encoding="utf-8")
    (child / "video" / "recording.mp4").write_bytes(b"not a real video")
    (parent / "groundtruth.json").write_text(
        json.dumps({"operations": [{"operation": "leak", "sensitive_file_path": "C:/secret.docx"}]}),
        encoding="utf-8",
    )

    case = discover_data_case(child, case_root=root)

    assert case.case_id == "1-email-QQemail-1/1-email-Outlook-2"
    assert case.groundtruth_file is None
    assert case.groundtruth_status == "missing_current_directory_with_ancestor_groundtruth"
    assert case.nearest_ancestor_groundtruth_file == parent / "groundtruth.json"


def test_dataset_case_discovery_promotes_single_session_to_parent_case(tmp_path: Path) -> None:
    root = tmp_path / "stage1"
    parent = root / "3-Messaging-TIM-5"
    child = parent / "session_20260420_222538"
    (child / "logs").mkdir(parents=True)
    (child / "video").mkdir()
    (child / "logs" / "keyevents.json").write_text(json.dumps(_records()), encoding="utf-8")
    (child / "video" / "recording.mp4").write_bytes(b"not a real video")
    (child / "INDEX.md").write_text("**Recording Time**: 2026-04-20 22:25:38\n", encoding="utf-8")
    (parent / "groundtruth.json").write_text(
        json.dumps(
            {
                "recording_start_time": "2026-04-20 22:00:00",
                "operations": [{"operation": "leak", "sensitive_file_path": "C:/secret.docx"}],
            }
        ),
        encoding="utf-8",
    )

    sensitive_config = tmp_path / "sensitive_files.json"
    sensitive_config.write_text(json.dumps({"sensitive_files": ["C:/secret.docx"]}), encoding="utf-8")
    discovered = discover_data_case_directories(root)
    case = discover_data_case(child, case_root=root, sensitive_files_config=sensitive_config)

    assert discovered == [parent]
    assert child not in discovered
    assert case.case_dir == parent
    assert case.case_id == "3-Messaging-TIM-5"
    assert case.groundtruth_file == parent / "groundtruth.json"
    assert case.groundtruth_status == "available"
    assert case.nearest_ancestor_groundtruth_file is None
    assert [session.session_dir for session in case.sessions] == [child]
    assert case.to_input_metadata()["session_count"] == 1
    assert case.sensitive_files == ("C:/secret.docx",)
    assert case.recording_start_ms == 1776723938000


def test_report_id_includes_case_name_for_artifact_folders() -> None:
    report_id = _build_report_id(Path("logs.json"), 528, "1-email-fastmail-1")

    assert report_id == "1-email-fastmail-1_logs_528"


def test_sensitive_files_config_is_the_initial_source_set(tmp_path: Path) -> None:
    config = tmp_path / "sensitive_files.json"
    config.write_text(
        json.dumps({"sensitive_files": ["C://Users//alice//Documents//strategy.docx", "c:/users/alice/documents/strategy.docx", 1]}),
        encoding="utf-8",
    )

    assert load_sensitive_files_config(config) == ("C:/Users/alice/Documents/strategy.docx",)


def test_policy_terms_are_loaded_from_external_config(tmp_path: Path) -> None:
    policy_file = tmp_path / "policy.json"
    policy_file.write_text(
        json.dumps(
            {
                "sink_tokens": ["澶栧彂瀹℃壒"],
                "sink_classification": [{"type": "approval_portal", "tokens": ["澶栧彂瀹℃壒"]}],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    policy = load_policy_config(policy_file)

    assert contains_any("姝ｅ湪鎻愪氦澶栧彂瀹℃壒", policy.sink_tokens)
    assert policy.sink_classification == (("approval_portal", ("澶栧彂瀹℃壒",)),)














def test_direct_keyframe_vlm_selection_uses_visual_frames() -> None:
    frames = [
        KeyFrame("medium", 1_000, "medium.jpg", 0.5, "medium:visual_change", window_id="window_0"),
        KeyFrame("strong", 2_000, "strong.jpg", 0.2, "strong:anchor", window_id="window_0"),
        KeyFrame("weak", 3_000, "weak.jpg", 1.0, "weak:visual_change", window_id="window_1"),
    ]

    selected = choose_keyframes_for_vlm(frames, max_frames=2, max_frames_per_window=1)

    assert [item.frame.frame_id for item in selected] == ["strong", "weak"]
    assert all(item.visual_note == "" for item in selected)


def test_direct_keyframe_vlm_selection_spreads_dense_window_anchors() -> None:
    frames = [
        KeyFrame(f"frame_{timestamp}", timestamp, "frame.jpg", 0.9, "strong:anchor", window_id="window_0")
        for timestamp in (30_771, 32_192, 32_727, 34_280, 34_715, 36_795)
    ]

    selected = choose_keyframes_for_vlm(frames, max_frames=3, max_frames_per_window=3)

    selected_ids = [item.frame.frame_id for item in selected]
    assert selected_ids[0] == "frame_30771"
    assert selected_ids[-1] == "frame_36795"
    assert "frame_34280" in selected_ids or "frame_34715" in selected_ids


def test_direct_keyframe_global_budget_spreads_dense_window_anchors() -> None:
    frames = [
        KeyFrame(f"frame_{timestamp}", timestamp, "frame.jpg", 0.9, "strong:anchor", window_id="window_0")
        for timestamp in (30_771, 32_192, 32_727, 34_280, 34_715, 36_795)
    ]

    selected = choose_keyframes_for_vlm(frames, max_frames=3)

    assert [item.frame.timestamp_ms for item in selected] == [30_771, 34_280, 36_795]


def test_explicit_vlm_budget_spreads_frames_without_semantic_scoring() -> None:
    frames = [
        KeyFrame("anchor_start", 0, "start.jpg", 0.2, "medium:anchor", window_id="window_0"),
        KeyFrame("activity_gap", 49_211, "gap.jpg", 0.9, "medium:activity_gap", window_id="window_0"),
        KeyFrame("anchor_end", 54_002, "end.jpg", 0.2, "medium:anchor", window_id="window_0"),
    ]

    selected = choose_keyframes_for_vlm(frames, max_frames=2)

    assert [item.frame.frame_id for item in selected] == ["anchor_start", "anchor_end"]


def test_vlm_selection_does_not_filter_when_under_explicit_budget() -> None:
    frames = [
        KeyFrame("action", 10_000, "action.jpg", 0.9, "strong:activity_gap", window_id="window_0"),
        KeyFrame("dialog", 11_000, "dialog.jpg", 0.2, "strong:anchor", window_id="window_0"),
        KeyFrame("attached", 20_000, "attached.jpg", 0.2, "strong:anchor", window_id="window_0"),
    ]

    selected = choose_keyframes_for_vlm(frames, max_frames=8)

    assert [item.frame.frame_id for item in selected] == ["action", "dialog", "attached"]


def test_direct_keyframe_vlm_selection_compacts_activity_gap_window() -> None:
    frames = [
        KeyFrame("early_gap", 3_000, "early.jpg", 0.9, "medium:activity_gap", window_id="window_0"),
        KeyFrame("context", 35_000, "context.jpg", 0.3, "medium:anchor", window_id="window_0"),
        KeyFrame("action", 49_000, "action.jpg", 0.1, "medium:activity_gap", window_id="window_0"),
        KeyFrame("post_action", 54_000, "post.jpg", 0.8, "medium:anchor", window_id="window_0"),
    ]

    selected = choose_keyframes_for_vlm(frames, max_frames=12)

    assert [item.frame.frame_id for item in selected] == ["early_gap", "context", "action", "post_action"]


def test_direct_keyframe_negative_budget_sends_all_keyframes() -> None:
    frames = [
        KeyFrame(f"frame_{timestamp}", timestamp, "frame.jpg", 0.9, "strong:anchor", window_id="window_0")
        for timestamp in (30_771, 32_192, 32_727, 34_280, 34_715, 36_795)
    ]

    selected = choose_keyframes_for_vlm(frames, max_frames=-1)

    assert [item.frame.timestamp_ms for item in selected] == [30_771, 32_192, 32_727, 34_280, 34_715, 36_795]


def test_vlm_grid_builder_keeps_direct_keyframe_mapping(tmp_path: Path) -> None:
    Image = pytest.importorskip("PIL.Image")
    keyframes = []
    for index, color in enumerate(((255, 0, 0), (0, 255, 0), (0, 0, 255))):
        image_path = tmp_path / f"frame_{index}.jpg"
        Image.new("RGB", (120, 80), color).save(image_path)
        keyframes.append(KeyFrame(f"frame_{index}", index * 1000, str(image_path), 0.9, "strong:anchor", window_id="window_0"))
    selected = choose_keyframes_for_vlm(keyframes, max_frames=-1)

    grids = build_vlm_frame_grids(selected, grid_size=2, output_dir=tmp_path / "grid")

    assert len(grids) == 1
    assert Path(grids[0].frame.image_path).exists()
    assert grids[0].frame.frame_id == "vlm_grid_0"
    assert [item["frame_id"] for item in grids[0].source_frames] == ["frame_0", "frame_1", "frame_2"]
    assert [item["cell_id"] for item in grids[0].source_frames] == ["A1", "A2", "B1"]


def test_vlm_grid_builder_supports_vertical_layout(tmp_path: Path) -> None:
    Image = pytest.importorskip("PIL.Image")
    keyframes = []
    for index, color in enumerate(((255, 0, 0), (0, 255, 0), (0, 0, 255))):
        image_path = tmp_path / f"frame_{index}.jpg"
        Image.new("RGB", (120, 80), color).save(image_path)
        keyframes.append(KeyFrame(f"frame_{index}", index * 1000, str(image_path), 0.9, "strong:anchor", window_id="window_0"))

    grids = build_vlm_frame_grids(
        choose_keyframes_for_vlm(keyframes, max_frames=-1),
        grid_size=1,
        grid_layout="2x1",
        output_dir=tmp_path / "vertical_grid",
    )

    assert len(grids) == 2
    assert [item["cell_id"] for item in grids[0].source_frames] == ["A1", "B1"]
    assert [item["cell_id"] for item in grids[1].source_frames] == ["A1"]


def test_vlm_grid_builder_never_mixes_analysis_windows(tmp_path: Path) -> None:
    Image = pytest.importorskip("PIL.Image")
    keyframes = []
    for index, window_id in enumerate(("window_a", "window_a", "window_b", "window_b")):
        image_path = tmp_path / f"window_frame_{index}.jpg"
        Image.new("RGB", (120, 80), (index * 40, 80, 120)).save(image_path)
        keyframes.append(
            KeyFrame(
                f"frame_{index}",
                index * 1_000,
                str(image_path),
                0.9,
                "strong:anchor",
                window_id=window_id,
            )
        )

    grids = build_vlm_frame_grids(
        choose_keyframes_for_vlm(keyframes, max_frames=-1),
        grid_size=2,
        output_dir=tmp_path / "window_grids",
    )

    assert [item.frame.window_id for item in grids] == ["window_a", "window_b"]
    assert [[source["frame_id"] for source in item.source_frames] for item in grids] == [
        ["frame_0", "frame_1"],
        ["frame_2", "frame_3"],
    ]
    assert len(vlm_frame_batches(grids, workers=4)) == 2






def test_vlm_input_images_are_resized_without_touching_raw_frame(tmp_path: Path) -> None:
    Image = pytest.importorskip("PIL.Image")
    raw = tmp_path / "raw.jpg"
    Image.new("RGB", (200, 100), (120, 80, 40)).save(raw)
    frame = KeyFrame("frame_0", 1_000, str(raw), 0.0, "medium:coverage")
    request_frame = VlmRequestFrame(frame, "", 0.0)

    prepared = prepare_vlm_frame_images([request_frame], max_image_side=100, output_dir=tmp_path / "vlm_input")

    assert prepared[0].frame.image_path != str(raw)
    assert Image.open(prepared[0].frame.image_path).size == (100, 50)
    assert Image.open(raw).size == (200, 100)


def test_vlm_workers_keep_one_evidence_window_in_one_batch() -> None:
    frames = list(range(5))

    batches = vlm_frame_batches(frames, workers=3)

    assert batches == [frames]
    assert vlm_frame_batches(frames, workers=1) == [frames]


def test_vlm_dispatch_uses_single_active_key_without_changing_parallelism() -> None:
    config = VisionConfig(
        vlm_api_key="primary",
        vlm_api_keys=("secondary", "primary"),
        vlm_workers=3,
        vlm_fast_dispatch=True,
    )

    clients = build_vlm_clients(config)

    assert [client.config.vlm_api_key for client in clients] == ["primary"]
    assert effective_vlm_parallelism(config) == 3
    assert effective_vlm_parallelism(VisionConfig(vlm_workers=3)) == 3


def test_vlm_key_pool_can_supply_the_only_configured_key() -> None:
    clients = build_vlm_clients(VisionConfig(vlm_api_keys=("secondary",), vlm_fast_dispatch=True))

    assert [client.config.vlm_api_key for client in clients] == ["secondary"]


def test_vlm_client_pool_uses_one_configured_plan_key() -> None:
    config = VisionConfig(
        vlm_api_key="sk-sp-coding-plan",
        vlm_api_keys=("sk-token-plan",),
        vlm_workers=10,
        vlm_fast_dispatch=True,
    )

    clients = build_vlm_clients(config)

    assert [client.config.vlm_api_key for client in clients] == ["sk-sp-coding-plan"]
    assert effective_vlm_parallelism(config) == 10


def test_vlm_endpoint_limiter_is_shared_across_concurrent_cases() -> None:
    config = VisionConfig(
        vlm_coding_api_key="coding-key",
        vlm_token_api_key="token-key",
        vlm_fast_dispatch=True,
        vlm_workers=10,
    )
    first_case = build_vlm_clients(config)
    second_case = build_vlm_clients(config)

    first_locks = _shared_vlm_endpoint_locks(first_case, workers_per_key=config.vlm_workers)
    second_locks = _shared_vlm_endpoint_locks(second_case, workers_per_key=config.vlm_workers)

    assert len(first_case) == 1
    assert len(second_case) == 1
    assert first_locks[id(first_case[0])] is second_locks[id(second_case[0])]


@dataclass
class _RetryClient:
    key: str
    fail: bool = False

    @property
    def config(self) -> SimpleNamespace:
        return SimpleNamespace(vlm_api_key=self.key)

    def analyze(self, *_: object, **__: object) -> VlmResponse:
        if self.fail:
            raise RuntimeError("invalid_api_key")
        return VlmResponse(text='{"events":[]}', provider="unit", model="unit", usage={"total_tokens": 1})


@dataclass
class _TransientRetryClient:
    remaining_failures: int

    @property
    def config(self) -> SimpleNamespace:
        return SimpleNamespace(vlm_api_key="transient", vlm_base_url="https://unit.test")

    def analyze(self, *_: object, **__: object) -> VlmResponse:
        if self.remaining_failures:
            self.remaining_failures -= 1
            raise TimeoutError("temporary timeout")
        return VlmResponse(text='{"events":[]}', provider="unit", model="unit", usage={"total_tokens": 1})


def test_vlm_batch_retries_another_key_when_the_assigned_key_fails() -> None:
    results = run_vlm_batches(
        [_RetryClient("invalid", fail=True), _RetryClient("valid")],  # type: ignore[arg-type]
        [[object()], [object()]],
        sensitive_files=[],
        active_apps=[],
        workers_per_key=1,
    )

    assert results["errors"] == []
    assert len(results["batches"]) == 2
    assert results["retry_warnings"] == ["vlm_key_retry[0]: RuntimeError: invalid_api_key"]


def test_vlm_batch_retries_transient_failure_with_the_same_key() -> None:
    results = run_vlm_batches(
        [_TransientRetryClient(remaining_failures=1)],  # type: ignore[arg-type]
        [[object()]],
        sensitive_files=[],
        active_apps=[],
        workers_per_key=1,
        retry_attempts=2,
        retry_backoff_seconds=0,
    )

    assert results["errors"] == []
    assert len(results["batches"]) == 1
    assert results["retry_warnings"] == ["vlm_transient_retry[0:1]: TimeoutError: temporary timeout"]


def test_vlm_batches_reuse_one_process_queue_across_cases() -> None:
    first_case = [_RetryClient("queue-coding"), _RetryClient("queue-token")]
    second_case = [_RetryClient("queue-coding"), _RetryClient("queue-token")]

    first_dispatcher = _shared_vlm_dispatcher(first_case, workers_per_key=1)  # type: ignore[arg-type]
    second_dispatcher = _shared_vlm_dispatcher(second_case, workers_per_key=1)  # type: ignore[arg-type]
    result = run_vlm_batches(
        second_case,  # type: ignore[arg-type]
        [[object()], [object()]],
        sensitive_files=[],
        active_apps=[],
        workers_per_key=1,
    )

    assert first_dispatcher is second_dispatcher
    assert result["errors"] == []
    assert result["dispatch"]["mode"] == "shared_process_queue"
    assert result["dispatch"]["snapshot"]["parallelism"] == 2
    assert result["dispatch"]["snapshot"]["submitted_batches"] >= 2


def test_vlm_worker_artifacts_combine_real_metrics_and_usage() -> None:
    summaries = [
        {"request_metrics": {"prompt_chars": 10, "image_count": 2, "image_pixels": 100, "image_megapixels": 0.1}},
        {"request_metrics": {"prompt_chars": 20, "image_count": 1, "image_pixels": 50, "image_megapixels": 0.05}},
    ]

    metrics = combine_vlm_request_metrics(summaries)
    usage = _combine_vlm_usage(
        [
            {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120},
            {"prompt_tokens": 80, "completion_tokens": 10, "total_tokens": 90},
        ]
    )
    request_payload = vlm_request_artifact_payload(
        summaries,
        workers=4,
        workers_per_key=2,
        fast_dispatch=True,
        api_key_count=2,
    )

    assert metrics["prompt_chars"] == 30
    assert metrics["image_count"] == 3
    assert metrics["image_pixels"] == 150
    assert metrics["image_megapixels"] == 0.15
    assert usage["prompt_tokens"] == 180
    assert usage["completion_tokens"] == 30
    assert usage["total_tokens"] == 210
    assert usage["batches"][0]["total_tokens"] == 120
    assert request_payload["workers"] == 4
    assert request_payload["batch_count"] == 2
    assert request_payload["dispatch"] == {
        "fast_dispatch": True,
        "api_key_count": 2,
        "workers_per_key": 2,
        "parallelism": 4,
    }




def test_frontend_app_recognition_generalizes_to_unseen_apps() -> None:
    assert identify_frontend_app(window_title="Compose - Proton Workspace Mail").category == "mail"
    assert identify_frontend_app(window_title="Upload files - Mega Cloud Drive").category == "cloud_drive"
    assert identify_frontend_app(window_title="Acme Assistant prompt").category == "ai_chat"
    assert identify_frontend_app(window_title="Bluetooth file transfer").category == "removable_media"


def test_frontend_app_recognition_prefers_product_over_browser_wrapper() -> None:
    assert identify_frontend_app(app_name="msedge.exe", window_title="Inbox - Outlook").category == "mail"
    assert identify_frontend_app(app_name="chrome.exe", window_title="ChatGPT").category == "ai_chat"
    assert identify_frontend_app(app_name="GitHub", window_title="GitHub - Microsoft Edge").category == "code_hosting"


def test_groundtruth_verdict_uses_configurable_dataset_criteria(tmp_path: Path) -> None:
    groundtruth = tmp_path / "groundtruth.json"
    groundtruth.write_text(
        json.dumps(
            {
                "operations": [
                    {"operation": "正常操作-打开文件", "sensitive_file_path": "C:/secret.docx"},
                    {"operation": "直接外发-上传到网盘，完成外传", "sensitive_file_path": "C:/secret.docx"},
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    verdict = evaluate_groundtruth(groundtruth)

    assert verdict.conclusion == "data_leak_risk_detected"
    assert len(verdict.leak_operations) == 1
    assert len(verdict.non_leak_operations) == 1


def test_groundtruth_verdict_counts_explicit_external_and_screen_share_phrases(tmp_path: Path) -> None:
    groundtruth = tmp_path / "groundtruth.json"
    groundtruth.write_text(
        json.dumps(
            {
                "operations": [
                    {"operation": "直接外发-邮件外发", "sensitive_file_path": "C:/secret.pdf"},
                    {"operation": "潜在隐藏行为-复制内容外发", "sensitive_file_path": "C:/secret.docx"},
                    {"operation": "潜在隐藏行为-Lark会议屏幕共享展示敏感文件", "sensitive_file_path": "C:/secret.xlsx"},
                    {"operation": "潜在隐藏行为-文件重命名", "sensitive_file_path": "C:/secret_rename.docx"},
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    verdict = evaluate_groundtruth(groundtruth)

    assert verdict.conclusion == "data_leak_risk_detected"
    assert [item.operation for item in verdict.leak_operations] == [
        "直接外发-邮件外发",
        "潜在隐藏行为-复制内容外发",
        "潜在隐藏行为-Lark会议屏幕共享展示敏感文件",
    ]
    assert [item.operation for item in verdict.unknown_risk_operations] == ["潜在隐藏行为-文件重命名"]


def test_groundtruth_verdict_records_suspicious_behavior_as_third_state(tmp_path: Path) -> None:
    groundtruth = tmp_path / "groundtruth.json"
    groundtruth.write_text(
        json.dumps(
            {
                "operations": [
                    {"operation": "正常操作-打开查看", "sensitive_file_path": "C:/secret.txt"},
                    {"operation": "潜在隐藏行为-内容提取-Base64在线转换", "sensitive_file_path": "C:/secret.txt"},
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    verdict = evaluate_groundtruth(groundtruth)

    assert verdict.conclusion == "suspicious_behavior_detected"
    assert len(verdict.leak_operations) == 0
    assert len(verdict.non_leak_operations) == 1
    assert len(verdict.unknown_risk_operations) == 1


def test_groundtruth_verdict_does_not_treat_monitor_name_as_leak(tmp_path: Path) -> None:
    groundtruth = tmp_path / "groundtruth.json"
    groundtruth.write_text(
        json.dumps(
            {
                "operations": [
                    {"operation": "正常操作-应用切换-返回数据泄露监控系统", "sensitive_file_path": "N/A"},
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    verdict = evaluate_groundtruth(groundtruth)

    assert verdict.conclusion == "no_confirmed_data_leak"
    assert len(verdict.leak_operations) == 0
    assert len(verdict.non_leak_operations) == 1


def test_groundtruth_english_leak_token_uses_word_boundaries_and_reads_plural_paths(tmp_path: Path) -> None:
    groundtruth = tmp_path / "groundtruth.json"
    paths = ["C:/derived/2.ksheet", "C:/derived/3.ksheet"]
    groundtruth.write_text(
        json.dumps(
            {
                "operations": [
                    {
                        "operation": "潜在隐藏行为-手动录入-D:/DataLeakDetector/output.txt",
                        "sensitive_file_paths": paths,
                    }
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    verdict = evaluate_groundtruth(groundtruth)

    assert verdict.conclusion == "suspicious_behavior_detected"
    assert verdict.unknown_risk_operations[0].sensitive_file == "; ".join(paths)


def test_pipeline_conclusion_keeps_detector_result_when_groundtruth_available(tmp_path: Path) -> None:
    log_file = tmp_path / "logs.json"
    groundtruth = tmp_path / "groundtruth.json"
    log_file.write_text(
        json.dumps(
            [
                {
                    "timestamp": "2026-01-01T00:00:00",
                    "event_type": "file_open",
                    "file_path": "C:/Users/alice/Documents/ordinary.txt",
                }
            ]
        ),
        encoding="utf-8",
    )
    groundtruth.write_text(
        json.dumps({"operations": [{"operation": "直接外发-发送敏感文件", "sensitive_file_path": "C:/secret.docx"}]}, ensure_ascii=False),
        encoding="utf-8",
    )

    report = run_pipeline(log_file=log_file, groundtruth_file=groundtruth)

    assert report["conclusion"] == "no_confirmed_data_leak"
    assert report["verdict"]["source"] == "reasoner"
    assert report["verdict"]["groundtruth_conclusion"] == "data_leak_risk_detected"
    assert report["leak_reasoner"]["detector_conclusion"] == "no_confirmed_data_leak"
    assert report["detection_core"]["method"] == "non_uniform_keyframes_vlm_datalog"
    assert report["detection_core"]["evaluation"]["groundtruth_is_evaluation_only"] is True


def test_pipeline_does_not_expose_groundtruth_to_frame_analysis(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    log_file = tmp_path / "logs.json"
    groundtruth = tmp_path / "groundtruth.json"
    groundtruth_text = json.dumps({"operations": [{"operation": "normal"}]}, ensure_ascii=False)
    log_file.write_text(
        json.dumps([{"timestamp": "2026-01-01T00:00:00", "event_type": "heartbeat"}]),
        encoding="utf-8",
    )
    groundtruth.write_text(groundtruth_text, encoding="utf-8")

    def fake_analyze_video_behavior(*_: object, **kwargs: object) -> dict:
        copied = Path(str(kwargs["artifact_dir"])) / "groundtruth.json"
        assert not copied.exists()
        return {
            "observations": [],
            "statistics": {
                "mode": "unit",
                "observations": 0,
                "vision": {"enabled": True, "mode": "hybrid", "log_mining": {"source": "unit"}},
            },
            "warnings": [],
            "errors": [],
        }

    monkeypatch.setattr("data_leak_detector.pipeline.analyze_video_behavior", fake_analyze_video_behavior)

    report = run_pipeline(
        log_file=log_file,
        output_dir=tmp_path / "out",
        groundtruth_file=groundtruth,
        vision_enabled=True,
    )

    assert Path(report["detail_files"]["groundtruth"]).read_text(encoding="utf-8") == groundtruth_text


def test_pipeline_writes_verdict_check_into_detail_dir(tmp_path: Path) -> None:
    log_file = tmp_path / "logs.json"
    groundtruth = tmp_path / "groundtruth.json"
    log_file.write_text(
        json.dumps([{"timestamp": "2026-01-01T00:00:00", "event_type": "heartbeat"}]),
        encoding="utf-8",
    )
    groundtruth.write_text(
        json.dumps({"operations": [{"operation": "leak", "sensitive_file_path": "C:/secret.docx"}]}, ensure_ascii=False),
        encoding="utf-8",
    )

    report = run_pipeline(
        log_file=log_file,
        output_dir=tmp_path / "out",
        groundtruth_file=groundtruth,
    )

    verdict_file = Path(report["detail_files"]["verdict_check"])
    verdict = json.loads(verdict_file.read_text(encoding="utf-8"))
    assert verdict_file.parent.name == report["report_id"]
    assert verdict["groundtruth_available"] is True
    assert verdict["expected_conclusion"] == "data_leak_risk_detected"
    assert verdict["detector_conclusion"] == "no_confirmed_data_leak"
    assert verdict["detector_correct"] is False
    assert verdict["final_correct"] is False


def test_pipeline_verdict_check_scores_suspicious_groundtruth_as_detector_state(tmp_path: Path) -> None:
    log_file = tmp_path / "logs.json"
    groundtruth = tmp_path / "groundtruth.json"
    log_file.write_text(
        json.dumps([{"timestamp": "2026-01-01T00:00:00", "event_type": "heartbeat"}]),
        encoding="utf-8",
    )
    groundtruth.write_text(
        json.dumps(
            {"operations": [{"operation": "潜在隐藏行为-内容提取-Base64在线转换", "sensitive_file_path": "C:/secret.txt"}]},
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    report = run_pipeline(
        log_file=log_file,
        output_dir=tmp_path / "out",
        groundtruth_file=groundtruth,
    )

    verdict = json.loads(Path(report["detail_files"]["verdict_check"]).read_text(encoding="utf-8"))
    assert verdict["expected_conclusion"] == "suspicious_behavior_detected"
    assert verdict["groundtruth_unknown_risk_operations"] == 1
    assert verdict["score_status"] == "scored"
    assert verdict["detector_correct"] is False
    assert verdict["final_correct"] is False


def test_json_loader_keeps_escaped_windows_paths(tmp_path: Path) -> None:
    log_file = tmp_path / "logs.json"
    log_file.write_text(
        '[{"timestamp":"2026-01-01T00:00:00","process_info":{"process_path":"C:\\\\Program Files\\\\App\\\\app.exe"}}]',
        encoding="utf-8",
    )

    records = load_json_records(log_file)

    assert records[0]["process_info"]["process_path"].endswith("App\\app.exe")


def test_pipeline_writes_report_for_inline_leak(tmp_path: Path) -> None:
    log_file = tmp_path / "sample.json"
    log_file.write_text(json.dumps(_records(), ensure_ascii=False), encoding="utf-8")
    sensitive_config = tmp_path / "sensitive_files.json"
    sensitive_config.write_text(
        json.dumps({"sensitive_files": ["C:/Users/alice/Documents/customer_salary.xlsx"]}),
        encoding="utf-8",
    )

    report = run_pipeline(
        log_file=log_file,
        output_dir=tmp_path / "out",
        sensitive_files_config=sensitive_config,
    )

    assert report["summary"]["leak_paths"] == 1
    assert Path(report["report_file"]).exists()
    saved_report = json.loads(Path(report["report_file"]).read_text(encoding="utf-8"))
    event_details = Path(saved_report["detail_files"]["event_correlator_details"])
    assert saved_report["event_correlator"]["counts"]["raw_log_events"] == len(_records())
    assert "raw_log_events" not in saved_report["event_correlator"]
    assert event_details.exists()
    event_details_payload = json.loads(event_details.read_text(encoding="utf-8"))
    assert event_details_payload["raw_log_events_count"] == len(_records())
    assert "raw_log_events" not in event_details_payload
    assert report["conclusion"] == "data_leak_risk_detected"
    assert report["graph"]["status"] == "not_supported"
    assert report["log_miner"]["source"] == "in_memory"
    assert report["frame_analyzer"]["statistics"]["vision"]["log_mining"]["source"] == "in_memory"


def test_screen_share_binds_future_filename_and_directory_identity() -> None:
    records = [
        {
            "timestamp": "2026-03-07T12:49:10.330",
            "event_type": "app_switch",
            "app_name": "Edge",
            "window_info": {"window_title": "teams.live.com 正在共享你的屏幕。"},
        },
        {
            "timestamp": "2026-03-07T12:49:14.435",
            "event_type": "app_switch",
            "app_name": "explorer",
            "window_info": {"window_title": "documents_1 - 文件资源管理器"},
        },
        {
            "timestamp": "2026-03-07T12:49:20.574",
            "event_type": "app_switch",
            "app_name": "wps",
            "window_info": {"window_title": "产品设计方案.docx - WPS Office"},
        },
    ]
    bundle = EventCorrelator().run(
        {
            "log_events": records,
            "frame_segments": [],
            "sensitive_files": [
                "D:/gdata/documents_1/产品设计方案.docx",
                "C:/other/documents_2/产品设计方案.docx",
            ],
            "non_vlm_enabled": True,
        }
    )

    assert bundle["upload_candidates"][0]["app_name"] == "Microsoft Teams"
    assert bundle["upload_candidates"][0]["sink_type"] == "screen_share"
    assert bundle["upload_candidates"][0]["original_file"] == "D:/gdata/documents_1/产品设计方案.docx"


def test_virtual_machine_clipboard_switch_is_external() -> None:
    original = "C:/Users/alice/Desktop/customer_contacts.pdf"
    records = [
        {
            "timestamp": "2026-01-01T00:00:00",
            "event_type": "app_switch",
            "app_name": "WPS",
            "window_info": {"window_title": "customer_contacts.pdf - WPS Office"},
        },
        {"timestamp": "2026-01-01T00:00:05", "event_type": "clipboard_text", "app_name": "WPS"},
        {
            "timestamp": "2026-01-01T00:00:12",
            "event_type": "app_switch",
            "app_name": "vmware",
            "window_info": {"window_title": "Windows 10 x64 - VMware Workstation"},
        },
    ]
    bundle = EventCorrelator().run(
        {"log_events": records, "frame_segments": [], "sensitive_files": [original], "non_vlm_enabled": True}
    )

    assert bundle["upload_candidates"][0]["sink_type"] == "virtual_machine"
    assert bundle["upload_candidates"][0]["original_file"] == original


def test_generic_browser_after_clipboard_is_not_an_external_sink() -> None:
    original = "C:/Users/alice/Desktop/customer_contacts.pdf"
    records = [
        {
            "timestamp": "2026-01-01T00:00:00",
            "event_type": "app_switch",
            "app_name": "WPS",
            "window_info": {"window_title": "customer_contacts.pdf - WPS Office"},
        },
        {"timestamp": "2026-01-01T00:00:05", "event_type": "clipboard_text", "app_name": "WPS"},
        {
            "timestamp": "2026-01-01T00:00:12",
            "event_type": "app_switch",
            "app_name": "Chrome",
            "window_info": {"window_title": "New Tab - Google Chrome"},
        },
    ]
    bundle = EventCorrelator().run(
        {"log_events": records, "frame_segments": [], "sensitive_files": [original], "non_vlm_enabled": True}
    )

    assert bundle["upload_candidates"] == []


def test_blank_feishu_document_after_clipboard_is_preparation_only() -> None:
    original = "C:/Users/alice/Desktop/customer_contacts.pdf"
    records = [
        {
            "timestamp": "2026-01-01T00:00:00",
            "event_type": "app_switch",
            "app_name": "WPS",
            "window_info": {"window_title": "customer_contacts.pdf - WPS Office"},
        },
        {"timestamp": "2026-01-01T00:00:05", "event_type": "clipboard_text", "app_name": "WPS"},
        {
            "timestamp": "2026-01-01T00:00:12",
            "event_type": "app_switch",
            "app_name": "Chrome",
            "window_info": {"window_title": "未命名文档 - 飞书云文档 - Google Chrome"},
        },
    ]
    bundle = EventCorrelator().run(
        {"log_events": records, "frame_segments": [], "sensitive_files": [original], "non_vlm_enabled": True}
    )

    assert bundle["upload_candidates"] == []

