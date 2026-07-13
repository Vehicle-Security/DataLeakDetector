"""Contract tests for the pipeline, dataset discovery, VLM parsing, and Neo4j adapter."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest

from data_leak_detector import run_pipeline
from data_leak_detector.datasets import discover_data_case
from data_leak_detector.event_correlator import EventCorrelator
from data_leak_detector.frame_analyzer import analyze_video_behavior
from data_leak_detector.frame_analyzer.artifacts import export_vision_artifacts
from data_leak_detector.frame_analyzer.vlm_dispatch import (
    _combine_vlm_usage,
    _shared_vlm_dispatcher,
    _shared_vlm_endpoint_locks,
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
    _hamming,
    _probe_timestamps,
    _should_keep_frame,
    _timestamp_groups,
    merge_analysis_windows,
)
from data_leak_detector.frame_analyzer.parser import ParsedVisionEvent, parse_vlm_response, parse_vlm_response_detailed, vision_events_to_observations
from data_leak_detector.frame_analyzer.vlm_client import VlmRequestFrame, VlmResponse, _prompt, build_vlm_frame_grids, choose_keyframes_for_vlm, prepare_vlm_frame_images
from data_leak_detector.log_mining import build_analysis_windows, mine_analysis_windows
from data_leak_detector.neo4j.importer import fingerprint_records, records_to_graph_events
from data_leak_detector.groundtruth import evaluate_groundtruth
from data_leak_detector.io import normalize_logs, normalize_path, same_file
from run_e2e import _release_direct_defaults, _reusable_precompute_baseline
from data_leak_detector.io import load_json_records
from data_leak_detector.leak_reasoner import DatalogEngine
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


def test_release_keeps_deterministic_log_evidence_enabled() -> None:
    args = SimpleNamespace(
        max_vlm_frames=None,
        release_debug_artifacts=True,
        neo4j_log_miner=False,
    )

    release_args = _release_direct_defaults({}, args)

    assert release_args["vision_enabled"] is True
    assert release_args["non_vlm_enabled"] is True


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


def test_analysis_windows_use_video_relative_timestamps_from_real_logs() -> None:
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

    assert windows[0].start_ms == 12_500
    assert windows[0].end_ms == 162_500
    assert windows[0].priority == "weak"
    assert windows[0].step_ms == 2_000


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
    assert windows[0].end_ms == 75_000
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

    assert any(window.priority == "activity" and 30_000 in window.anchor_ms for window in windows)
    assert any(window.priority == "strong" and 60_000 in window.anchor_ms for window in windows)


def test_sensitive_activity_window_runs_until_explicit_close_and_filters_system_apps() -> None:
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
    activity = next(window for window in windows if window.priority == "activity")

    assert activity.start_ms == 0
    assert activity.end_ms == 1_380_000
    assert "ChatGPT" in activity.active_apps
    assert "System" not in activity.active_apps
    assert activity.active_ranges == ((660_000, 1_380_000),)


def test_sensitive_activity_window_excludes_blank_shell_desktop_interval() -> None:
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

    activity = next(window for window in build_analysis_windows(logs, [sensitive], VisionConfig()) if window.priority == "activity")

    assert activity.active_ranges == ((2_000, 4_999),)


def test_sensitive_activity_window_excludes_internal_monitor_window() -> None:
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

    activity = next(window for window in build_analysis_windows(logs, [sensitive], VisionConfig()) if window.priority == "activity")

    assert activity.active_ranges == ((2_000, 7_999),)


def test_sensitive_activity_window_excludes_untitled_wallpaper_overlay() -> None:
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

    activity = next(window for window in build_analysis_windows(logs, [sensitive], VisionConfig()) if window.priority == "activity")

    assert activity.active_ranges == ((2_000, 7_999),)


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

    assert [window.priority for window in windows] == ["activity"]


def test_unclosed_sensitive_activity_window_reaches_last_recorded_video_time() -> None:
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

    activity = next(window for window in build_analysis_windows(logs, [sensitive], VisionConfig()) if window.priority == "activity")

    assert activity.end_ms == 1_860_000


def test_derivation_action_without_known_sensitive_file_gets_vlm_window() -> None:
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

    assert windows[0].priority == "medium"
    assert windows[0].anchor_ms == (30_000,)


def test_clipboard_copy_is_strong_only_while_sensitive_file_is_open() -> None:
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

    assert any(window.priority == "strong" and 30_000 in window.anchor_ms for window in windows)
    assert not any(window.priority == "strong" and 90_000 in window.anchor_ms for window in windows)


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


def test_sink_file_selection_dialog_foreground_logs_become_strong_anchors() -> None:
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

    assert windows[0].priority == "strong"
    assert windows[0].start_ms == 29_280
    assert 34_280 in windows[0].anchor_ms


def test_dense_sink_file_selection_dialog_anchors_are_thinned() -> None:
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

    assert windows[0].priority == "strong"
    assert windows[0].anchor_ms == (13_570, 17_000, 24_000)
    assert windows[0].max_keyframes == VisionConfig().max_keyframes_per_strong_window


def test_cloud_drive_file_selection_dialog_becomes_strong_anchor() -> None:
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

    assert windows[0].priority == "strong"
    assert windows[0].anchor_ms == (17_525, 20_525)


def test_browser_cloud_drive_file_selection_uses_nearby_context() -> None:
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

    assert windows[0].priority == "strong"
    assert windows[0].anchor_ms == (18_925, 21_925)


def test_workspace_file_selection_keeps_upload_followup_anchors() -> None:
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

    assert windows[0].priority == "strong"
    assert windows[0].anchor_ms == (10_644, 13_644, 18_644, 26_644, 30_644)
    assert windows[0].end_ms == 30_644


def test_sensitive_clipboard_to_ai_sink_keeps_post_switch_evidence_anchors() -> None:
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
    strong = next(window for window in windows if window.priority == "strong")

    assert strong.action_anchor_ms == (44_000, 53_000)


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

    assert [window.priority for window in windows] == ["strong", "activity"]
    assert windows[0].start_ms == 55_000
    assert windows[0].end_ms == 75_000
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
    assert windows[0].anchor_ms == (0, 3_000, 8_000)


def test_sensitive_activity_window_replaces_dense_generic_medium_windows() -> None:
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
    activity = [window for window in windows if window.priority == "activity"]

    assert len(activity) == 1
    assert activity[0].start_ms == 0
    assert activity[0].end_ms == 1_200_000
    assert not [window for window in windows if window.priority == "medium"]


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
    assert build_analysis_windows(logs, [], VisionConfig(include_weak_windows=True))[0].priority == "weak"






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


def test_risk_window_does_not_keep_unrelated_app_switch_context() -> None:
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
    assert windows[0].anchor_ms == (60_000,)


def test_long_windows_have_temporal_coverage_targets() -> None:
    window = AnalysisWindow(0, 156_000, "medium", priority="medium", step_ms=1_000, max_keyframes=18)

    coverage = _coverage_timestamps(window)

    assert len(coverage) == 12
    assert coverage[0] == 0
    assert coverage[-1] == 156_000
    assert any(25_000 <= timestamp <= 40_000 for timestamp in coverage)


def test_sensitive_activity_windows_do_not_force_uniform_coverage_frames() -> None:
    window = AnalysisWindow(0, 156_000, "sensitive_activity:secret.docx", priority="activity", max_keyframes=18)

    assert _coverage_timestamps(window) == ()


def test_anchored_strong_window_prioritizes_risk_anchors_over_coverage() -> None:
    window = AnalysisWindow(0, 60_000, "upload", priority="strong", step_ms=250, anchor_ms=(10_000, 40_000))

    assert _coverage_timestamps(window) == ()
    assert _probe_timestamps(window, VisionConfig()) == [10_000, 40_000, 9_750, 10_250, 39_750, 40_250]


def test_sensitive_activity_window_probes_relative_positions_between_file_activity_anchors() -> None:
    window = AnalysisWindow(
        0,
        3_600_000,
        "sensitive_activity:secret.docx",
        priority="activity",
        step_ms=1_000,
        anchor_ms=(120_000, 3_200_000),
    )

    probes = _probe_timestamps(window, VisionConfig())

    assert {2_430_000, 2_815_000, 90_000, 3_500_000}.issubset(probes)
    assert {119_000, 120_000, 121_000, 3_199_000, 3_200_000, 3_201_000}.issubset(probes)


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

    assert probes == [10_000, 50_000, 9_750, 10_250, 49_750, 50_250]


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
    assert any(30_000 <= timestamp <= 34_000 for timestamp in coverage)




def test_frame_hash_distance_can_detect_near_duplicates() -> None:
    assert _hamming((0b101010, 64), (0b101011, 64)) == 1


def test_keyframe_filter_has_no_time_based_force_keep() -> None:
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

    assert keep_duplicate_after_long_gap is False
    assert keep_different_frame is True
    assert keep_duplicate_log_anchor is True
    assert keep_changed_log_anchor is True


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


def test_global_dedupe_keeps_latest_near_duplicate_activity_gap_frame() -> None:
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

    assert [item.frame_id for item in kept] == ["late"]
    assert duplicates[0].frame.frame_id == "early"


def test_normalize_path_repairs_gbk_text_decoded_as_latin1() -> None:
    garbled_name = "公司合同".encode("gb18030").decode("latin1")

    assert normalize_path(f"C:/Users/alice/Desktop/{garbled_name}.docx") == "C:/Users/alice/Desktop/公司合同.docx"
    assert same_file(f"C:/Users/alice/Desktop/{garbled_name}.docx", "C:/Users/alice/Desktop/公司合同.docx")


def test_file_dialog_flow_keeps_stable_result_after_selection() -> None:
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

    assert [frame.frame_id for frame in focused] == ["dialog_final", "result_final", "saved"]


def test_file_dialog_flow_keeps_available_opening_and_result_evidence() -> None:
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

    assert [frame.frame_id for frame in focused] == ["dialog_open", "result_final"]


def test_strong_action_window_keeps_compact_activity_followup_evidence() -> None:
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

    assert [frame.frame_id for frame in focused] == ["uploading", "save_pdf", "confirmation"]


def test_strong_action_window_keeps_nearest_sensitive_source_context() -> None:
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

    assert [frame.frame_id for frame in focused] == ["source_near", "copy", "result"]


def test_sensitive_file_reading_does_not_emit_activity_only_frames() -> None:
    frames = [
        KeyFrame("opened", 1_000, "opened.jpg", 0.1, "activity:anchor", window_id="window_0"),
        KeyFrame("closed", 9_000, "closed.jpg", 0.2, "activity:anchor", window_id="window_0"),
    ]
    windows = [
        AnalysisWindow(0, 10_000, "sensitive_activity:secret.docx", priority="activity", anchor_ms=(1_000, 9_000)),
    ]

    assert _focus_actionable_keyframes(frames, [], windows) == []


def test_generic_strong_app_switch_does_not_emit_a_desktop_frame() -> None:
    frame = KeyFrame("desktop", 1_000, "desktop.jpg", 1.0, "strong:anchor", window_id="window_0")
    windows = [AnalysisWindow(0, 2_000, "strong:app_switch:window_monitor", priority="strong")]

    assert _focus_actionable_keyframes([frame], [], windows) == []


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


def test_unresolved_tim_window_keeps_embedded_send_phases() -> None:
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

    assert [frame.frame_id for frame in focused] == ["action", "sent"]


def test_pipeline_records_add_only_tim_upload_keyevents(tmp_path: Path) -> None:
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
        ("t0", "app_switch"),
        ("t1", "file_upload"),
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


def test_strong_derivation_keeps_only_its_compact_action_phase() -> None:
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

    assert [frame.frame_id for frame in focused] == ["export", "result"]


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


def test_clipboard_text_keeps_pre_action_context_not_a_post_action_desktop() -> None:
    frames = [
        KeyFrame("selected_text", 9_000, "word.jpg", 0.2, "strong:anchor", window_id="window_0"),
        KeyFrame("desktop", 10_000, "desktop.jpg", 0.3, "strong:anchor", window_id="window_0"),
    ]
    windows = [
        AnalysisWindow(0, 12_000, "strong:clipboard_text", priority="strong", action_anchor_ms=(10_000,))
    ]

    focused = _focus_actionable_keyframes(frames, [], windows)

    assert [frame.frame_id for frame in focused] == ["selected_text"]


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

    assert [window.priority for window in windows] == ["strong", "activity"]
    assert windows[0].anchor_ms == (10_000,)


def test_clipboard_capture_uses_recent_sensitive_signal_when_open_event_is_missing() -> None:
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
        ]
    )

    windows = build_analysis_windows(logs, [sensitive], VisionConfig())
    action = next(window for window in windows if window.priority == "strong")

    assert action.action_anchor_ms == (10_000,)
    assert "clipboard" in action.reason


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
    bundle = EventCorrelator().run(
        {
            "session_id": "unit",
            "log_events": _records(),
            "frame_segments": [],
            "sensitive_files": ["C:/Users/alice/Documents/customer_salary.xlsx"],
        }
    )

    assert bundle["analysis_status"] == "success"
    assert bundle["upload_candidates"]
    assert bundle["file_lineage"]["direct_file_mappings"]["C:/Users/alice/Desktop/customer_salary_part1.xlsx"].endswith("customer_salary.xlsx")
    assert any(fact["relation"] == "LeakFile" for fact in bundle["datalog_facts"])


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
    activity = next(window for window in windows if window.priority == "activity")
    assert strong.anchor_ms == (46_000, 49_000, 54_000)
    assert activity.start_ms == activity.end_ms == 46_000
    assert activity.max_keyframes == VisionConfig().max_keyframes_per_window


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


def test_datalog_engine_finds_derived_file_leak() -> None:
    engine = DatalogEngine()
    engine.add_fact("OpenFile", "open_1", "excel.exe", "secret.xlsx", 1)
    engine.add_fact("TransferFile", "copy_1", "excel.exe", "secret.xlsx", "secret_copy.xlsx", 2)
    engine.add_fact("LeakFile", "upload_1", "excel.exe", "secret_copy.xlsx", "network", 3)

    leaks = engine.query_leak()

    assert len(leaks) == 1
    assert leaks[0].full_path == "open_1 -> copy_1 -> upload_1"


def test_datalog_engine_derives_clipboard_cross_process_transfer() -> None:
    engine = DatalogEngine()
    engine.add_fact("OpenFile", "open_1", "excel.exe", "secret.xlsx", 1)
    engine.add_fact("TransferFile", "copy_1", "excel.exe", "secret.xlsx", "Clipboard", 2)
    engine.add_clipboard_operation("clip_write", "excel.exe", "clip_read", "browser.exe", "Clipboard", 3, 4)
    engine.add_fact("LeakFile", "send_1", "browser.exe", "Clipboard", "chat_upload", 5)

    leaks = engine.query_leak()

    assert len(leaks) == 1
    assert leaks[0].leaking_proc == "browser.exe"


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


def test_dataset_case_discovery_can_inherit_ancestor_groundtruth(tmp_path: Path) -> None:
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
    case = discover_data_case(
        child,
        case_root=root,
        inherit_ancestor_groundtruth=True,
        sensitive_files_config=sensitive_config,
    )

    assert case.case_id == "3-Messaging-TIM-5/session_20260420_222538"
    assert case.groundtruth_file == parent / "groundtruth.json"
    assert case.groundtruth_status == "inherited_from_ancestor"
    assert case.nearest_ancestor_groundtruth_file == parent / "groundtruth.json"
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


def test_direct_keyframe_vlm_selection_retains_activity_gap_evidence() -> None:
    frames = [
        KeyFrame("anchor_start", 0, "start.jpg", 0.2, "medium:anchor", window_id="window_0"),
        KeyFrame("activity_gap", 49_211, "gap.jpg", 0.9, "medium:activity_gap", window_id="window_0"),
        KeyFrame("anchor_end", 54_002, "end.jpg", 0.2, "medium:anchor", window_id="window_0"),
    ]

    selected = choose_keyframes_for_vlm(frames, max_frames=2)

    assert [item.frame.frame_id for item in selected] == ["anchor_start", "activity_gap"]


def test_direct_keyframe_vlm_selection_keeps_terminal_result_after_activity_gap() -> None:
    frames = [
        KeyFrame("action", 10_000, "action.jpg", 0.9, "strong:activity_gap", window_id="window_0"),
        KeyFrame("dialog", 11_000, "dialog.jpg", 0.2, "strong:anchor", window_id="window_0"),
        KeyFrame("attached", 20_000, "attached.jpg", 0.2, "strong:anchor", window_id="window_0"),
    ]

    selected = choose_keyframes_for_vlm(frames, max_frames=8)

    assert [item.frame.frame_id for item in selected] == ["action", "attached"]


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


def test_vlm_workers_split_frames_into_contiguous_batches() -> None:
    frames = list(range(5))

    batches = vlm_frame_batches(frames, workers=3)

    assert batches == [[0, 1], [2, 3], [4]]
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

