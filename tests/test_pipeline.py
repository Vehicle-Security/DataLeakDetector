"""Contract tests for the pipeline, dataset discovery, VLM parsing, and Neo4j adapter."""

from __future__ import annotations

import json
from pathlib import Path

from data_leak_detector import run_pipeline
from data_leak_detector.datasets import discover_data_case
from data_leak_detector.event_correlator import EventCorrelator
from data_leak_detector.frame_analyzer import analyze_video_behavior
from data_leak_detector.frame_analyzer.parser import parse_vlm_response, vision_events_to_observations
from data_leak_detector.graph.store import Neo4jGraphStore
from data_leak_detector.io import normalize_logs
from data_leak_detector.io import load_json_records
from data_leak_detector.leak_reasoner import DatalogEngine


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


def test_dataset_case_discovery_uses_real_data_layout(tmp_path: Path) -> None:
    case_dir = tmp_path / "case"
    logs_dir = case_dir / "logs"
    video_dir = case_dir / "video"
    logs_dir.mkdir(parents=True)
    video_dir.mkdir()
    (logs_dir / "keyevents.json").write_text(json.dumps(_records()), encoding="utf-8")
    (video_dir / "recording.mp4").write_bytes(b"not a real video")
    (case_dir / "groundtruth.json").write_text(
        json.dumps({"operations": [{"sensitive_file_path": "C:/Users/alice/Documents/customer_salary.xlsx"}]}),
        encoding="utf-8",
    )

    case = discover_data_case(case_dir)

    assert case.log_file.name == "keyevents.json"
    assert case.video_file and case.video_file.name == "recording.mp4"
    assert case.sensitive_files == ("C:/Users/alice/Documents/customer_salary.xlsx",)


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

    report = run_pipeline(
        log_file=log_file,
        output_dir=tmp_path / "out",
        sensitive_files=["C:/Users/alice/Documents/customer_salary.xlsx"],
        neo4j_enabled=False,
    )

    assert report["summary"]["leak_paths"] == 1
    assert Path(report["report_file"]).exists()
    assert report["conclusion"] == "data_leak_risk_detected"
    assert report["graph"]["status"] == "skipped"


def test_neo4j_writer_generates_graph_queries(tmp_path: Path) -> None:
    log_file = tmp_path / "sample.json"
    log_file.write_text(json.dumps(_records(), ensure_ascii=False), encoding="utf-8")
    report = run_pipeline(
        log_file=log_file,
        sensitive_files=["C:/Users/alice/Documents/customer_salary.xlsx"],
        neo4j_enabled=False,
    )
    tx = _FakeTransaction()

    Neo4jGraphStore._write_report_tx(tx, report, clear_session=False)

    cypher = "\n".join(query for query, _ in tx.calls)
    assert "DLDReport" in cypher
    assert "DLDLogEvent" in cypher
    assert "DLDFrameObservation" in cypher
    assert "DLDCorrelatedEvent" in cypher
    assert "DLDUploadCandidate" in cypher
    assert "DLDDatalogFact" in cypher
    assert "DLDLeakPath" in cypher
    assert "DERIVED_FROM" in cypher


class _FakeTransaction:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []

    def run(self, query: str, parameters: dict | None = None, **kwargs) -> None:
        merged = dict(parameters or {})
        merged.update(kwargs)
        self.calls.append((query, merged))
