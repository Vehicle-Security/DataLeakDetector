"""Contract tests for the canonical pipeline and Neo4j graph adapter.

These tests cover the behavior that must stay stable while the internals remain
small and modular: frame observation extraction, event correlation, lineage
reasoning, report writing, and graph Cypher generation.
"""

from __future__ import annotations

import json
from pathlib import Path

from data_leak_detector import run_pipeline
from data_leak_detector.event_correlator import EventCorrelator
from data_leak_detector.frame_analyzer import analyze_video_behavior
from data_leak_detector.io import normalize_logs
from data_leak_detector.leak_reasoner import DatalogEngine
from data_leak_detector.graph.store import Neo4jGraphStore


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
    records = _records()

    bundle = EventCorrelator().run(
        {
            "session_id": "unit",
            "log_events": records,
            "frame_segments": [],
            "sensitive_files": ["C:/Users/alice/Documents/customer_salary.xlsx"],
        }
    )

    assert bundle["analysis_status"] == "success"
    assert bundle["upload_candidates"]
    assert bundle["file_lineage"]["direct_file_mappings"]["C:/Users/alice/Desktop/customer_salary_part1.xlsx"].endswith("customer_salary.xlsx")
    assert any(fact["relation"] == "LeakFile" for fact in bundle["datalog_facts"])


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


def test_pipeline_writes_report_for_sample_leak(tmp_path: Path) -> None:
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
    assert report["conclusion"] == "发现数据泄露风险"
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
