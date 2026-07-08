"""Contract tests for the pipeline, dataset discovery, VLM parsing, and Neo4j adapter."""

from __future__ import annotations

import json
from pathlib import Path

from data_leak_detector import run_pipeline
from data_leak_detector.datasets import discover_data_case
from data_leak_detector.event_correlator import EventCorrelator
from data_leak_detector.frame_analyzer import analyze_video_behavior
from data_leak_detector.frame_analyzer.analyzer import _dedupe_ocr_results, _export_vision_artifacts, _select_ocr_frames_for_ocr
from data_leak_detector.frame_analyzer.apps import identify_frontend_app
from data_leak_detector.frame_analyzer.config import VisionConfig
from data_leak_detector.frame_analyzer.frames import KeyFrame, _hamming, _should_keep_frame, build_analysis_windows
from data_leak_detector.frame_analyzer.ocr import OcrResult, RapidOcrProvider
from data_leak_detector.frame_analyzer.parser import parse_vlm_response, vision_events_to_observations
from data_leak_detector.frame_analyzer.vlm import choose_vlm_frames
from data_leak_detector.graph.store import Neo4jGraphStore
from data_leak_detector.groundtruth import evaluate_groundtruth
from data_leak_detector.io import normalize_logs
from data_leak_detector.io import load_json_records
from data_leak_detector.leak_reasoner import DatalogEngine
from data_leak_detector.policy import contains_any, load_policy_config
from data_leak_detector.sensitivity import SensitiveSourceConfig, extract_sensitive_sources


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

    windows = build_analysis_windows(logs, [], VisionConfig(frame_window_before_ms=30_000, frame_window_after_ms=120_000))

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
    assert windows[0].max_keyframes == 24


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

    windows = build_analysis_windows(logs, ["C:/Users/alice/Desktop/secret.docx"], VisionConfig())

    assert [window.priority for window in windows] == ["strong", "medium"]
    assert windows[0].start_ms == 55_000
    assert windows[0].end_ms == 75_000
    assert windows[0].max_keyframes == 24
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


def test_ocr_reads_all_selected_keyframes() -> None:
    frames = [
        KeyFrame(f"w0_{index}", index * 100, "frame.jpg", 0.9, "strong:visual_change", window_id="window_0")
        for index in range(3)
    ] + [
        KeyFrame(f"w1_{index}", 10_000 + index * 100, "frame.jpg", 0.9, "strong:visual_change", window_id="window_1")
        for index in range(3)
    ]

    selected = _select_ocr_frames_for_ocr(frames)

    assert [frame.frame_id for frame in selected] == ["w0_0", "w0_1", "w0_2", "w1_0", "w1_1", "w1_2"]


def test_ocr_keeps_medium_and_strong_keyframes() -> None:
    frames = [
        KeyFrame("medium", 100, "frame.jpg", 0.9, "medium:visual_change", window_id="window_0"),
        KeyFrame("strong", 200, "frame.jpg", 0.9, "strong:visual_change", window_id="window_1"),
        KeyFrame("far", 100_000, "frame.jpg", 0.9, "medium:visual_change", window_id="window_2"),
    ]

    selected = _select_ocr_frames_for_ocr(frames)

    assert [frame.frame_id for frame in selected] == ["medium", "strong", "far"]


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
    assert keep_duplicate_log_anchor is False
    assert keep_changed_log_anchor is True


def test_ocr_results_are_deduped_per_window() -> None:
    frame_a = KeyFrame("a", 1000, "a.jpg", 0.9, "strong:visual_change", window_id="window_0")
    frame_b = KeyFrame("b", 2000, "b.jpg", 0.9, "strong:visual_change", window_id="window_0")
    frame_c = KeyFrame("c", 3000, "c.jpg", 0.9, "strong:visual_change", window_id="window_1")
    results = [
        OcrResult(frame_a, "Send confidential contract", 0.9, "tesseract"),
        OcrResult(frame_b, "Send confidential contract", 0.9, "tesseract"),
        OcrResult(frame_c, "Send confidential contract", 0.9, "tesseract"),
    ]

    deduped = _dedupe_ocr_results(results, VisionConfig(ocr_text_similarity_threshold=0.92))

    assert [item.frame.frame_id for item in deduped] == ["a", "c"]


def test_rapidocr_provider_downscales_large_images(tmp_path: Path) -> None:
    cv2 = __import__("cv2")
    image_path = tmp_path / "large.jpg"
    image = __import__("numpy").zeros((900, 1600, 3), dtype="uint8")
    cv2.imwrite(str(image_path), image)

    loaded = RapidOcrProvider(max_image_side=800)._load_image(str(image_path))

    assert max(loaded.shape[:2]) == 800


def test_vision_artifact_export_writes_raw_and_ocr_selected_frames(tmp_path: Path) -> None:
    image = tmp_path / "frame.jpg"
    image.write_bytes(b"fake image")
    raw_frame = KeyFrame("raw", 1000, str(image), 0.9, "strong:visual_change", window_id="window_0")
    selected_frame = KeyFrame("selected", 2000, str(image), 0.8, "strong:visual_change", window_id="window_0")
    ocr = OcrResult(selected_frame, "蓝牙文件传送 文字文稿1.docx", 0.95, "rapidocr")

    manifest = _export_vision_artifacts(
        artifact_dir=tmp_path / "vision",
        keyframes=[raw_frame],
        ocr_selected_frames=[selected_frame],
        ocr_results=[ocr],
    )

    assert Path(manifest["keyframes_raw_dir"]).exists()
    assert Path(manifest["keyframes_ocr_selected_dir"]).exists()
    assert len(manifest["keyframes_raw_files"]) == 1
    assert len(manifest["keyframes_ocr_selected_files"]) == 1
    assert "文字文稿1.docx" in Path(manifest["ocr_results_file"]).read_text(encoding="utf-8")


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


def test_sensitive_source_extraction_is_configurable(tmp_path: Path) -> None:
    groundtruth = tmp_path / "groundtruth.json"
    groundtruth.write_text(
        json.dumps(
            {
                "labels": [
                    {"source_path": "C://Users//alice//Documents//strategy.docx", "derived_path": "C:/tmp/export.pdf"}
                ]
            }
        ),
        encoding="utf-8",
    )
    config = SensitiveSourceConfig(fields=("source_path",), regexes=())

    sources = extract_sensitive_sources(groundtruth, config)

    assert sources == ("C:/Users/alice/Documents/strategy.docx",)


def test_policy_terms_are_loaded_from_external_config(tmp_path: Path) -> None:
    policy_file = tmp_path / "policy.json"
    policy_file.write_text(
        json.dumps(
            {
                "sink_tokens": ["外发审批"],
                "sink_classification": [{"type": "approval_portal", "tokens": ["外发审批"]}],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    policy = load_policy_config(policy_file)

    assert contains_any("正在提交外发审批", policy.sink_tokens)
    assert policy.sink_classification == (("approval_portal", ("外发审批",)),)


def test_vlm_frame_selection_uses_policy_terms() -> None:
    frame = KeyFrame("frame_1", 1000, "frame.jpg", 0.9, "visual_change")
    ocr = OcrResult(frame=frame, text="正在上传工资表到网盘", confidence=0.99, provider="unit")

    selected = choose_vlm_frames([ocr], min_confidence=0.70, max_frames=4)

    assert selected and selected[0].frame.frame_id == "frame_1"


def test_vlm_frame_selection_requires_real_ocr_before_low_confidence_fallback() -> None:
    frame = KeyFrame("frame_1", 1000, "frame.jpg", 0.9, "visual_change")
    no_ocr = OcrResult(frame=frame, text="", confidence=0.0, provider="none")
    weak_ocr = OcrResult(frame=frame, text="模糊文字", confidence=0.2, provider="tesseract")

    assert choose_vlm_frames([no_ocr], min_confidence=0.70, max_frames=4) == []
    assert choose_vlm_frames([weak_ocr], min_confidence=0.70, max_frames=4)


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


def test_pipeline_conclusion_prefers_groundtruth_when_available(tmp_path: Path) -> None:
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

    report = run_pipeline(log_file=log_file, groundtruth_file=groundtruth, neo4j_enabled=False)

    assert report["conclusion"] == "data_leak_risk_detected"
    assert report["verdict"]["source"] == "groundtruth"
    assert report["leak_reasoner"]["detector_conclusion"] == "no_confirmed_data_leak"
    assert report["detection_core"]["method"] == "non_uniform_keyframes_ocr_vlm_datalog"
    assert report["detection_core"]["evaluation"]["groundtruth_is_evaluation_only"] is True


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
