import importlib.util
import json
import os
import sys
import types
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
REALISTIC_CASES_PATH = REPO_ROOT / "fixtures" / "realistic_log_cases.json"
QWEN_VLM_CASES_PATH = REPO_ROOT / "fixtures" / "qwen_vlm_response_cases.json"


def load_module_from_path(module_name: str, file_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class E2ERegressionTests(unittest.TestCase):
    def test_prompt_loader_ignores_conflicting_prompts_module(self):
        prompt_loader_path = REPO_ROOT / "1-FrameAnalyzer" / "prompt_loader.py"
        original_prompts = sys.modules.get("prompts")
        sys.modules["prompts"] = types.ModuleType("prompts")
        try:
            prompt_loader = load_module_from_path("frame_prompt_loader_test", prompt_loader_path)
        finally:
            if original_prompts is None:
                sys.modules.pop("prompts", None)
            else:
                sys.modules["prompts"] = original_prompts

        self.assertTrue(hasattr(prompt_loader.PROMPTS, "RETRIEVE_FRAMES_PROMPT"))
        self.assertTrue(hasattr(prompt_loader.PROMPTS, "SCENE_DEEP_DIVE_PROMPT"))

    def test_critical_accuracy_fixtures_do_not_contain_mojibake(self):
        bad_tokens = ["鍛", "姝", "绮", "閭", "娴", "鐢", "鏈", "涓", "杈", "宸", "甯", "瀹", "寰", "浼", "钖", "妫", "闀", "�"]
        paths = [
            REPO_ROOT / "1-FrameAnalyzer" / "agent.py",
            REALISTIC_CASES_PATH,
            QWEN_VLM_CASES_PATH,
        ]

        offenders = []
        for path in paths:
            text = path.read_text(encoding="utf-8")
            found = [token for token in bad_tokens if token in text]
            if found:
                offenders.append(f"{path.relative_to(REPO_ROOT)}: {found}")

        self.assertEqual(offenders, [])

    def test_frame_analyzer_limits_vlm_frames_and_keeps_context(self):
        module_dir = REPO_ROOT / "1-FrameAnalyzer"
        sys.path.insert(0, str(module_dir))
        old_limit = os.environ.get("DLD_VLM_MAX_FRAMES")
        os.environ["DLD_VLM_MAX_FRAMES"] = "5"
        try:
            agent_module = load_module_from_path(
                "frame_agent_optimization_test",
                module_dir / "agent.py",
            )
            agent = agent_module.VideoFileOperationAgent.__new__(agent_module.VideoFileOperationAgent)
            frames = [
                {
                    "idx": idx,
                    "type": "key_event",
                    "time": f"2026-06-28 10:00:{idx:02d}",
                    "ocr_score": idx,
                    "ocr_text": f"salary table frame {idx}",
                }
                for idx in range(1, 11)
            ]
            frames.append(dict(frames[4]))
            frames.extend(
                [
                    {"idx": 13, "type": "context", "time": "2026-06-28 10:00:13"},
                    {"idx": 18, "type": "context", "time": "2026-06-28 10:00:18"},
                    {"idx": 25, "type": "context", "time": "2026-06-28 10:00:25"},
                ]
            )

            selected, meta = agent._select_vlm_frames(frames)
        finally:
            sys.path.pop(0)
            if old_limit is None:
                os.environ.pop("DLD_VLM_MAX_FRAMES", None)
            else:
                os.environ["DLD_VLM_MAX_FRAMES"] = old_limit

        self.assertEqual(len(selected), 5)
        self.assertEqual(meta["candidate_hit_frames"], 14)
        self.assertEqual(meta["deduped_hit_frames"], 13)
        self.assertEqual(meta["vlm_sent_frames"], 5)
        self.assertEqual([item["idx"] for item in selected[-3:]], [13, 18, 25])

    def test_qwen_vlm_response_postprocessing_filters_duplicates_and_noise(self):
        module_dir = REPO_ROOT / "1-FrameAnalyzer"
        sys.path.insert(0, str(module_dir))
        try:
            agent_module = load_module_from_path(
                "frame_agent_qwen_postprocess_test",
                module_dir / "agent.py",
            )
            agent = agent_module.VideoFileOperationAgent.__new__(agent_module.VideoFileOperationAgent)
        finally:
            sys.path.pop(0)

        with open(QWEN_VLM_CASES_PATH, "r", encoding="utf-8") as handle:
            cases = json.load(handle)

        for case in cases:
            with self.subTest(case=case["id"]):
                parsed = agent._parse_vlm_response_content(case["response"])
                raw_events = agent._coerce_event_list(parsed)
                final_events, meta = agent._filter_vlm_events(raw_events, case["keywords"])

                self.assertEqual(len(final_events), case["expected_kept"])
                if case["expected_kept"]:
                    self.assertEqual(final_events[0]["operation_type"], case["expected_operation"])
                self.assertEqual(meta["vlm_kept_events"], case["expected_kept"])
                self.assertEqual(meta["vlm_dropped_events"], len(raw_events) - case["expected_kept"])

    def test_qwen_guardrail_prompt_contains_false_positive_constraints(self):
        module_dir = REPO_ROOT / "1-FrameAnalyzer"
        sys.path.insert(0, str(module_dir))
        try:
            agent_module = load_module_from_path(
                "frame_agent_qwen_guardrail_test",
                module_dir / "agent.py",
            )
            guardrails = agent_module.VideoFileOperationAgent._qwen_guardrail_prompt()
        finally:
            sys.path.pop(0)

        self.assertIn('{"events": []}', guardrails)
        self.assertIn("Do not create duplicate events", guardrails)
        self.assertIn("A chat window alone is not enough", guardrails)

    def test_sync_processed_statistics_uses_processed_count(self):
        stats_module = load_module_from_path(
            "upload_detector_stats_test",
            REPO_ROOT / "3-RiskHunter" / "upload_detector_stats.py",
        )
        state = {"processed_count": 13, "statistics": {}}

        stats_module.sync_processed_statistics(state)

        self.assertEqual(state["statistics"]["total_events_processed"], 13)

    def test_vlm_fallback_gate_runs_for_ai_only_sample_but_skips_benign_sample(self):
        run_e2e = load_module_from_path("run_e2e_vlm_gate_test", REPO_ROOT / "run_e2e.py")
        module_dir = REPO_ROOT / "3-RiskHunter"
        sys.path.insert(0, str(module_dir))
        try:
            log_first_module = load_module_from_path(
                "log_first_detector_vlm_gate_test",
                module_dir / "log_first_detector.py",
            )
        finally:
            sys.path.pop(0)

        sensitive_file = "C:/Users/test/Desktop/\u5458\u5de5\u85aa\u8d44\u660e\u7ec6\u8868Q4.xlsx"
        detector = log_first_module.LogFirstDetector(
            sensitive_files=[sensitive_file],
            blacklist_apps=["ChatGPT"],
            whitelist_apps=["Excel"],
        )
        ai_only_logs = [
            {
                "timestamp": "2026-06-28T10:00:00.000",
                "event_type": "file_open",
                "file_path": sensitive_file,
                "file_name": "\u5458\u5de5\u85aa\u8d44\u660e\u7ec6\u8868Q4.xlsx",
                "process_info": {"process_name": "excel.exe"},
                "window_info": {"window_title": "\u5458\u5de5\u85aa\u8d44\u660e\u7ec6\u8868Q4.xlsx - Excel"},
            },
            {
                "timestamp": "2026-06-28T10:00:18.000",
                "event_type": "clipboard_paste",
                "file_path": "",
                "file_name": "",
                "process_info": {"process_name": "msedge.exe"},
                "window_info": {
                    "window_title": "ChatGPT - New chat - summarize employee salary table"
                },
            },
        ]
        ai_log_first = detector.analyze(ai_only_logs)

        should_run_ai, ai_meta = run_e2e._should_use_vlm_fallback(ai_only_logs, ai_log_first)

        self.assertEqual(ai_log_first["upload_events"], [])
        self.assertTrue(should_run_ai)
        self.assertEqual(ai_meta["decision"], "run")
        self.assertIn("ai_context_near_sensitive_log", ai_meta["reasons"])

        benign_logs = [
            ai_only_logs[0],
            {
                "timestamp": "2026-06-28T10:00:18.000",
                "event_type": "modified",
                "file_path": sensitive_file,
                "file_name": "\u5458\u5de5\u85aa\u8d44\u660e\u7ec6\u8868Q4.xlsx",
                "process_info": {"process_name": "excel.exe"},
                "window_info": {"window_title": "\u5458\u5de5\u85aa\u8d44\u660e\u7ec6\u8868Q4.xlsx - Excel"},
            },
        ]
        benign_log_first = detector.analyze(benign_logs)

        should_run_benign, benign_meta = run_e2e._should_use_vlm_fallback(benign_logs, benign_log_first)

        self.assertEqual(benign_log_first["upload_events"], [])
        self.assertFalse(should_run_benign)
        self.assertEqual(benign_meta["decision"], "skip")
        self.assertIn("no_ai_or_ambiguous_exfil_context", benign_meta["reasons"])

    def test_vlm_fallback_gate_handles_noisy_and_complex_contexts(self):
        run_e2e = load_module_from_path("run_e2e_complex_gate_test", REPO_ROOT / "run_e2e.py")
        module_dir = REPO_ROOT / "3-RiskHunter"
        sys.path.insert(0, str(module_dir))
        old_window = os.environ.get("DLD_VLM_FALLBACK_WINDOW_SEC")
        os.environ["DLD_VLM_FALLBACK_WINDOW_SEC"] = "300"
        self.addCleanup(
            lambda: (
                os.environ.pop("DLD_VLM_FALLBACK_WINDOW_SEC", None)
                if old_window is None
                else os.environ.__setitem__("DLD_VLM_FALLBACK_WINDOW_SEC", old_window)
            )
        )
        try:
            log_first_module = load_module_from_path(
                "log_first_detector_complex_gate_test",
                module_dir / "log_first_detector.py",
            )
        finally:
            sys.path.pop(0)

        sensitive_file = "C:/Users/test/Desktop/\u5185\u90e8\u6218\u7565\u89c4\u5212.docx"
        detector = log_first_module.LogFirstDetector(
            sensitive_files=[sensitive_file],
            blacklist_apps=["ChatGPT", "Feishu"],
            whitelist_apps=["Word"],
        )
        sensitive_open = {
            "timestamp": "2026-06-28T09:00:00.000",
            "event_type": "file_open",
            "file_path": sensitive_file,
            "file_name": "\u5185\u90e8\u6218\u7565\u89c4\u5212.docx",
            "process_info": {"process_name": "winword.exe"},
            "window_info": {"window_title": "\u5185\u90e8\u6218\u7565\u89c4\u5212.docx - Word"},
        }

        far_ai_noise = [
            sensitive_open,
            {
                "timestamp": "2026-06-28T09:20:01.000",
                "event_type": "window_focus",
                "file_path": "",
                "file_name": "",
                "process_info": {"process_name": "msedge.exe"},
                "window_info": {"window_title": "ChatGPT - unrelated travel planning"},
            },
        ]
        far_log_first = detector.analyze(far_ai_noise)
        should_run_far, far_meta = run_e2e._should_use_vlm_fallback(far_ai_noise, far_log_first)

        self.assertFalse(should_run_far)
        self.assertEqual(far_meta["decision"], "skip")
        self.assertIn("no_ai_or_ambiguous_exfil_context", far_meta["reasons"])

        no_sensitive_ai = [
            {
                "timestamp": "2026-06-28T09:01:00.000",
                "event_type": "window_focus",
                "file_path": "",
                "file_name": "",
                "process_info": {"process_name": "msedge.exe"},
                "window_info": {"window_title": "ChatGPT - summarize public news"},
            }
        ]
        no_sensitive_log_first = detector.analyze(no_sensitive_ai)
        should_run_no_sensitive, no_sensitive_meta = run_e2e._should_use_vlm_fallback(
            no_sensitive_ai,
            no_sensitive_log_first,
        )

        self.assertFalse(should_run_no_sensitive)
        self.assertEqual(no_sensitive_meta["decision"], "skip")
        self.assertIn("no_sensitive_log_context", no_sensitive_meta["reasons"])

        clipboard_near_sensitive = [
            sensitive_open,
            {
                "timestamp": "2026-06-28T09:00:30.000",
                "event_type": "clipboard_paste",
                "file_path": "",
                "file_name": "",
                "process_info": {"process_name": "lark.exe"},
                "window_info": {"window_title": "Feishu external partner - paste into message composer"},
            },
        ]
        clipboard_log_first = detector.analyze(clipboard_near_sensitive)
        should_run_clipboard, clipboard_meta = run_e2e._should_use_vlm_fallback(
            clipboard_near_sensitive,
            clipboard_log_first,
        )

        self.assertTrue(should_run_clipboard)
        self.assertEqual(clipboard_meta["decision"], "run")
        self.assertIn("ambiguous_exfil_context_near_sensitive_log", clipboard_meta["reasons"])

        ordinary_chat_near_sensitive = [
            sensitive_open,
            {
                "timestamp": "2026-06-28T09:00:30.000",
                "event_type": "window_focus",
                "file_path": "",
                "file_name": "",
                "process_info": {"process_name": "lark.exe"},
                "window_info": {"window_title": "Feishu team chat - daily standup"},
            },
        ]
        ordinary_chat_log_first = detector.analyze(ordinary_chat_near_sensitive)
        should_run_chat, chat_meta = run_e2e._should_use_vlm_fallback(
            ordinary_chat_near_sensitive,
            ordinary_chat_log_first,
        )

        self.assertFalse(should_run_chat)
        self.assertEqual(chat_meta["decision"], "skip")
        self.assertIn("no_ai_or_ambiguous_exfil_context", chat_meta["reasons"])

    def test_realistic_log_fixtures_match_expected_detection_policy(self):
        run_e2e = load_module_from_path("run_e2e_realistic_fixture_test", REPO_ROOT / "run_e2e.py")
        module_dir = REPO_ROOT / "3-RiskHunter"
        sys.path.insert(0, str(module_dir))
        old_window = os.environ.get("DLD_VLM_FALLBACK_WINDOW_SEC")
        os.environ["DLD_VLM_FALLBACK_WINDOW_SEC"] = "300"
        self.addCleanup(
            lambda: (
                os.environ.pop("DLD_VLM_FALLBACK_WINDOW_SEC", None)
                if old_window is None
                else os.environ.__setitem__("DLD_VLM_FALLBACK_WINDOW_SEC", old_window)
            )
        )
        try:
            log_first_module = load_module_from_path(
                "log_first_detector_realistic_fixture_test",
                module_dir / "log_first_detector.py",
            )
        finally:
            sys.path.pop(0)

        with open(REALISTIC_CASES_PATH, "r", encoding="utf-8") as handle:
            cases = json.load(handle)

        for case in cases:
            with self.subTest(case=case["id"]):
                detector = log_first_module.LogFirstDetector(
                    sensitive_files=case["sensitive_files"],
                    blacklist_apps=[
                        "ChatGPT",
                        "Feishu",
                        "Lark",
                        "163邮箱",
                        "mail.163.com",
                        "msedge.exe",
                    ],
                    whitelist_apps=[
                        "Excel",
                        "Word",
                        "WeCom",
                        "企业微信",
                        "WeCom.exe",
                    ],
                )
                result = detector.analyze(case["logs"])
                expected = case["expected"]

                self.assertEqual(len(result["upload_events"]), expected["upload_events"])
                self.assertEqual(len(result["alert_events"]), expected["alert_events"])

                if expected["upload_events"]:
                    first_event = result["upload_events"][0]
                    self.assertEqual(first_event.app_category, expected["app_category"])
                    self.assertTrue(first_event.original_file)
                    self.assertEqual(expected["vlm_decision"], "skip_deterministic")
                    continue

                should_run_vlm, meta = run_e2e._should_use_vlm_fallback(case["logs"], result)
                self.assertEqual(meta["decision"], expected["vlm_decision"])
                self.assertEqual(should_run_vlm, expected["vlm_decision"] == "run")
                self.assertIn(expected["vlm_reason"], meta["reasons"])

    def test_realistic_fixture_matrix_covers_major_policy_paths(self):
        with open(REALISTIC_CASES_PATH, "r", encoding="utf-8") as handle:
            cases = json.load(handle)

        decisions = {case["expected"]["vlm_decision"] for case in cases}
        categories = {
            case["expected"].get("app_category")
            for case in cases
            if case["expected"].get("app_category")
        }
        reasons = {
            case["expected"].get("vlm_reason")
            for case in cases
            if case["expected"].get("vlm_reason")
        }

        self.assertGreaterEqual(len(cases), 12)
        self.assertIn("skip_deterministic", decisions)
        self.assertIn("run", decisions)
        self.assertIn("skip", decisions)
        self.assertIn("blacklist", categories)
        self.assertIn("whitelist", categories)
        self.assertIn("no_sensitive_log_context", reasons)
        self.assertIn("no_ai_or_ambiguous_exfil_context", reasons)
        self.assertIn("ambiguous_exfil_context_near_sensitive_log", reasons)

    def test_connected_fact_injection_builds_leak_path_for_derived_upload(self):
        run_e2e = load_module_from_path("run_e2e_regression_test", REPO_ROOT / "run_e2e.py")
        run_e2e.import_modules()
        engine = run_e2e.DatalogEngine()
        logs = [
            {
                "timestamp": "2026-03-27T12:30:00.000",
                "event_type": "file_open",
                "file_path": "C:/Users/test/Desktop/员工薪资明细表Q4.xlsx",
                "file_name": "员工薪资明细表Q4.xlsx",
                "process_info": {"process_name": "wps.exe"},
                "window_info": {"window_title": "员工薪资明细表Q4.xlsx - WPS"},
            },
            {
                "timestamp": "2026-03-27T12:30:42.000",
                "event_type": "created",
                "file_path": "C:/Users/test/Desktop/员工薪资明细表Q4_part1.xlsx",
                "file_name": "员工薪资明细表Q4_part1.xlsx",
                "process_info": {"process_name": "msedge.exe"},
                "window_info": {"window_title": ""},
            },
            {
                "timestamp": "2026-03-27T12:31:46.000",
                "event_type": "file_upload",
                "file_path": "C:/Users/test/Desktop/员工薪资明细表Q4_part1.xlsx",
                "file_name": "员工薪资明细表Q4_part1.xlsx",
                "process_info": {"process_name": "msedge.exe"},
                "window_info": {"window_title": "QQ邮箱 - 网页版"},
            },
        ]

        injected_facts = run_e2e._inject_connected_facts_from_module3(
            engine,
            {"file_mappings": {}, "alert_events": [], "info_events": []},
            logs,
        )
        leak_paths = engine.query_leak()
        engine.cleanup()

        self.assertEqual(len(leak_paths), 1)
        self.assertEqual(leak_paths[0].leaking_proc, "msedge.exe")
        self.assertTrue(leak_paths[0].leaked_file.endswith("员工薪资明细表Q4_part1.xlsx"))
        self.assertTrue(any(fact.relation == "CrossProcessTransfer" for fact in injected_facts))

    def test_extract_hidden_operations_splits_semicolon_delimited_outputs(self):
        module_dir = REPO_ROOT / "2-FileTracker"
        sys.path.insert(0, str(module_dir))
        try:
            tools_module = load_module_from_path(
                "behavior_analysis_tools_regression_test",
                module_dir / "behavior_analysis_tools.py",
            )
        finally:
            sys.path.pop(0)

        result = tools_module.extract_hidden_operations.invoke(
            {
                "frame_analysis_result": {
                    "events": [
                        {
                            "behavior_category": "潜在隐藏行为",
                            "operation_type": "格式转换",
                            "original_filename": "员工薪资明细表Q4.xlsx",
                            "modified_filename": "员工薪资明细表Q4_part1.xlsx; 员工薪资明细表Q4_part2.xlsx",
                            "app_name": "cmd.exe",
                            "time_range": "2026-03-27 12:30:33 - 2026-03-27 12:30:48",
                            "involved_timestamps": ["2026-03-27 12:30:33", "2026-03-27 12:30:48"],
                        }
                    ]
                }
            }
        )

        self.assertTrue(result["has_hidden_behavior"])
        self.assertEqual(
            [item["new_file"] for item in result["hidden_operations"]],
            ["员工薪资明细表Q4_part1.xlsx", "员工薪资明细表Q4_part2.xlsx"],
        )

    def test_update_worklist_skips_requeue_for_known_sensitive_derived_file(self):
        module_dir = REPO_ROOT / "2-FileTracker"
        sys.path.insert(0, str(module_dir))
        try:
            nodes_module = load_module_from_path(
                "behavior_analysis_nodes_regression_test",
                module_dir / "behavior_analysis_nodes.py",
            )
            worklist_module = load_module_from_path(
                "worklist_manager_regression_test",
                module_dir / "worklist_manager.py",
            )
        finally:
            sys.path.pop(0)

        manager = worklist_module.WorklistManager(
            sensitive_files=["C:/Users/test/Desktop/员工薪资明细表Q4_part1.xlsx"]
        )
        state = {
            "current_event": worklist_module.SensitiveFileEvent(
                event_id="source",
                original_file="C:/Users/test/Desktop/员工薪资明细表Q4.xlsx",
                current_file="C:/Users/test/Desktop/员工薪资明细表Q4.xlsx",
                event_type="file_open",
                process_info={},
                timestamp="2026-03-27T12:30:10",
            ),
            "new_events": [
                worklist_module.SensitiveFileEvent(
                    event_id="derived",
                    original_file="C:/Users/test/Desktop/员工薪资明细表Q4.xlsx",
                    current_file="C:/Users/test/Desktop/员工薪资明细表Q4_part1.xlsx",
                    event_type="derived_from_格式转换",
                    process_info={"app_name": "cmd.exe"},
                    timestamp="2026-03-27T12:30:33",
                    is_hidden=True,
                    raw_event={"operation_type": "格式转换", "description": "", "time_range": ""},
                )
            ],
        }

        nodes_module.update_worklist_node(state, manager)

        self.assertEqual(manager.size(), 0)
        self.assertEqual(
            manager.get_original_file("C:/Users/test/Desktop/员工薪资明细表Q4_part1.xlsx"),
            "C:/Users/test/Desktop/员工薪资明细表Q4.xlsx",
        )

    def test_python_datalog_engine_does_not_expand_transfer_cycles(self):
        module_dir = REPO_ROOT / "4-ThreatDetector"
        sys.path.insert(0, str(module_dir))
        try:
            engine_module = load_module_from_path(
                "python_datalog_engine_regression_test",
                module_dir / "datalog" / "python_datalog_engine.py",
            )
        finally:
            sys.path.pop(0)

        engine = engine_module.PythonDatalogEngine()
        engine.add_fact("OpenFile", "open_1", "excel.exe", "orig.xlsx", 1)
        engine.add_fact("TransferFile", "transfer_1", "excel.exe", "orig.xlsx", "part1.xlsx", 2)
        engine.add_fact("TransferFile", "transfer_2", "excel.exe", "part1.xlsx", "orig.xlsx", 3)
        engine.add_fact("LeakFile", "leak_1", "excel.exe", "part1.xlsx", "email", 4)

        leak_paths = engine.run_inference()

        self.assertEqual(len(leak_paths), 1)
        self.assertEqual(leak_paths[0].leaked_file, "part1.xlsx")
        self.assertEqual(leak_paths[0].full_path, "open_1 -> transfer_1 -> leak_1")


if __name__ == "__main__":
    unittest.main()
