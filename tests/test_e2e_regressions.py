import importlib.util
import sys
import types
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


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

    def test_sync_processed_statistics_uses_processed_count(self):
        stats_module = load_module_from_path(
            "upload_detector_stats_test",
            REPO_ROOT / "3-RiskHunter" / "upload_detector_stats.py",
        )
        state = {"processed_count": 13, "statistics": {}}

        stats_module.sync_processed_statistics(state)

        self.assertEqual(state["statistics"]["total_events_processed"], 13)

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
