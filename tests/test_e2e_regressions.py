import sys
import types
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]

FRAME_ROOT = REPO_ROOT / "01-FrameAnalyzer"
CORRELATOR_ROOT = REPO_ROOT / "02-EventCorrelator"
REASONER_ROOT = REPO_ROOT / "03-LeakReasoner"
MAIN_ROOT = REPO_ROOT / "main"

for path in (MAIN_ROOT, FRAME_ROOT, CORRELATOR_ROOT, REASONER_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


def load_module_from_path(module_name: str, file_path: Path):
    import importlib.util

    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class E2ERegressionTests(unittest.TestCase):
    def test_prompt_loader_ignores_conflicting_prompts_module(self):
        prompt_loader_path = FRAME_ROOT / "frame_analyzer" / "legacy_prompt_loader.py"
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

    def test_frame_adapter_preserves_multi_output_split(self):
        from frame_analyzer import adapt_legacy_frame_result

        result = adapt_legacy_frame_result(
            {
                "search_range": {},
                "events": [
                    {
                        "app_name": "cmd.exe",
                        "behavior_category": "hidden_transformation",
                        "operation_type": "format_conversion",
                        "original_filename": "employee_salary_q4.xlsx",
                        "modified_filename": "employee_salary_q4_part1.xlsx; employee_salary_q4_part2.xlsx",
                        "time_range": "2026-03-27 12:30:33 - 2026-03-27 12:30:48",
                        "involved_timestamps": ["2026-03-27 12:30:33", "2026-03-27 12:30:48"],
                        "description": "split file",
                    }
                ],
            }
        )

        self.assertEqual(len(result["segments"]), 1)
        self.assertTrue(result["segments"][0]["segment_id"].startswith("segment_"))
        self.assertNotIn("legacy_", result["segments"][0]["segment_id"])
        self.assertEqual(
            result["segments"][0]["related_resources"],
            ["employee_salary_q4_part1.xlsx", "employee_salary_q4_part2.xlsx"],
        )

    def test_new_lineage_state_resolves_known_derived_file(self):
        from event_correlator.lineage import LineageState

        lineage = LineageState(
            sensitive_roots={"C:/Users/test/Desktop/employee_salary_q4.xlsx"},
            direct_mappings={},
            root_mappings={},
        )
        lineage.add_mapping(
            "C:/Users/test/Desktop/employee_salary_q4.xlsx",
            "C:/Users/test/Desktop/employee_salary_q4_part1.xlsx",
        )

        self.assertEqual(
            lineage.resolve_root("C:/Users/test/Desktop/employee_salary_q4_part1.xlsx", 10),
            "C:/Users/test/Desktop/employee_salary_q4.xlsx",
        )

    def test_new_python_datalog_engine_does_not_expand_transfer_cycles(self):
        from leak_reasoner.python_datalog_engine import PythonDatalogEngine

        engine = PythonDatalogEngine()
        engine.add_fact("OpenFile", "open_1", "excel.exe", "orig.xlsx", 1)
        engine.add_fact("TransferFile", "transfer_1", "excel.exe", "orig.xlsx", "part1.xlsx", 2)
        engine.add_fact("TransferFile", "transfer_2", "excel.exe", "part1.xlsx", "orig.xlsx", 3)
        engine.add_fact("LeakFile", "leak_1", "excel.exe", "part1.xlsx", "email", 4)

        leak_paths = engine.run_inference()

        self.assertEqual(len(leak_paths), 1)
        self.assertEqual(leak_paths[0].leaked_file, "part1.xlsx")
        self.assertEqual(leak_paths[0].full_path, "open_1 -> transfer_1 -> leak_1")

    def test_pipeline_support_infers_context_from_sample_files(self):
        import main_v2

        desktop = main_v2.get_windows_desktop()
        sample_root = desktop / "10-2"
        log_events = main_v2._load_log_events(sample_root)
        context = main_v2._build_sample_context(sample_root, log_events, main_v2.load_pipeline_config())

        self.assertEqual(context.record_id, "10-2")
        self.assertTrue(context.sensitive_files)
        self.assertTrue(context.target_keywords)
        self.assertTrue(context.video_path.endswith(".mp4"))
        self.assertFalse(context.context_inference["groundtruth_used"])
        self.assertEqual(context.context_inference["mode"], "full")


if __name__ == "__main__":
    unittest.main()
