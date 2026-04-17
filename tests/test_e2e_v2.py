import sys
import unittest
from pathlib import Path
from unittest import mock
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parents[1]
MAIN_ROOT = REPO_ROOT / "main"

if str(MAIN_ROOT) not in sys.path:
    sys.path.insert(0, str(MAIN_ROOT))


import main_v2  # noqa: E402


class E2EV2Tests(unittest.TestCase):
    def test_demo_segments_are_derived_from_sample_metadata(self):
        seg_10_2 = main_v2.build_demo_segments("10-2")
        seg_5_2 = main_v2.build_demo_segments("5-2")

        self.assertGreaterEqual(len(seg_10_2), 2)
        self.assertGreaterEqual(len(seg_5_2), 2)
        self.assertTrue(any("附件" in segment["operation_type"] for segment in seg_10_2))
        self.assertTrue(any("屏幕" in segment["operation_type"] for segment in seg_5_2))
        self.assertTrue(all(segment["segment_id"].startswith("demo_segment_") for segment in seg_10_2))

    def test_full_context_is_inferred_without_groundtruth_dependency(self):
        desktop = main_v2.get_windows_desktop()
        sample_root = desktop / "10-2"
        log_events = main_v2._load_log_events(sample_root)
        context = main_v2._build_sample_context(sample_root, log_events, main_v2.load_pipeline_config())

        self.assertEqual(context.context_inference["mode"], "full")
        self.assertFalse(context.context_inference["groundtruth_used"])
        self.assertTrue(context.sensitive_files)
        self.assertIn("QQ邮箱", context.target_keywords)

    def test_full_mode_blocks_when_frame_analyzer_fails(self):
        desktop = main_v2.get_windows_desktop()
        sample_root = desktop / "10-2"

        with mock.patch.object(
            main_v2.FrameAnalyzerService,
            "analyze",
            return_value={
                "status": "failed",
                "segments": [],
                "summary": {},
                "analysis_metadata": {
                    "analysis_backend": "legacy_adapter",
                    "analysis_backend_version": "legacy_adapter_v1",
                    "cache_hit": False,
                    "fresh_run_requested": False,
                },
            },
        ):
            result = main_v2.run_single_sample(
                sample_root,
                mode="full",
                pipeline_config=main_v2.load_pipeline_config(),
            )

        self.assertFalse(result["detected"])
        self.assertEqual(result["frame_analysis"]["status"], "failed")
        self.assertEqual(result["correlation_bundle"]["analysis_status"], "blocked_by_frame_analyzer")
        self.assertEqual(result["reasoner_output"]["analysis_status"], "blocked_by_frame_analyzer")
        self.assertEqual(result["correlation_bundle"]["upload_candidates"], [])
        self.assertEqual(result["reasoner_output"]["risk_cases"], [])

    def test_full_mode_blocks_when_frame_analyzer_fails(self):
        desktop = main_v2.get_windows_desktop()
        sample_root = desktop / "10-2"

        with mock.patch.object(
            main_v2.FrameAnalyzerService,
            "analyze",
            return_value={
                "status": "failed",
                "segments": [],
                "summary": {},
                "analysis_metadata": {
                    "analysis_backend": "legacy_adapter",
                    "analysis_backend_version": "legacy_adapter_v1",
                    "cache_hit": False,
                    "fresh_run_requested": False,
                },
            },
        ):
            result = main_v2.run_single_sample(
                sample_root,
                mode="full",
                pipeline_config=main_v2.load_pipeline_config(),
            )

        self.assertFalse(result["detected"])
        self.assertEqual(result["frame_analysis"]["status"], "failed")
        self.assertEqual(result["correlation_bundle"]["analysis_status"], "blocked_by_frame_analyzer")
        self.assertEqual(result["reasoner_output"]["analysis_status"], "blocked_by_frame_analyzer")
        self.assertEqual(result["correlation_bundle"]["upload_candidates"], [])
        self.assertEqual(result["reasoner_output"]["risk_cases"], [])


if __name__ == "__main__":
    unittest.main()
