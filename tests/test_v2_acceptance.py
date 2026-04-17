import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
MAIN_ROOT = REPO_ROOT / "main"

if str(MAIN_ROOT) not in sys.path:
    sys.path.insert(0, str(MAIN_ROOT))


import main_v2  # noqa: E402


class V2AcceptanceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pipeline_config = main_v2.load_pipeline_config()
        desktop = main_v2.get_windows_desktop()
        cls.sample_10_2 = desktop / "10-2"
        cls.sample_5_2 = desktop / "5-2"
        cls.result_10_2 = main_v2.run_single_sample(
            cls.sample_10_2,
            mode="full",
            pipeline_config=cls.pipeline_config,
        )
        cls.result_5_2 = main_v2.run_single_sample(
            cls.sample_5_2,
            mode="full",
            pipeline_config=cls.pipeline_config,
        )

    def test_v2_pipeline_detects_10_2_and_5_2_in_full_mode(self):
        self.assertTrue(self.result_10_2["detected"])
        self.assertTrue(self.result_5_2["detected"])
        self.assertEqual(self.result_10_2["frame_mode"], "full")
        self.assertEqual(self.result_5_2["frame_mode"], "full")
        self.assertEqual(self.result_10_2["frame_analysis"]["mode"], "full")
        self.assertIn("cache_hit", self.result_10_2["frame_analysis"]["metadata"])
        self.assertEqual(
            self.result_10_2["frame_analysis"]["metadata"]["analysis_backend"],
            "legacy_adapter",
        )
        self.assertIn("fresh_run_requested", self.result_10_2["frame_analysis"]["metadata"])
        self.assertIn("segments", self.result_10_2["frame_analysis"])
        self.assertIn("summary", self.result_10_2["frame_analysis"])
        self.assertGreaterEqual(len(self.result_10_2["frame_analysis"]["segments"]), 1)

    def test_full_mode_produces_single_case_for_mail_attachment_sample(self):
        self.assertEqual(len(self.result_10_2["correlation_bundle"]["upload_candidates"]), 1)
        self.assertEqual(len(self.result_10_2["reasoner_output"]["risk_cases"]), 1)
        candidate = self.result_10_2["correlation_bundle"]["upload_candidates"][0]
        risk_case = self.result_10_2["reasoner_output"]["risk_cases"][0]
        self.assertEqual(candidate["sink_type"], "mail_attachment")
        self.assertEqual(candidate["object_binding"]["binding_type"], "lineage")
        self.assertEqual(risk_case["sink_type"], "mail_attachment")
        self.assertEqual(risk_case["leak_channel"], "mail_attachment")
        ambiguous_precursors = [
            event
            for event in self.result_10_2["correlation_bundle"]["correlated_events"]
            if event["status"] == "ambiguous"
        ]
        self.assertEqual(ambiguous_precursors, [])

    def test_full_mode_keeps_screen_share_detection_for_meeting_sample(self):
        self.assertEqual(len(self.result_5_2["correlation_bundle"]["upload_candidates"]), 1)
        self.assertEqual(len(self.result_5_2["reasoner_output"]["risk_cases"]), 1)
        candidate = self.result_5_2["correlation_bundle"]["upload_candidates"][0]
        risk_case = self.result_5_2["reasoner_output"]["risk_cases"][0]
        self.assertEqual(candidate["sink_type"], "screen_share")
        self.assertIn(
            candidate["object_binding"]["binding_type"],
            {"temporal_screen_share_binding", "lineage", "basename_match"},
        )
        self.assertEqual(risk_case["sink_type"], "screen_share")
        self.assertEqual(risk_case["leak_channel"], "screen_share")

    def test_run_single_sample_surfaces_fresh_run_flag(self):
        result = main_v2.run_single_sample(
            self.sample_10_2,
            mode="full",
            pipeline_config=self.pipeline_config,
            fresh_run=False,
        )

        self.assertIn("fresh_run", result)
        self.assertFalse(result["fresh_run"])
        self.assertIn("cache_hit", result["frame_analysis"]["metadata"])
        self.assertFalse(result["frame_analysis"]["metadata"]["fresh_run_requested"])
        self.assertIn("segments", result["frame_analysis"])
        self.assertIn("summary", result["frame_analysis"])


if __name__ == "__main__":
    unittest.main()
