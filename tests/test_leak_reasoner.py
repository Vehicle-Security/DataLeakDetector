import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_ROOT = REPO_ROOT / "03-LeakReasoner"

if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))


from leak_reasoner import LeakReasoner  # noqa: E402
from leak_reasoner.datalog_bridge import LeakDatalogBridge  # noqa: E402


class LeakReasonerTests(unittest.TestCase):
    def test_reasoner_builds_case_from_upload_candidate(self):
        payload = {
            "session_id": "10-2",
            "correlation_bundle": {
                "upload_candidates": [
                    {
                        "candidate_id": "upload_1",
                        "session_id": "10-2",
                        "timestamp": "2026-03-27 12:31:46",
                        "original_file": "C:/Users/test/Desktop/orig.xlsx",
                        "current_files": [
                            "C:/Users/test/Desktop/orig_part1.xlsx",
                            "C:/Users/test/Desktop/orig_part2.xlsx",
                        ],
                        "app_name": "QQ邮箱",
                        "operation_type": "邮件附件外发",
                        "sink_type": "mail_attachment",
                        "evidence_refs": ["log:1", "segment:1"],
                        "mapping_links": [
                            "C:/Users/test/Desktop/orig.xlsx -> C:/Users/test/Desktop/orig_part1.xlsx",
                            "C:/Users/test/Desktop/orig.xlsx -> C:/Users/test/Desktop/orig_part2.xlsx",
                        ],
                        "confidence": 0.95,
                        "status": "linked",
                    }
                ]
            },
        }

        output = LeakReasoner().run(payload)

        self.assertEqual(output["analysis_status"], "success")
        self.assertEqual(len(output["risk_cases"]), 1)
        self.assertEqual(output["risk_cases"][0]["severity"], "high")
        self.assertEqual(output["risk_cases"][0]["sink_type"], "mail_attachment")
        self.assertEqual(output["risk_cases"][0]["leak_channel"], "mail_attachment")
        self.assertGreaterEqual(output["metrics"]["leak_paths_output"], 1)
        self.assertIn("datalog_leak_path_confirmed", output["risk_cases"][0]["reasons"])

    def test_datalog_bridge_preserves_screen_share_channel(self):
        leak_paths = LeakDatalogBridge().run(
            [
                {
                    "fact_type": "upload_candidate",
                    "timestamp": "2026-03-27 12:46:20",
                    "original_file": "C:/Users/test/Desktop/secret.txt",
                    "current_files": ["C:/Users/test/Desktop/secret.txt"],
                    "sink_type": "screen_share",
                    "app_name": "TencentMeeting",
                }
            ]
        )

        self.assertEqual(len(leak_paths), 1)
        self.assertEqual(leak_paths[0]["leak_channel"], "screen_share")

    def test_reasoner_does_not_emit_case_without_matching_leak_path(self):
        payload = {
            "session_id": "negative",
            "correlation_bundle": {
                "upload_candidates": [
                    {
                        "candidate_id": "upload_1",
                        "session_id": "negative",
                        "timestamp": "2026-03-27 12:31:46",
                        "original_file": "C:/Users/test/Desktop/orig.xlsx",
                        "current_files": [],
                        "app_name": "UnknownApp",
                        "operation_type": "unknown_operation",
                        "sink_type": "unknown_sink",
                        "evidence_refs": ["log:1"],
                        "mapping_links": [],
                        "confidence": 0.95,
                        "status": "linked",
                    }
                ]
            },
        }

        output = LeakReasoner().run(payload)

        self.assertEqual(output["analysis_status"], "no_case")
        self.assertEqual(output["metrics"]["leak_paths_output"], 0)
        self.assertEqual(output["risk_cases"], [])


if __name__ == "__main__":
    unittest.main()
