import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_ROOT = REPO_ROOT / "02-EventCorrelator"

if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))


from event_correlator import EventCorrelator, EventCorrelatorConfig  # noqa: E402
from event_correlator.lineage import LineageBuilder  # noqa: E402
from event_correlator.timeline import TimelineNormalizer  # noqa: E402


class EventCorrelatorTests(unittest.TestCase):
    def setUp(self):
        self.config = EventCorrelatorConfig()
        self.correlator = EventCorrelator(self.config)

    def test_lineage_builder_infers_parent_for_derived_file(self):
        payload = {
            "session_id": "10-2",
            "log_events": [
                {
                    "timestamp": "2026-03-27T12:30:00.000",
                    "event_type": "file_open",
                    "file_path": "C:/Users/test/Desktop/orig.xlsx",
                    "process_info": {"process_name": "wps.exe"},
                },
                {
                    "timestamp": "2026-03-27T12:30:42.000",
                    "event_type": "created",
                    "file_path": "C:/Users/test/Desktop/orig_part1.xlsx",
                    "process_info": {"process_name": "python.exe"},
                },
            ],
            "frame_segments": [],
            "sensitive_files": ["C:/Users/test/Desktop/orig.xlsx"],
        }
        context = TimelineNormalizer().normalize(payload, self.config.as_dict())
        lineage = LineageBuilder(max_depth=10).build(context)

        self.assertEqual(
            lineage.resolve_root("C:/Users/test/Desktop/orig_part1.xlsx", 10),
            "C:/Users/test/Desktop/orig.xlsx",
        )
        self.assertEqual(
            lineage.build_full_chain("C:/Users/test/Desktop/orig_part1.xlsx", 10),
            "C:/Users/test/Desktop/orig.xlsx -> C:/Users/test/Desktop/orig_part1.xlsx",
        )

    def test_correlator_builds_upload_candidate_for_upload_log(self):
        payload = {
            "session_id": "10-2",
            "record_id": "10-2",
            "recording_start_time": "2026-03-27 12:30:00",
            "sensitive_files": ["C:/Users/test/Desktop/orig.xlsx"],
            "log_events": [
                {
                    "timestamp": "2026-03-27T12:30:42.000",
                    "event_type": "created",
                    "file_path": "C:/Users/test/Desktop/orig_part1.xlsx",
                    "process_info": {"process_name": "python.exe"},
                },
                {
                    "timestamp": "2026-03-27T12:31:46.000",
                    "event_type": "file_upload",
                    "file_path": "C:/Users/test/Desktop/orig_part1.xlsx",
                    "process_info": {"process_name": "msedge.exe"},
                    "window_info": {"window_title": "QQMail - Web"},
                },
            ],
            "frame_segments": [
                {
                    "segment_id": "seg_1",
                    "time_range": "2026-03-27 12:31:46 - 2026-03-27 12:32:17",
                    "app_name": "QQMail",
                    "operation_type": "mail_attachment_upload",
                    "primary_resource": "orig_part1.xlsx",
                    "related_resources": [],
                    "action_description": "user uploads an attachment in web mail",
                    "visible_evidence": ["attachment orig_part1.xlsx", "send button"],
                    "supporting_timestamps": ["2026-03-27 12:31:46"],
                    "confidence": 0.9,
                }
            ],
        }

        bundle = self.correlator.run(payload)

        self.assertEqual(bundle["analysis_status"], "success")
        self.assertGreaterEqual(len(bundle["correlated_events"]), 1)
        self.assertGreaterEqual(len(bundle["upload_candidates"]), 1)
        self.assertEqual(
            bundle["file_lineage"]["direct_file_mappings"].get("C:/Users/test/Desktop/orig_part1.xlsx"),
            "C:/Users/test/Desktop/orig.xlsx",
        )
        self.assertEqual(
            bundle["upload_candidates"][0]["original_file"],
            "C:/Users/test/Desktop/orig.xlsx",
        )

    def test_correlator_deduplicates_equivalent_upload_candidates(self):
        payload = {
            "session_id": "dedup",
            "sensitive_files": ["C:/Users/test/Desktop/orig.xlsx"],
            "log_events": [
                {
                    "timestamp": "2026-03-27T12:31:46.000",
                    "event_type": "file_upload",
                    "file_path": "C:/Users/test/Desktop/orig.xlsx",
                    "process_info": {"process_name": "msedge.exe"},
                },
                {
                    "timestamp": "2026-03-27T12:31:48.000",
                    "event_type": "file_upload",
                    "file_path": "C:/Users/test/Desktop/orig.xlsx",
                    "process_info": {"process_name": "msedge.exe"},
                },
            ],
            "frame_segments": [],
        }

        bundle = self.correlator.run(payload)

        self.assertEqual(len(bundle["upload_candidates"]), 1)
        self.assertEqual(bundle["statistics"]["upload_candidates_output"], 1)


if __name__ == "__main__":
    unittest.main()
