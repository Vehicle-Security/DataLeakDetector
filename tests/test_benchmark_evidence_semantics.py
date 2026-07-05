from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_PATH = REPO_ROOT / "tools" / "benchmark_nas_samples.py"
MAIN_DIR = REPO_ROOT / "main"
if str(MAIN_DIR) not in sys.path:
    sys.path.insert(0, str(MAIN_DIR))

from data_leak_detector.evidence_semantics import decide_evidence_outcome  # noqa: E402


def load_benchmark_module():
    spec = importlib.util.spec_from_file_location("benchmark_nas_samples", BENCHMARK_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class BenchmarkEvidenceSemanticsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.benchmark = load_benchmark_module()

    def test_log_selection_rule_is_risk_only(self) -> None:
        bm = self.benchmark
        sensitive_files = ["C:/secret/plan.pdf"]
        actions = bm._log_rule_actions(
            "case-staging",
            {
                "positive": True,
                "rules": ["file_selected"],
                "evidence": {
                    "file_selected": [
                        {
                            "file_path": "C:/secret/plan.pdf",
                            "timestamp": "2026-07-05T10:00:00",
                            "event_type": "file_selected",
                        }
                    ]
                },
            },
            sensitive_files,
        )

        decision = bm._run_datalog_on_audit_actions("case-staging", actions, sensitive_files)

        self.assertTrue(decision["risk_positive"])
        self.assertFalse(decision["confirmed_leak"])
        self.assertNotIn("file_selected", bm.LOG_RULE_LEAK_RULES)
        self.assertFalse(any(fact["relation"] == "LeakFile" for fact in decision["facts"]))

    def test_completed_upload_can_confirm_leak(self) -> None:
        bm = self.benchmark
        sensitive_files = ["C:/secret/plan.pdf"]
        actions = [
            {
                "action_id": "case-completed:upload",
                "action_type": "upload_complete",
                "risk_level": "completed",
                "time": "2026-07-05T10:01:00",
                "app": "browser",
                "app_category": "cloud_storage",
                "source_file": "C:/secret/plan.pdf",
                "confidence": 0.98,
                "description": "upload completed successfully for C:/secret/plan.pdf",
                "evidence_source": "remote_vlm",
            }
        ]

        decision = bm._run_datalog_on_audit_actions("case-completed", actions, sensitive_files)

        self.assertTrue(decision["risk_positive"])
        self.assertTrue(decision["confirmed_leak"])
        self.assertTrue(any(fact["relation"] == "LeakFile" for fact in decision["facts"]))

    def test_clipboard_actions_feed_datalog_clipboard_facts(self) -> None:
        bm = self.benchmark
        sensitive_files = ["C:/secret/plan.pdf"]
        actions = [
            {
                "action_id": "case-clipboard:copy",
                "action_type": "copy_content",
                "risk_level": "selected_or_attached",
                "time": "2026-07-05T10:00:00",
                "app": "excel.exe",
                "source_file": "C:/secret/plan.pdf",
                "description": "copied sensitive content from C:/secret/plan.pdf",
                "evidence_source": "remote_vlm",
            },
            {
                "action_id": "case-clipboard:paste",
                "action_type": "paste_content",
                "risk_level": "content_exposed",
                "time": "2026-07-05T10:00:30",
                "app": "browser.exe",
                "app_category": "ai_service",
                "source_file": "C:/secret/plan.pdf",
                "description": "pasted sensitive content into an external AI input",
                "evidence_source": "remote_vlm",
            },
        ]

        facts = bm._audit_actions_to_datalog_facts("case-clipboard", actions, sensitive_files)
        relations = [fact["relation"] for fact in facts]

        self.assertIn("ClipboardWrite", relations)
        self.assertIn("ClipboardRead", relations)

    def test_explicit_cross_process_action_feeds_datalog_fact(self) -> None:
        bm = self.benchmark
        sensitive_files = ["C:/secret/plan.pdf"]
        actions = [
            {
                "action_id": "case-cross:transfer",
                "action_type": "paste_content",
                "risk_level": "content_exposed",
                "time": "2026-07-05T10:00:30",
                "app": "browser.exe",
                "source_file": "C:/secret/plan.pdf",
                "from_process": "excel.exe",
                "to_process": "browser.exe",
                "shared_data": "C:/secret/plan.pdf",
                "description": "sensitive content moved from Excel to browser",
                "evidence_source": "remote_vlm",
            }
        ]

        facts = bm._audit_actions_to_datalog_facts("case-cross", actions, sensitive_files)

        self.assertTrue(any(fact["relation"] == "CrossProcessTransfer" for fact in facts))

    def test_vlm_contradictory_completed_action_is_downgraded(self) -> None:
        bm = self.benchmark
        sensitive_files = ["C:/secret/plan.pdf"]
        actions = bm._vlm_actions(
            "case-vlm-contradiction",
            {
                "status": "success",
                "risk_level": "completed",
                "is_violation": True,
                "observed_actions": [
                    {
                        "action_id": "bad-send",
                        "action_type": "send_message",
                        "risk_level": "completed",
                        "app": "browser.exe",
                        "app_category": "messaging",
                        "source_file": "C:/secret/plan.pdf",
                        "description": "not sent yet; no visual confirmation; no sensitive file exposed",
                    }
                ],
            },
        )

        decision = bm._run_datalog_on_audit_actions("case-vlm-contradiction", actions, sensitive_files)

        self.assertEqual(actions[0]["action_type"], "none")
        self.assertEqual(actions[0]["risk_level"], "none")
        self.assertEqual(actions[0]["consistency_reason"], "downgraded_vlm_contradiction")
        self.assertFalse(decision["risk_positive"])
        self.assertFalse(decision["confirmed_leak"])

    def test_log_rule_without_file_evidence_does_not_bind_first_sensitive_file(self) -> None:
        bm = self.benchmark
        actions = bm._log_rule_actions(
            "case-unbound-log",
            {
                "positive": True,
                "rules": ["upload_staging"],
                "evidence": {
                    "upload_staging": [
                        {
                            "timestamp": "2026-07-05T10:00:00",
                            "event_type": "upload_staging",
                            "detail": "upload dialog opened without a file path",
                        }
                    ]
                },
            },
            ["C:/secret/plan.pdf"],
        )

        self.assertEqual(actions, [])

    def test_vlm_only_positive_requires_confirmed_risk(self) -> None:
        bm = self.benchmark

        self.assertFalse(
            bm._vlm_only_confirmed_positive(
                {"status": "success", "is_violation": True, "risk_level": "selected_or_attached"}
            )
        )
        self.assertTrue(
            bm._vlm_only_confirmed_positive(
                {"status": "success", "is_violation": True, "risk_level": "content_exposed"}
            )
        )

    def test_benchmark_summary_exposes_ablation_metrics(self) -> None:
        bm = self.benchmark
        summary = bm.BenchmarkSummary().to_dict()["summary"]

        self.assertIn("rules_only", summary)
        self.assertIn("vlm_only", summary)
        self.assertEqual(summary["final_semantics"], "confirmed_leak")

    def test_semantic_frame_coverage_extracts_vlm_anchors(self) -> None:
        bm = self.benchmark
        coverage = bm._semantic_frame_coverage(
            {
                "reason": "Sensitive content is visible in the input of an external AI service.",
                "frame_selection": [
                    {
                        "index": 1,
                        "image_sent": True,
                        "ocr_ran": True,
                        "ocr_flags": ["sensitive_file"],
                        "selection_reason": "event_anchor_transfer",
                    }
                ],
                "observed_actions": [
                    {
                        "action_type": "external_exposure",
                        "risk_level": "content_exposed",
                        "app_category": "ai_service",
                        "source_file": "C:/secret/plan.pdf",
                        "description": "pasted into the external input",
                    }
                ],
            }
        )

        self.assertTrue(coverage["available"])
        self.assertTrue(coverage["content_exposed_anchor"])
        self.assertTrue(coverage["external_sink_anchor"])
        self.assertTrue(coverage["sensitive_object_anchor"])
        self.assertEqual(coverage["sampled_frames"], 1)

    def test_monitor_ui_ocr_does_not_create_completion_flag(self) -> None:
        bm = self.benchmark

        monitor_flags = bm._ocr_risk_flags(
            "Win Monitor localhost:5000 Safe 已完成 Windows 数据泄露行为监控系统",
            ["C:/secret/plan.pdf"],
        )
        upload_flags = bm._ocr_risk_flags(
            "Upload completed successfully",
            ["C:/secret/plan.pdf"],
        )

        self.assertNotIn("completion_keyword", monitor_flags)
        self.assertIn("completion_keyword", upload_flags)

    def test_recording_start_ignores_stale_groundtruth_when_video_matches_logs(self) -> None:
        bm = self.benchmark
        start = bm._recording_start(
            {"recording_start_time": "2026-03-22 17:40:18"},
            [{"timestamp": "2026-03-25T12:11:48.401"}],
            Path("recording_20260325_121148.mp4"),
        )

        self.assertEqual(start, bm._parse_dt("2026-03-25 12:11:48"))

    def test_segment_scoring_prefers_strong_transfer_window(self) -> None:
        bm = self.benchmark
        early_segment = (bm._parse_dt("2026-03-25 12:11:48"), bm._parse_dt("2026-03-25 12:12:33"))
        late_segment = (bm._parse_dt("2026-03-25 12:12:23"), bm._parse_dt("2026-03-25 12:13:08"))
        logs = [
            {
                "timestamp": "2026-03-25T12:11:50",
                "event_type": "modified",
                "app_name": "Edge",
                "file_path": f"C:/noise/{index}.tmp",
                "window_info": {"window_title": "Win Monitor"},
            }
            for index in range(24)
        ]
        logs.append(
            {
                "timestamp": "2026-03-25T12:12:56",
                "event_type": "modified",
                "app_name": "wpspdf",
                "file_path": "C:/Users/clhcl/Desktop/客户联系方式.pdf",
                "window_info": {"window_title": "发送文件"},
            }
        )
        fallback = {"candidate_events": logs}

        early_score, early_meta = bm._segment_signal_score(early_segment, fallback, logs)
        late_score, late_meta = bm._segment_signal_score(late_segment, fallback, logs)

        self.assertGreater(late_score, early_score)
        self.assertGreater(late_meta["strong_transfer_hits"], early_meta["strong_transfer_hits"])

    def test_remote_vlm_send_dialog_attachment_is_risk_not_confirmed(self) -> None:
        bm = self.benchmark
        sensitive_files = ["C:/Users/clhcl/Desktop/客户联系方式.pdf"]
        actions = [
            {
                "action_id": "case-send-dialog:vlm_action_0",
                "action_type": "attach_file",
                "risk_level": "selected_or_attached",
                "time": "2026-03-25T12:12:49",
                "app": "WPS PDF",
                "app_category": "desktop_app",
                "source_file": "C:/Users/clhcl/Desktop/客户联系方式.pdf",
                "description": (
                    "The sensitive file is visible in the Send File dialog, "
                    "selected or attached for sending, but no final send confirmation is visible."
                ),
                "evidence_source": "remote_vlm",
            }
        ]

        decision = bm._run_datalog_on_audit_actions("case-send-dialog", actions, sensitive_files)

        self.assertTrue(decision["risk_positive"])
        self.assertFalse(decision["confirmed_leak"])
        self.assertFalse(any(fact["relation"] == "LeakFile" for fact in decision["facts"]))


class EvidenceDecisionTest(unittest.TestCase):
    def test_risk_positive_does_not_make_final_positive(self) -> None:
        decision = decide_evidence_outcome(
            datalog_risk_positive=True,
            datalog_confirmed=False,
            log_rule_positive=True,
            log_rule_rules=["file_selected", "upload_staging"],
        )

        self.assertTrue(decision.risk_positive)
        self.assertFalse(decision.confirmed_leak)
        self.assertFalse(decision.final_positive)
        self.assertEqual(decision.final_semantics, "confirmed_leak")

    def test_confirmed_log_rule_makes_final_positive(self) -> None:
        decision = decide_evidence_outcome(
            datalog_risk_positive=False,
            datalog_confirmed=False,
            log_rule_positive=True,
            log_rule_rules=["upload_event"],
        )

        self.assertTrue(decision.risk_positive)
        self.assertTrue(decision.confirmed_leak)
        self.assertTrue(decision.final_positive)
        self.assertEqual(decision.reasoning_source, "log_rule")

    def test_screen_capture_log_rule_makes_final_positive(self) -> None:
        decision = decide_evidence_outcome(
            datalog_risk_positive=False,
            datalog_confirmed=False,
            log_rule_positive=True,
            log_rule_rules=["screen_capture"],
        )

        self.assertTrue(decision.confirmed_leak)
        self.assertTrue(decision.final_positive)


if __name__ == "__main__":
    unittest.main()
