from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_PATH = REPO_ROOT / "tools" / "benchmark_nas_samples.py"


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


if __name__ == "__main__":
    unittest.main()
