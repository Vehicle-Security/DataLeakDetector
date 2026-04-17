import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_ROOT = REPO_ROOT / "03-LeakReasoner"

if str(MODULE_ROOT) not in os.sys.path:
    os.sys.path.insert(0, str(MODULE_ROOT))


def load_module(module_name: str):
    spec = importlib.util.find_spec("leak_reasoner.config")
    if spec is None or spec.loader is None:
        raise ImportError("Unable to load leak_reasoner.config")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class UploadConfigTests(unittest.TestCase):
    def setUp(self):
        self.original_env = os.environ.get("LEAK_REASONER_CONFIG")

    def tearDown(self):
        if self.original_env is None:
            os.environ.pop("LEAK_REASONER_CONFIG", None)
        else:
            os.environ["LEAK_REASONER_CONFIG"] = self.original_env

    def test_default_config_loads(self):
        module = load_module("leak_reasoner_config_default_test")
        config = module.LeakReasonerConfig()

        self.assertEqual(config.high_risk_score, 80)
        self.assertIn("mail_attachment", config.high_risk_sink_types)
        self.assertIn("internal_mail", config.trusted_sink_types)

    def test_env_override_config_loads(self):
        payload = {
            "schema_version": "v2-test",
            "high_risk_score": 90,
            "medium_risk_score": 60,
            "min_confidence_for_case": 0.7,
            "trusted_sink_types": ["internal_chat"],
            "high_risk_sink_types": ["screen_share"],
        }

        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False)
            temp_path = fh.name

        try:
            os.environ["LEAK_REASONER_CONFIG"] = temp_path
            module = load_module("leak_reasoner_config_override_test")
            config = module.LeakReasonerConfig()
            self.assertEqual(config.schema_version, "v2-test")
            self.assertEqual(config.high_risk_score, 90)
            self.assertEqual(config.medium_risk_score, 60)
            self.assertEqual(config.min_confidence_for_case, 0.7)
            self.assertEqual(config.trusted_sink_types, ["internal_chat"])
            self.assertEqual(config.high_risk_sink_types, ["screen_share"])
        finally:
            os.unlink(temp_path)

    def test_config_ranges_are_reasonable(self):
        module = load_module("leak_reasoner_config_policy_test")
        config = module.LeakReasonerConfig()

        self.assertGreater(config.high_risk_score, config.medium_risk_score)
        self.assertGreaterEqual(config.min_confidence_for_case, 0.0)
        self.assertLessEqual(config.min_confidence_for_case, 1.0)


if __name__ == "__main__":
    unittest.main()
