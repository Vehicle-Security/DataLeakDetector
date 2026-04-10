import importlib.util
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_ROOT / "3-RiskHunter" / "upload_detection_config.py"


def load_module(module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class UploadConfigTests(unittest.TestCase):
    def setUp(self):
        self.original_env = os.environ.get("UPLOAD_DETECTION_CONFIG")

    def tearDown(self):
        if self.original_env is None:
            os.environ.pop("UPLOAD_DETECTION_CONFIG", None)
        else:
            os.environ["UPLOAD_DETECTION_CONFIG"] = self.original_env

    def test_default_config_loads(self):
        module = load_module("upload_detection_config_default_test")
        config = module.UploadDetectionConfig()

        self.assertGreater(len(config.sensitive_files), 0)
        self.assertIn("QQ邮箱", config.blacklist_apps)
        self.assertIn("WPS", config.whitelist_apps)

    def test_env_override_config_loads(self):
        payload = {
            "sensitive_files": ["C:/demo/secret.txt"],
            "blacklist_apps": ["BadApp"],
            "whitelist_apps": ["GoodApp"],
            "detection_rules": {
                "upload_keywords": ["上传"],
                "upload_operations": ["网页上传"],
                "alert_levels": {
                    "critical": "严重",
                    "warning": "警告",
                    "info": "信息",
                },
            },
        }

        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False)
            temp_path = fh.name

        try:
            os.environ["UPLOAD_DETECTION_CONFIG"] = temp_path
            module = load_module("upload_detection_config_override_test")
            config = module.UploadDetectionConfig()
            self.assertEqual(config.sensitive_files, ["C:/demo/secret.txt"])
            self.assertEqual(config.blacklist_apps, ["BadApp"])
            self.assertEqual(config.whitelist_apps, ["GoodApp"])
        finally:
            os.unlink(temp_path)

    def test_category_and_alert_policy_remain_compatible(self):
        module = load_module("upload_detection_config_policy_test")
        config = module.UploadDetectionConfig()

        self.assertEqual(config.get_app_category("QQ邮箱"), "blacklist")
        self.assertEqual(config.get_app_category("WPS"), "whitelist")
        self.assertEqual(config.should_alert("blacklist", "直接外发"), (True, "critical"))
        self.assertEqual(config.should_alert("whitelist", "直接外发"), (False, "info"))


if __name__ == "__main__":
    unittest.main()
