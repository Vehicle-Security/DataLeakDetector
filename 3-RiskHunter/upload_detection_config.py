import json
import os
from pathlib import Path
from typing import Any


def _module_dir() -> Path:
    return Path(__file__).resolve().parent


def _config_dir() -> Path:
    return _module_dir() / "config"


def _default_config_path() -> Path:
    return _config_dir() / "upload_detection.default.json"


def _local_config_path() -> Path:
    return _config_dir() / "upload_detection.local.json"


class UploadDetectionConfig:
    def __init__(self, config_path: str | os.PathLike[str] | None = None):
        self.config_path = self._resolve_config_path(config_path)
        raw_config = self._load_json(self.config_path)
        normalized = self._normalize_config(raw_config)

        self.sensitive_files = normalized["sensitive_files"]
        self.blacklist_apps = normalized["blacklist_apps"]
        self.whitelist_apps = normalized["whitelist_apps"]
        self.detection_rules = normalized["detection_rules"]

    def _resolve_config_path(self, config_path: str | os.PathLike[str] | None) -> Path:
        candidates: list[Path] = []

        if config_path:
            candidates.append(Path(config_path))

        env_config_path = os.getenv("UPLOAD_DETECTION_CONFIG")
        if env_config_path:
            candidates.append(Path(env_config_path))

        candidates.append(_local_config_path())
        candidates.append(_default_config_path())

        for candidate in candidates:
            if candidate.exists():
                return candidate

        raise FileNotFoundError(
            "No upload detection config found. "
            f"Checked: {[str(path) for path in candidates]}"
        )

    def _load_json(self, config_path: Path) -> dict[str, Any]:
        with config_path.open("r", encoding="utf-8") as fh:
            return json.load(fh)

    def _normalize_string_list(self, values: Any) -> list[str]:
        if not isinstance(values, list):
            return []

        normalized: list[str] = []
        seen = set()

        for value in values:
            text = str(value or "").strip()
            if not text or text in seen:
                continue
            seen.add(text)
            normalized.append(text)

        return normalized

    def _normalize_alert_levels(self, values: Any) -> dict[str, str]:
        defaults = {
            "critical": "严重",
            "warning": "警告",
            "info": "信息",
        }

        if not isinstance(values, dict):
            return defaults

        normalized = defaults.copy()
        for key in defaults:
            text = str(values.get(key, defaults[key]) or defaults[key]).strip()
            normalized[key] = text or defaults[key]
        return normalized

    def _normalize_config(self, raw_config: Any) -> dict[str, Any]:
        if not isinstance(raw_config, dict):
            raise ValueError("Upload detection config must be a JSON object")

        detection_rules = raw_config.get("detection_rules", {})
        if not isinstance(detection_rules, dict):
            detection_rules = {}

        return {
            "sensitive_files": self._normalize_string_list(raw_config.get("sensitive_files", [])),
            "blacklist_apps": self._normalize_string_list(raw_config.get("blacklist_apps", [])),
            "whitelist_apps": self._normalize_string_list(raw_config.get("whitelist_apps", [])),
            "detection_rules": {
                "upload_keywords": self._normalize_string_list(
                    detection_rules.get("upload_keywords", [])
                ),
                "upload_operations": self._normalize_string_list(
                    detection_rules.get("upload_operations", [])
                ),
                "alert_levels": self._normalize_alert_levels(
                    detection_rules.get("alert_levels", {})
                ),
            },
        }

    def is_sensitive_file(self, file_path: str) -> bool:
        if not file_path:
            return False

        normalized_target = str(file_path).strip()
        if normalized_target in self.sensitive_files:
            return True

        target_name = os.path.basename(normalized_target)
        for sensitive_file in self.sensitive_files:
            if os.path.basename(sensitive_file) == target_name:
                return True

        return False

    def get_app_category(self, app_name: str) -> str:
        if not app_name:
            return "unknown"

        app_name_lower = app_name.lower()

        for blacklist_app in self.blacklist_apps:
            if blacklist_app.lower() in app_name_lower:
                return "blacklist"

        for whitelist_app in self.whitelist_apps:
            if whitelist_app.lower() in app_name_lower:
                return "whitelist"

        return "unknown"

    def should_alert(self, app_category: str, behavior_category: str) -> tuple[bool, str]:
        if app_category == "whitelist":
            return False, "info"

        if app_category == "blacklist" and "外发" in behavior_category:
            return True, "critical"

        if app_category == "blacklist":
            return True, "warning"

        if app_category == "unknown" and "外发" in behavior_category:
            return False, "info"

        return False, "info"

    def as_dict(self) -> dict[str, Any]:
        return {
            "sensitive_files": list(self.sensitive_files),
            "blacklist_apps": list(self.blacklist_apps),
            "whitelist_apps": list(self.whitelist_apps),
            "detection_rules": {
                "upload_keywords": list(self.detection_rules.get("upload_keywords", [])),
                "upload_operations": list(self.detection_rules.get("upload_operations", [])),
                "alert_levels": dict(self.detection_rules.get("alert_levels", {})),
            },
        }


config = UploadDetectionConfig()


def get_sensitive_files():
    return config.sensitive_files


def get_blacklist_apps():
    return config.blacklist_apps


def get_whitelist_apps():
    return config.whitelist_apps


def is_sensitive_file(file_path: str) -> bool:
    return config.is_sensitive_file(file_path)


def get_app_category(app_name: str) -> str:
    return config.get_app_category(app_name)


def should_alert(app_category: str, behavior_category: str) -> tuple[bool, str]:
    return config.should_alert(app_category, behavior_category)


if __name__ == "__main__":
    print("=" * 80)
    print("Upload detection config")
    print("=" * 80)
    print(f"Config path: {config.config_path}")
    print(f"Sensitive files: {len(config.sensitive_files)}")
    print(f"Blacklist apps: {len(config.blacklist_apps)}")
    print(f"Whitelist apps: {len(config.whitelist_apps)}")
