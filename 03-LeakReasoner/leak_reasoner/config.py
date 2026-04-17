import json
import os
from pathlib import Path
from typing import Any


def _module_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _config_dir() -> Path:
    return _module_root() / "config"


def _default_config_path() -> Path:
    return _config_dir() / "leak_reasoner.default.json"


def _local_config_path() -> Path:
    return _config_dir() / "leak_reasoner.local.json"


class LeakReasonerConfig:
    def __init__(self, config_path: str | os.PathLike[str] | None = None):
        self.config_path = self._resolve_config_path(config_path)
        raw_config = self._load_json(self.config_path)
        normalized = self._normalize(raw_config)

        self.schema_version = normalized["schema_version"]
        self.high_risk_score = normalized["high_risk_score"]
        self.medium_risk_score = normalized["medium_risk_score"]
        self.min_confidence_for_case = normalized["min_confidence_for_case"]
        self.trusted_sink_types = normalized["trusted_sink_types"]
        self.high_risk_sink_types = normalized["high_risk_sink_types"]

    def _resolve_config_path(self, config_path: str | os.PathLike[str] | None) -> Path:
        candidates: list[Path] = []

        if config_path:
            candidates.append(Path(config_path))

        env_path = os.getenv("LEAK_REASONER_CONFIG")
        if env_path:
            candidates.append(Path(env_path))

        candidates.append(_local_config_path())
        candidates.append(_default_config_path())

        for candidate in candidates:
            if candidate.exists():
                return candidate

        raise FileNotFoundError(
            "No LeakReasoner config found. "
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

    def _normalize(self, raw_config: Any) -> dict[str, Any]:
        if not isinstance(raw_config, dict):
            raise ValueError("LeakReasoner config must be a JSON object")

        return {
            "schema_version": str(raw_config.get("schema_version", "v2") or "v2"),
            "high_risk_score": int(raw_config.get("high_risk_score", 80) or 80),
            "medium_risk_score": int(raw_config.get("medium_risk_score", 50) or 50),
            "min_confidence_for_case": float(raw_config.get("min_confidence_for_case", 0.4) or 0.4),
            "trusted_sink_types": self._normalize_string_list(raw_config.get("trusted_sink_types", [])),
            "high_risk_sink_types": self._normalize_string_list(raw_config.get("high_risk_sink_types", [])),
        }
