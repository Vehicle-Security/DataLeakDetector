import json
import os
from pathlib import Path
from typing import Any


def _module_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _config_dir() -> Path:
    return _module_root() / "config"


def _default_config_path() -> Path:
    return _config_dir() / "event_correlation.default.json"


def _local_config_path() -> Path:
    return _config_dir() / "event_correlation.local.json"


class EventCorrelatorConfig:
    def __init__(self, config_path: str | os.PathLike[str] | None = None):
        self.config_path = self._resolve_config_path(config_path)
        raw_config = self._load_json(self.config_path)
        normalized = self._normalize(raw_config)

        self.schema_version = normalized["schema_version"]
        self.time_window_tolerance_seconds = normalized["time_window_tolerance_seconds"]
        self.dedup_bucket_granularity = normalized["dedup_bucket_granularity"]
        self.max_lineage_depth = normalized["max_lineage_depth"]
        self.path_resolution_strategy = normalized["path_resolution_strategy"]
        self.allow_ambiguous_candidates = normalized["allow_ambiguous_candidates"]
        self.merge_evidence_on_dedup = normalized["merge_evidence_on_dedup"]
        self.min_correlation_score = normalized["min_correlation_score"]
        self.upload_operation_keywords = normalized["upload_operation_keywords"]
        self.upload_event_types = normalized["upload_event_types"]

    def _resolve_config_path(self, config_path: str | os.PathLike[str] | None) -> Path:
        candidates: list[Path] = []

        if config_path:
            candidates.append(Path(config_path))

        env_path = os.getenv("EVENT_CORRELATION_CONFIG")
        if env_path:
            candidates.append(Path(env_path))

        candidates.append(_local_config_path())
        candidates.append(_default_config_path())

        for candidate in candidates:
            if candidate.exists():
                return candidate

        raise FileNotFoundError(
            "No EventCorrelator config found. "
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
            raise ValueError("EventCorrelator config must be a JSON object")

        return {
            "schema_version": str(raw_config.get("schema_version", "v2") or "v2"),
            "time_window_tolerance_seconds": int(
                raw_config.get("time_window_tolerance_seconds", 60) or 60
            ),
            "dedup_bucket_granularity": str(
                raw_config.get("dedup_bucket_granularity", "minute") or "minute"
            ),
            "max_lineage_depth": int(raw_config.get("max_lineage_depth", 10) or 10),
            "path_resolution_strategy": str(
                raw_config.get("path_resolution_strategy", "log_first") or "log_first"
            ),
            "allow_ambiguous_candidates": bool(
                raw_config.get("allow_ambiguous_candidates", True)
            ),
            "merge_evidence_on_dedup": bool(
                raw_config.get("merge_evidence_on_dedup", True)
            ),
            "min_correlation_score": float(raw_config.get("min_correlation_score", 0.55) or 0.55),
            "upload_operation_keywords": self._normalize_string_list(
                raw_config.get("upload_operation_keywords", [])
            ),
            "upload_event_types": self._normalize_string_list(
                raw_config.get("upload_event_types", [])
            ),
        }

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "time_window_tolerance_seconds": self.time_window_tolerance_seconds,
            "dedup_bucket_granularity": self.dedup_bucket_granularity,
            "max_lineage_depth": self.max_lineage_depth,
            "path_resolution_strategy": self.path_resolution_strategy,
            "allow_ambiguous_candidates": self.allow_ambiguous_candidates,
            "merge_evidence_on_dedup": self.merge_evidence_on_dedup,
            "min_correlation_score": self.min_correlation_score,
            "upload_operation_keywords": list(self.upload_operation_keywords),
            "upload_event_types": list(self.upload_event_types),
        }
