from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any

from .adapter import FrameAnalyzerAdapter, UNKNOWN_RESOURCE_MARKERS, adapt_legacy_frame_result


USER_PATH_PATTERN = re.compile(r"([A-Za-z]:[\\/]+Users[\\/]+)([^\\/]+)")


def _sanitize_export_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _sanitize_export_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_sanitize_export_value(item) for item in value]
    if isinstance(value, str):
        return USER_PATH_PATTERN.sub(r"\1<redacted>", value)
    return value


@dataclass
class FrameAnalyzerRequest:
    video_path: str
    recording_start_time: str
    search_start_time: str
    search_end_time: str
    target_keywords: list[str]
    force_refresh: bool = False


class FrameAnalyzerService:
    def __init__(self):
        self.adapter = FrameAnalyzerAdapter()
        self.cache_dir = Path(os.environ.get("FRAME_ANALYZER_CACHE_DIR", "output/frame_cache"))
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.backend_name = "legacy_adapter"
        self.cache_schema_version = "v3"
        self.backend_version = "legacy_adapter_v1"
        self.prompt_signature = "legacy_prompt_bundle_v1"

    def analyze(self, request: FrameAnalyzerRequest) -> dict:
        cache_path = self._cache_path(request)
        if cache_path.exists() and not request.force_refresh:
            with cache_path.open("r", encoding="utf-8") as fh:
                cached = json.load(fh)
            normalized_cached = self._normalize_cached_result(cached, request=request, cache_path=cache_path)
            export_cached = _sanitize_export_value(normalized_cached)
            if export_cached != cached:
                with cache_path.open("w", encoding="utf-8") as fh:
                    json.dump(export_cached, fh, ensure_ascii=False, indent=2)
            return normalized_cached

        result = self.adapter.analyze_with_legacy_backend(
            rec_start_time_str=request.recording_start_time,
            search_start_time_str=request.search_start_time,
            search_end_time_str=request.search_end_time,
            target_keywords=request.target_keywords,
            video_path=request.video_path,
        )
        if "segments" in result and "status" in result:
            normalized = result
        else:
            normalized = self.adapt_legacy_result(result)

        normalized = self._attach_analysis_provenance(
            normalized,
            request=request,
            cache_path=cache_path,
            cache_hit=False,
        )

        with cache_path.open("w", encoding="utf-8") as fh:
            json.dump(_sanitize_export_value(normalized), fh, ensure_ascii=False, indent=2)

        return normalized

    def adapt_legacy_result(self, legacy_result: dict) -> dict:
        return adapt_legacy_frame_result(legacy_result)

    def _normalize_cached_result(
        self,
        cached_result: dict,
        *,
        request: FrameAnalyzerRequest,
        cache_path: Path,
    ) -> dict:
        if not isinstance(cached_result, dict):
            return cached_result

        segments = []
        for index, segment in enumerate(list(cached_result.get("segments", []) or [])):
            if not isinstance(segment, dict):
                continue
            normalized_segment = dict(segment)
            segment_id = str(normalized_segment.get("segment_id", f"segment_{index}") or f"segment_{index}")
            if segment_id.startswith("legacy_segment_"):
                segment_id = f"segment_{segment_id.removeprefix('legacy_segment_')}"
            normalized_segment["segment_id"] = segment_id

            primary_resource = str(normalized_segment.get("primary_resource", "") or "").strip()
            if primary_resource in UNKNOWN_RESOURCE_MARKERS:
                normalized_segment["primary_resource"] = ""

            related_resources = []
            for item in list(normalized_segment.get("related_resources", []) or []):
                text = str(item or "").strip()
                if text and text not in UNKNOWN_RESOURCE_MARKERS:
                    related_resources.append(text)
            normalized_segment["related_resources"] = related_resources

            visible_evidence = []
            for item in list(normalized_segment.get("visible_evidence", []) or []):
                text = str(item or "").strip()
                if text and text not in UNKNOWN_RESOURCE_MARKERS:
                    visible_evidence.append(text)
            normalized_segment["visible_evidence"] = visible_evidence
            normalized_segment.setdefault("analysis_backend", "legacy_adapter")
            segments.append(normalized_segment)

        normalized = dict(cached_result)
        normalized["segments"] = segments
        analysis_metadata = dict(normalized.get("analysis_metadata", {}) or {})
        legacy_force_refresh = analysis_metadata.pop("force_refresh", None)
        analysis_metadata.setdefault("analysis_backend", self.backend_name)
        analysis_metadata.setdefault("analysis_backend_version", self.backend_version)
        analysis_metadata.setdefault("prompt_signature", self.prompt_signature)
        analysis_metadata.setdefault("cache_schema_version", self.cache_schema_version)
        analysis_metadata["cache_hit"] = True
        analysis_metadata["request_signature"] = self._request_digest(request)
        analysis_metadata["cache_path"] = str(cache_path)
        analysis_metadata["fresh_run_requested"] = bool(request.force_refresh)
        analysis_metadata.setdefault("video_path", request.video_path)
        analysis_metadata.setdefault(
            "search_window",
            {
                "recording_start_time": request.recording_start_time,
                "search_start_time": request.search_start_time,
                "search_end_time": request.search_end_time,
            },
        )
        analysis_metadata.setdefault("target_keywords", list(request.target_keywords))
        if legacy_force_refresh is not None and not request.force_refresh:
            analysis_metadata["cached_result_fresh_run_requested"] = bool(legacy_force_refresh)
        normalized["analysis_metadata"] = analysis_metadata
        normalized.setdefault(
            "summary",
            {
                "apps": sorted({segment.get("app_name", "") for segment in segments if segment.get("app_name", "")}),
                "operations": sorted(
                    {segment.get("operation_type", "") for segment in segments if segment.get("operation_type", "")}
                ),
                "resources": sorted(
                    {
                        item
                        for segment in segments
                        for item in [segment.get("primary_resource", ""), *(segment.get("related_resources", []) or [])]
                        if item
                    }
                ),
            },
        )
        return normalized

    def _cache_path(self, request: FrameAnalyzerRequest) -> Path:
        digest = self._request_digest(request)
        return self.cache_dir / f"{digest}.json"

    def _request_digest(self, request: FrameAnalyzerRequest) -> str:
        key = "|".join(
            [
                request.video_path,
                request.recording_start_time,
                request.search_start_time,
                request.search_end_time,
                ",".join(request.target_keywords),
                self.backend_name,
                self.backend_version,
                self.prompt_signature,
                self.cache_schema_version,
            ]
        )
        return hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]

    def _attach_analysis_provenance(
        self,
        result: dict[str, Any],
        request: FrameAnalyzerRequest,
        cache_path: Path,
        *,
        cache_hit: bool,
    ) -> dict[str, Any]:
        normalized = dict(result)
        analysis_metadata = dict(normalized.get("analysis_metadata", {}) or {})
        analysis_metadata.update(
            {
                "analysis_backend": self.backend_name,
                "analysis_backend_version": self.backend_version,
                "prompt_signature": self.prompt_signature,
                "cache_hit": cache_hit,
                "cache_schema_version": self.cache_schema_version,
                "request_signature": self._request_digest(request),
                "cache_path": str(cache_path),
                "fresh_run_requested": bool(request.force_refresh),
                "video_path": request.video_path,
                "search_window": {
                    "recording_start_time": request.recording_start_time,
                    "search_start_time": request.search_start_time,
                    "search_end_time": request.search_end_time,
                },
                "target_keywords": list(request.target_keywords),
            }
        )
        normalized["analysis_metadata"] = analysis_metadata
        normalized.setdefault("summary", {"apps": [], "operations": [], "resources": []})
        return normalized
