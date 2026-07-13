"""Correlate logs, visual observations, file lineage, and upload candidates."""

from __future__ import annotations

import re
from dataclasses import replace
from functools import lru_cache
from pathlib import Path
from typing import Any

from ..io import normalize_logs, normalize_path, same_file
from ..models import CorrelatedEvent
from ..policy import SINK_TOKENS, TRANSFER_TOKENS, contains_any
from .candidates import build_upload_candidates
from .classification import behavior_category, original_file_from_metadata, operation_from_text, target_file_from_metadata
from .config import EventCorrelatorConfig
from .facts import build_datalog_facts
from .lineage import Lineage
from .observations import ObservationIndex, normalize_observations
from .output import lineage_payload, operation_record


class EventCorrelator:
    """Correlate logs, visual observations, lineage, and upload candidates."""

    def __init__(self, config: EventCorrelatorConfig | None = None):
        self.config = config or EventCorrelatorConfig()

    def run(self, payload: dict[str, Any]) -> dict[str, Any]:
        config = self.config
        if "non_vlm_enabled" in payload:
            config = replace(config, non_vlm_enabled=bool(payload.get("non_vlm_enabled")))
        session_start_ms = int(payload.get("recording_start_ms") or 0) or None
        logs = normalize_logs([item for item in payload.get("log_events") or [] if isinstance(item, dict)], session_start_ms=session_start_ms)
        observations = normalize_observations(payload.get("frame_segments") or [])
        if not config.non_vlm_enabled:
            observations = [item for item in observations if item.source == "vlm"]
        sensitive_files = self._collect_sensitive_files(payload.get("sensitive_files") or [])
        # Log lineage is binding context for VLM evidence as well as deterministic
        # log evidence, so it must remain available in VLM-only runs.
        lineage = self._build_lineage(logs, sensitive_files)
        self._add_visual_lineage(observations, sensitive_files, lineage)
        correlated = self._correlate(logs, observations, sensitive_files, lineage, config=config)
        uploads = build_upload_candidates(correlated, default_confidence=config.upload_confidence)
        facts = build_datalog_facts(correlated, uploads, lineage)

        return {
            "session_id": str(payload.get("session_id") or payload.get("record_id") or "session"),
            "analysis_status": "success" if correlated or uploads else "no_match",
            "analysis_windows": self._analysis_windows(logs, sensitive_files),
            "correlated_events": [item.to_dict() for item in correlated],
            "operation_records": [operation_record(item) for item in correlated],
            "upload_candidates": [item.to_dict() for item in uploads],
            "file_lineage": lineage_payload(lineage),
            "datalog_facts": [item.to_dict() for item in facts],
            "statistics": {
                "log_events_input": len(logs),
                "frame_segments_input": len(observations),
                "sensitive_files": len(sensitive_files),
                "correlated_events_output": len(correlated),
                "upload_candidates_output": len(uploads),
                "lineage_direct_mappings": len(lineage.direct),
                "datalog_facts_output": len(facts),
                "non_vlm_enabled": config.non_vlm_enabled,
            },
            "errors": [],
        }

    def derived_sensitive_files(self, logs, sensitive_files: list[str]) -> list[str]:
        lineage = self._build_lineage(logs, sensitive_files)
        derived: list[str] = []
        for file_path in lineage.direct:
            if self._resolve_original(file_path, sensitive_files, lineage):
                derived.append(normalize_path(file_path))
        return _dedupe_paths(derived)

    def _collect_sensitive_files(self, explicit: list[Any]) -> list[str]:
        """Normalize the maintained source set; never discover sources at runtime."""

        sensitive: list[str] = []
        for item in explicit:
            path = normalize_path(item)
            if path and not any(same_file(path, existing) for existing in sensitive):
                sensitive.append(path)
        return sensitive

    def _build_lineage(self, logs, sensitive_files: list[str]) -> Lineage:
        lineage = Lineage()
        known = list(sensitive_files)
        known_keys = {normalize_path(item).lower() for item in known if normalize_path(item)}
        known_stems = [_known_stem(item) for item in known]
        known_stems = [item for item in known_stems if item[0]]
        last_sensitive_by_process: dict[str, str] = {}

        for event in sorted(logs, key=lambda item: item.timestamp_ms):
            process_key = (event.process_name or event.app_name or "").lower()
            original = original_file_from_metadata(event.raw)
            target = target_file_from_metadata(event.raw) or event.file_path
            if not _may_contribute_lineage(event, original, target, known_keys, known_stems):
                continue
            text = _event_search_text(event)
            if original and self._resolve_original(original, sensitive_files, lineage):
                lineage.add(target, original)
                _remember_known(target, known, known_keys, known_stems)
            elif target:
                inferred_source = _source_from_derived_filename(target, sensitive_files)
                if inferred_source and _has_derived_transfer_evidence(event, text):
                    lineage.add(target, inferred_source)
                    _remember_known(target, known, known_keys, known_stems)

            resolved = self._resolve_original(event.file_path, sensitive_files, lineage)
            if resolved and process_key:
                last_sensitive_by_process[process_key] = resolved

            if event.file_path and contains_any(text, TRANSFER_TOKENS):
                source = original or last_sensitive_by_process.get(process_key, "") or _guess_source_by_stem_from_index(target, known_stems)
                if source and not self._resolve_original(source, sensitive_files, lineage):
                    source = ""
                if source and (original or _is_generated_descendant_name(Path(normalize_path(target)).stem.lower(), Path(normalize_path(source)).stem.lower())):
                    lineage.add(target, source)
                    _remember_known(target, known, known_keys, known_stems)
        return lineage

    def _correlate(
        self,
        logs,
        observations,
        sensitive_files: list[str],
        lineage: Lineage,
        *,
        config: EventCorrelatorConfig | None = None,
    ) -> list[CorrelatedEvent]:
        config = config or self.config
        correlated: list[CorrelatedEvent] = []
        observation_time_mode = self._observation_time_mode(observations)
        original_cache: dict[str, str] = {}

        for log in sorted(logs, key=lambda item: item.timestamp_ms):
            path_key = normalize_path(log.file_path).lower()
            if path_key not in original_cache:
                original_cache[path_key] = self._resolve_original(log.file_path, sensitive_files, lineage)
            original = original_cache[path_key]
            observation = self._best_observation_for_log(
                log,
                observations,
                observation_time_mode,
                original,
                sensitive_files,
                lineage,
            )

            if not original and observation:
                original = self._resolve_observation_original(observation, sensitive_files, lineage)
            if not config.non_vlm_enabled and observation is None:
                continue
            if config.non_vlm_enabled and observation is None and not _is_standalone_log_evidence(log):
                continue
            if not original:
                continue
            text = " ".join(
                [
                    _event_search_text(log),
                    observation.description if observation else "",
                    observation.operation_type if observation else "",
                    observation.resource if observation else "",
                    " ".join(observation.related_resources) if observation else "",
                ]
            )
            removable_transfer = _is_removable_media_transfer(log, observation, text)
            behavior = "data_exfiltration_candidate" if removable_transfer else behavior_category(text)
            confidence = self.config.upload_confidence if behavior == "data_exfiltration_candidate" else 0.68
            if observation:
                confidence = max(confidence, observation.confidence)
            current_file = target_file_from_metadata(log.raw) or log.file_path or (observation.resource if observation and observation.resource else original)
            if original and current_file and _mentions_file(current_file, original) and not Path(normalize_path(current_file)).suffix:
                current_file = original

            correlated.append(
                CorrelatedEvent(
                    event_id=f"corr_{len(correlated)}",
                    timestamp=log.timestamp,
                    event_type=log.event_type,
                    app_name=(log.app_name or log.process_name or (observation.app_name if observation else "")),
                    original_file=original,
                    current_file=current_file,
                    operation_type="external_sink_interaction" if removable_transfer else _correlated_operation_type(log, observation, text),
                    behavior_category=behavior,
                    confidence=round(min(confidence, 1.0), 3),
                    evidence_refs=tuple(
                        [f"log:{log.event_id}"] + ([f"frame:{observation.observation_id}"] if observation else [])
                    ),
                    join_reasons=tuple(self._join_reasons(log, observation, original, sensitive_files, lineage)),
                )
            )
        correlated.extend(self._correlate_visual_only(observations, sensitive_files, lineage, start_index=len(correlated)))
        return correlated

    def _best_observation_for_log(
        self,
        log,
        observations,
        observation_time_mode: str,
        original: str,
        sensitive_files: list[str],
        lineage: Lineage,
    ):
        log_time = self._log_observation_time(log, observation_time_mode)
        best = None
        best_score = -1
        for observation in observations:
            if not _observation_allowed_for_log(log, observation):
                continue
            distance = abs(log_time - self._observation_center(observation))
            score = self._observation_join_score(log, observation, original, sensitive_files, lineage)
            if not _has_file_binding(log, observation, original, sensitive_files, lineage):
                continue
            if score > best_score or (score == best_score and best is not None and distance < abs(log_time - self._observation_center(best))):
                best = observation
                best_score = score
        return best

    def _observation_join_score(self, log, observation, original: str, sensitive_files: list[str], lineage: Lineage) -> int:
        if observation is None:
            return -1
        text = _observation_search_text(observation)
        score = 0
        resolved = self._resolve_observation_original(observation, sensitive_files, lineage)
        if original and resolved and same_file(original, resolved):
            score += 6
        elif original and _mentions_file(text, original):
            score += 5
        elif log.file_path and _mentions_file(text, log.file_path):
            score += 4
        elif resolved:
            score += 2
        if observation.operation_type == "external_sink_interaction" or contains_any(text, SINK_TOKENS):
            score += 3
        if _is_transfer_observation(observation):
            score += 1
        if observation.source != "log_anchored":
            score += 1
        return score

    def _join_reasons(self, log, observation, original: str, sensitive_files: list[str], lineage: Lineage) -> list[str]:
        reasons = ["log_event"]
        if original:
            reasons.append("sensitive_file_resolved")
        if _is_sink_log(log):
            reasons.append("explicit_sink_log")
        if observation is not None:
            resolved = self._resolve_observation_original(observation, sensitive_files, lineage)
            if resolved and original and same_file(resolved, original):
                reasons.append("visual_mentions_sensitive_file")
            text = _observation_search_text(observation)
            if observation.operation_type == "external_sink_interaction" or contains_any(text, SINK_TOKENS):
                reasons.append("visual_sink_context")
            if _is_transfer_observation(observation):
                reasons.append("visual_transfer_context")
        joined_text = " ".join(
            [
                _event_search_text(log),
                observation.description if observation else "",
                observation.operation_type if observation else "",
                observation.resource if observation else "",
                " ".join(observation.related_resources) if observation else "",
            ]
        )
        if _is_removable_media_transfer(log, observation, joined_text):
            reasons.append("removable_media_sink")
        return reasons

    @staticmethod
    def _observation_center(observation) -> int:
        return observation.start_ms if not observation.end_ms else (observation.start_ms + observation.end_ms) // 2

    def _observation_time_mode(self, observations) -> str:
        # Visual keyframe evidence uses video-relative milliseconds. Some imported
        # or test VLM observations may already contain absolute epoch millis.
        return "absolute" if any(item.start_ms > 10_000_000_000 for item in observations) else "video"

    def _log_observation_time(self, log, mode: str) -> int:
        if mode == "absolute":
            return log.timestamp_ms
        return log.video_time_ms if log.video_time_ms >= 0 else log.timestamp_ms

    def _resolve_original(self, file_path: str, sensitive_files: list[str], lineage: Lineage) -> str:
        if not file_path:
            return ""
        lookup = _sensitive_lookup(tuple(sensitive_files))
        sensitive = _lookup_sensitive_source(file_path, lookup, allow_stem_reference=True)
        if sensitive:
            return sensitive
        root = lineage.root(file_path)
        return _lookup_sensitive_source(root, lookup, allow_stem_reference=False)

    def _resolve_observation_original(self, observation, sensitive_files: list[str], lineage: Lineage) -> str:
        for candidate in [observation.resource, *observation.related_resources]:
            resolved = self._resolve_original(candidate, sensitive_files, lineage)
            if resolved:
                return resolved
        description = observation.description.lower()
        for sensitive in sensitive_files:
            if sensitive and (sensitive.lower() in description or _mentions_file(description, sensitive)):
                return sensitive
        return ""

    def _add_visual_lineage(self, observations, sensitive_files: list[str], lineage: Lineage) -> None:
        for observation in observations:
            if observation.source == "log_anchored":
                continue
            original = self._resolve_visual_original_without_lineage(observation, sensitive_files)
            if not original:
                continue
            text = _observation_search_text(observation)
            if not (_is_transfer_observation(observation) or _is_external_observation(observation) or contains_any(text, SINK_TOKENS)):
                continue
            for candidate in [observation.resource, *observation.related_resources]:
                derived = normalize_path(candidate)
                if _is_visual_derived_candidate(derived, original):
                    lineage.add(derived, original)

    def _resolve_visual_original_without_lineage(self, observation, sensitive_files: list[str]) -> str:
        candidates = [observation.resource, *observation.related_resources, observation.description]
        for candidate in candidates:
            for sensitive in sensitive_files:
                if same_file(candidate, sensitive) or _matches_sensitive_file_reference(candidate, sensitive) or _mentions_file(candidate, sensitive):
                    return sensitive
        return ""

    def _correlate_visual_only(self, observations, sensitive_files: list[str], lineage: Lineage, start_index: int) -> list[CorrelatedEvent]:
        visual_events: list[CorrelatedEvent] = []
        for observation in observations:
            if observation.source == "log_anchored":
                continue
            original = self._resolve_observation_original(observation, sensitive_files, lineage)
            text = f"{observation.description} {observation.operation_type} {observation.resource} {' '.join(observation.related_resources)}"
            if not (contains_any(text, SINK_TOKENS) or _is_transfer_observation(observation)):
                continue
            if not original and not _is_unbound_visual_risk(observation, text):
                continue
            behavior = behavior_category(text) if original else "unknown_risk"
            current_file = self._visual_current_file(observation, original, sensitive_files)
            visual_events.append(
                CorrelatedEvent(
                    event_id=f"corr_{start_index + len(visual_events)}",
                    timestamp="",
                    event_type="visual_observation",
                    app_name=observation.app_name,
                    original_file=original,
                    current_file=current_file,
                    operation_type=observation.operation_type,
                    behavior_category=behavior,
                    confidence=round(min(max(observation.confidence, 0.70), 1.0), 3),
                    evidence_refs=(f"frame:{observation.observation_id}",),
                    join_reasons=tuple(_visual_join_reasons(observation, original)),
                )
            )
        return visual_events

    def _visual_current_file(self, observation, original: str, sensitive_files: list[str]) -> str:
        current_file = observation.resource or original
        if not current_file:
            return original
        for sensitive in sensitive_files:
            if same_file(current_file, sensitive) or _matches_sensitive_file_reference(current_file, sensitive):
                return sensitive
        if original and _mentions_file(observation.description, original) and not _mentions_file(current_file, original):
            return original
        return current_file

    def _analysis_windows(self, logs, sensitive_files: list[str]) -> list[dict[str, Any]]:
        ranges: dict[str, list[int]] = {}
        resolved_cache: dict[str, str] = {}
        for event in logs:
            if not event.timestamp_ms or not event.file_path:
                continue
            path_key = normalize_path(event.file_path).lower()
            if path_key not in resolved_cache:
                resolved_cache[path_key] = self._resolve_original(event.file_path, sensitive_files, Lineage())
            sensitive = resolved_cache[path_key]
            if not sensitive:
                continue
            bounds = ranges.setdefault(sensitive, [event.timestamp_ms, event.timestamp_ms])
            bounds[0] = min(bounds[0], event.timestamp_ms)
            bounds[1] = max(bounds[1], event.timestamp_ms)
        return [
            {"sensitive_file": sensitive, "start_ms": bounds[0], "end_ms": bounds[1]}
            for sensitive, bounds in ranges.items()
        ]


def _event_search_text(event) -> str:
    raw = event.raw
    parts: list[str] = [
        event.event_type,
        event.file_path,
        event.process_name,
        event.app_name,
        event.window_title,
        event.description,
    ]
    for key in ("file_name", "file_extension", "content_preview", "operation", "description", "type", "path", "destination_path"):
        value = raw.get(key)
        if value is not None:
            parts.append(str(value))
    for key in ("extra", "upload_detection", "process_info", "window_info"):
        parts.extend(_flatten_search_parts(raw.get(key)))
    return " ".join(item.strip() for item in parts if item and item.strip())


def _observation_search_text(observation) -> str:
    return " ".join(
        item
        for item in (
            observation.operation_type,
            observation.app_name,
            observation.resource,
            " ".join(observation.related_resources),
            observation.description,
            observation.source,
        )
        if item
    )


def _is_external_observation(observation) -> bool:
    text = _observation_search_text(observation)
    return observation.operation_type == "external_sink_interaction" or contains_any(text, SINK_TOKENS)


def _has_file_binding(log, observation, original: str, sensitive_files: list[str], lineage: Lineage) -> bool:
    text = _observation_search_text(observation)
    resolved = ""
    for candidate in [observation.resource, *observation.related_resources]:
        for sensitive in sensitive_files:
            if same_file(candidate, sensitive) or _matches_sensitive_file_reference(candidate, sensitive):
                resolved = sensitive
                break
        if resolved:
            break
        root = lineage.root(candidate)
        for sensitive in sensitive_files:
            if same_file(root, sensitive):
                resolved = sensitive
                break
        if resolved:
            break
    if original and resolved and same_file(original, resolved):
        return True
    if original and _mentions_file(text, original):
        return True
    return bool(log.file_path and _mentions_file(text, log.file_path))


def _is_transfer_observation(observation) -> bool:
    text = _observation_search_text(observation)
    return observation.operation_type == "file_or_content_transfer" or contains_any(text, TRANSFER_TOKENS)


def _observation_allowed_for_log(log, observation) -> bool:
    if observation.source == "log_anchored":
        return False
    return not _is_external_observation(observation) or _is_sink_log(log)


def _is_sink_log(log) -> bool:
    if log.event_type in {"file_selected", "file_upload", "upload", "uploaded", "upload_complete", "send_click"}:
        return True
    if (log.process_name or log.app_name or "").lower() == "fsquirt.exe":
        return True
    extra = log.raw.get("extra") if isinstance(log.raw.get("extra"), dict) else {}
    raw_operation = str(log.raw.get("operation") or extra.get("raw_operation") or "")
    category = str(extra.get("category") or "")
    return raw_operation in {"file_selected", "file_upload", "upload", "send_click"} or contains_any(category, ("文件上传", "直接外发"))


def _is_standalone_log_evidence(log) -> bool:
    """Keep deterministic facts focused on actions instead of file-system noise."""

    event_type = log.event_type.lower()
    if _is_sink_log(log) or event_type in {
        "copy",
        "copied",
        "file_copied",
        "move",
        "moved",
        "file_moved",
        "rename",
        "renamed",
        "file_renamed",
        "paste",
        "clipboard_write",
        "clipboard_read",
        "clipboard_text",
        "clipboard_image",
        "export",
        "print",
        "compress",
        "archive",
        "screenshot",
        "screen_record",
        "screen_share",
    }:
        return True
    if original_file_from_metadata(log.raw):
        return True
    if event_type not in {"created", "modified", "file_created", "file_modified"}:
        return False
    text = _event_search_text(log)
    return contains_any(text, TRANSFER_TOKENS + SINK_TOKENS) or _is_removable_media_context(text)


def _is_removable_media_transfer(log, observation, text: str) -> bool:
    combined = " ".join(
        [
            text,
            log.file_path,
            log.description,
            log.window_title,
            observation.resource if observation else "",
            " ".join(observation.related_resources) if observation else "",
            observation.description if observation else "",
        ]
    )
    if not _is_removable_media_context(combined):
        return False
    return contains_any(combined, TRANSFER_TOKENS + SINK_TOKENS) or log.event_type in {
        "created",
        "modified",
        "renamed",
        "copied",
        "copy",
        "file_created",
        "file_moved",
        "file_copied",
    }


def _is_removable_media_context(text: str) -> bool:
    normalized = normalize_path(text).lower()
    removable_terms = (
        "usb",
        "removable",
        "removable media",
        "removable drive",
        "flash drive",
        "thumb drive",
        "u disk",
        "udisk",
        "external drive",
        "可移动",
        "可移动存储",
        "可移动磁盘",
        "移动磁盘",
        "移动硬盘",
        "u盘",
        "u 盘",
    )
    return contains_any(normalized, removable_terms)


def _has_derived_transfer_evidence(event, text: str) -> bool:
    file_change_events = {
        "created",
        "modified",
        "renamed",
        "moved",
        "copied",
        "file_created",
        "file_modified",
        "file_renamed",
        "file_moved",
        "file_copied",
    }
    if event.event_type.lower() not in file_change_events:
        return False
    context = f"{text} {event.window_title} {event.description}"
    return _is_removable_media_context(context) or contains_any(context, TRANSFER_TOKENS)


def _source_from_derived_filename(file_path: str, sensitive_files: list[str]) -> str:
    candidate_stem = Path(normalize_path(file_path)).stem.lower()
    if not candidate_stem:
        return ""
    matches = [
        sensitive
        for sensitive in sensitive_files
        if _is_generated_descendant_name(candidate_stem, Path(normalize_path(sensitive)).stem.lower())
    ]
    return matches[0] if len(matches) == 1 else ""


def _is_generated_descendant_name(candidate_stem: str, source_stem: str) -> bool:
    if not source_stem or candidate_stem == source_stem:
        return False
    return candidate_stem.startswith(f"{source_stem}_") or candidate_stem.startswith(f"{source_stem} (")


def _correlated_operation_type(log, observation, text: str) -> str:
    if _is_sink_log(log):
        return "external_sink_interaction"
    if observation is not None and observation.operation_type:
        return observation.operation_type
    return operation_from_text(text, log.event_type)


def _visual_join_reasons(observation, original: str) -> list[str]:
    reasons = ["visual_only"]
    if original:
        reasons.append("visual_mentions_sensitive_file")
    text = _observation_search_text(observation)
    if observation.operation_type == "external_sink_interaction" or contains_any(text, SINK_TOKENS):
        reasons.append("visual_sink_context")
    if _is_transfer_observation(observation):
        reasons.append("visual_transfer_context")
    return reasons


def _mentions_file(text: str, file_path: str) -> bool:
    normalized_text = normalize_path(text).lower()
    normalized_file = normalize_path(file_path).lower()
    if not normalized_text or not normalized_file:
        return False
    name = Path(normalized_file).name.lower()
    stem = Path(name).stem.lower()
    return normalized_file in normalized_text or (name and name in normalized_text) or (len(stem) >= 4 and stem in normalized_text)


def _matches_sensitive_file_reference(file_path: str, sensitive_file: str) -> bool:
    normalized_ref = normalize_path(file_path).lower()
    normalized_sensitive = normalize_path(sensitive_file).lower()
    if not normalized_ref or not normalized_sensitive:
        return False
    ref_name = Path(normalized_ref).name.lower()
    sensitive_name = Path(normalized_sensitive).name.lower()
    sensitive_stem = Path(sensitive_name).stem.lower()
    if not ref_name or not sensitive_name or not sensitive_stem:
        return False
    if Path(ref_name).suffix:
        return False
    return len(ref_name) >= 4 and ref_name == sensitive_stem


def _is_unbound_visual_risk(observation, text: str) -> bool:
    normalized = text.lower()
    return (
        observation.operation_type in {"external_sink_interaction", "file_or_content_transfer"}
        and any(marker in normalized for marker in ("direct_leak", "hidden_transfer", "unknown_risk"))
    )


def _is_visual_derived_candidate(file_path: str, original: str) -> bool:
    normalized = normalize_path(file_path).strip().strip("\"'")
    lowered = normalized.lower()
    if not lowered or lowered in {"unknown", "未知", "n/a", "na", "none", "null", "-"}:
        return False
    if same_file(normalized, original) or _matches_sensitive_file_reference(normalized, original):
        return False
    return True


def _flatten_search_parts(value: Any) -> list[str]:
    if isinstance(value, dict):
        parts: list[str] = []
        for item in value.values():
            parts.extend(_flatten_search_parts(item))
        return parts
    if isinstance(value, list | tuple):
        parts = []
        for item in value:
            parts.extend(_flatten_search_parts(item))
        return parts
    text = str(value or "").strip()
    return [text] if text else []


def _remember_known(path: str, known: list[str], known_keys: set[str], known_stems: list[tuple[str, str]]) -> None:
    normalized = normalize_path(path)
    key = normalized.lower()
    if not key or key in known_keys:
        return
    known.append(normalized)
    known_keys.add(key)
    stem = _known_stem(normalized)
    if stem[0]:
        known_stems.append(stem)


def _known_stem(path: str) -> tuple[str, str]:
    normalized = normalize_path(path)
    stem = Path(normalized).stem.lower()
    return stem, normalized


def _may_contribute_lineage(
    event,
    original: str,
    target: str,
    known_keys: set[str],
    known_stems: list[tuple[str, str]],
) -> bool:
    """Cheaply reject file-system noise before expanding raw event metadata."""

    if original:
        return True
    normalized_target = normalize_path(target).lower()
    if normalized_target in known_keys:
        return True

    candidate_stem = Path(normalized_target).stem.lower()
    if candidate_stem and any(_is_generated_descendant_name(candidate_stem, stem) for stem, _ in known_stems):
        return True

    event_type = event.event_type.lower()
    if event_type in {"app_switch", "window_changed", "window_closed", "file_selected", "file_upload", "upload", "uploaded", "upload_complete"}:
        return True

    extra = event.raw.get("extra") if isinstance(event.raw.get("extra"), dict) else {}
    raw_operation = str(event.raw.get("operation") or extra.get("raw_operation") or "").lower()
    return any(token in raw_operation for token in ("copy", "paste", "export", "print", "save_as", "compress", "upload", "transfer"))


def _guess_source_by_stem_from_index(file_path: str, known_stems: list[tuple[str, str]]) -> str:
    stem = Path(normalize_path(file_path)).stem.lower()
    if not stem:
        return ""
    for known_stem, known_path in known_stems:
        if known_stem and (stem.startswith(known_stem) or known_stem.startswith(stem)):
            return known_path
    return ""


def _dedupe_paths(paths: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for path in paths:
        normalized = normalize_path(path)
        key = normalized.lower()
        if not key or key in seen:
            continue
        seen.add(key)
        result.append(normalized)
    return result


@lru_cache(maxsize=128)
def _sensitive_lookup(sensitive_files: tuple[str, ...]) -> tuple[dict[str, str], dict[str, str], dict[str, str]]:
    exact: dict[str, str] = {}
    basenames: dict[str, str] = {}
    stems: dict[str, str] = {}
    for sensitive in sensitive_files:
        normalized = normalize_path(sensitive)
        lowered = normalized.lower()
        if not lowered or _is_placeholder_sensitive_ref(lowered):
            continue
        exact.setdefault(lowered, sensitive)
        name = Path(lowered).name
        path = Path(name)
        if path.suffix and len(path.stem) >= 2:
            basename_key = re.sub(r"\s+(\.[^.]+)$", r"\1", name)
            basenames.setdefault(basename_key, sensitive)
        if path.stem:
            stems.setdefault(path.stem, sensitive)
    return exact, basenames, stems


def _lookup_sensitive_source(
    file_path: str,
    lookup: tuple[dict[str, str], dict[str, str], dict[str, str]],
    *,
    allow_stem_reference: bool,
) -> str:
    normalized = normalize_path(file_path).lower()
    if not normalized or _is_placeholder_sensitive_ref(normalized):
        return ""
    exact, basenames, stems = lookup
    matched = exact.get(normalized)
    if matched:
        return matched
    name = Path(normalized).name
    path = Path(name)
    if path.suffix and len(path.stem) >= 2:
        basename_key = re.sub(r"\s+(\.[^.]+)$", r"\1", name)
        return basenames.get(basename_key, "")
    if allow_stem_reference and len(name) >= 4:
        return stems.get(name, "")
    return ""


def _is_placeholder_sensitive_ref(value: str) -> bool:
    normalized = value.strip().strip("\"'").lower()
    return normalized in {"n/a", "na", "none", "null", "unknown", "-"} or normalized.startswith("n/a ")


