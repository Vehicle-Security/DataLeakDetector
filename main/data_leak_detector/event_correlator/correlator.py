"""Correlate logs, visual observations, file lineage, and upload candidates."""

from __future__ import annotations

import re
from dataclasses import replace
from functools import lru_cache
from pathlib import Path
from typing import Any

from ..frame_analyzer.apps import identify_frontend_app
from ..io import normalize_logs, normalize_path, parse_timestamp_ms, same_file
from ..models import CorrelatedEvent
from ..policy import RISKY_APP_CATEGORIES, SINK_TOKENS, TRANSFER_TOKENS, contains_any
from .candidates import build_upload_candidates
from .classification import behavior_category, original_file_from_metadata, operation_from_text, target_file_from_metadata
from .config import EventCorrelatorConfig
from .facts import build_datalog_facts
from .lineage import Lineage
from .observations import ObservationIndex, normalize_observations
from .output import landing_locations, lineage_payload, operation_record


_CLIPBOARD_SOURCE_HORIZON_MS = 120_000
_CLIPBOARD_TARGET_HORIZON_MS = 15_000
_SCREENSHOT_CLIPBOARD_CONTEXT_HORIZON_MS = 30_000
_SCREENSHOT_CLIPBOARD_TARGET_HORIZON_MS = 60_000
_DOCUMENT_TITLE_SOURCE_HORIZON_MS = 300_000
_CLIPBOARD_TARGET_EVENTS = {"created", "modified", "file_created", "file_modified"}
_CLIPBOARD_DOCUMENT_EXTENSIONS = {
    ".bmp",
    ".csv",
    ".doc",
    ".docx",
    ".jpeg",
    ".jpg",
    ".m4a",
    ".pdf",
    ".png",
    ".ppt",
    ".pptx",
    ".rar",
    ".sql",
    ".txt",
    ".xls",
    ".xlsx",
    ".zip",
}
_CLIPBOARD_HIDDEN_PATH_MARKERS = (
    "/appdata/",
    "/cache/",
    "/cachedata/",
    "/program files/",
    "/programdata/",
    "/temp/",
    "/windows/",
)
_DECLARED_VISUAL_OUTBOUND_ACTIONS = {
    "attach_file",
    "ai_chat_upload",
    "ai_prompt_input",
    "ai_prompt_paste",
    "chat_paste",
    "chat_send",
    "cloud_sync",
    "cloud_upload",
    "commit",
    "copy_paste_to_ai",
    "copy_to_removable_media",
    "document_translation_upload",
    "email_send",
    "file_send",
    "file_upload",
    "http_post",
    "network_upload",
    "article_publish",
    "paste_exfiltration",
    "paste_to_ai",
    "paste_to_web",
    "publish",
    "post_question",
    "screen_share",
    "send",
    "send_click",
    "share_screen",
    "upload",
    "upload_complete",
    "upload_file_to_ai",
    "screenshot_paste_to_chat",
    "screenshot_to_chat",
    "web_form_composition",
    "web_upload",
    "folder_sync",
}
_OUTBOUND_SOURCE_OBJECT_ACTIONS = {
    "ai_chat_upload",
    "ai_prompt_input",
    "ai_prompt_paste",
    "chat_paste",
    "chat_send",
    "chat_upload",
    "copy_paste_to_ai",
    "email_send",
    "file_send",
    "file_upload",
    "paste_exfiltration",
    "paste_to_ai",
    "paste_to_web",
    "upload",
    "upload_file_to_ai",
    "web_upload",
}


class EventCorrelator:
    """Correlate logs, visual observations, lineage, and upload candidates."""

    def __init__(self, config: EventCorrelatorConfig | None = None):
        self.config = config or EventCorrelatorConfig()
        self._lineage_artifact_times: dict[str, int] = {}

    def run(self, payload: dict[str, Any]) -> dict[str, Any]:
        config = self.config
        if "non_vlm_enabled" in payload:
            config = replace(config, non_vlm_enabled=bool(payload.get("non_vlm_enabled")))
        session_start_ms = int(payload.get("recording_start_ms") or 0) or None
        logs = normalize_logs([item for item in payload.get("log_events") or [] if isinstance(item, dict)], session_start_ms=session_start_ms)
        visual_recording_start_ms = int(session_start_ms or 0) or next(
            (
                event.timestamp_ms - event.video_time_ms
                for event in logs
                if event.timestamp_ms and event.video_time_ms > 0
            ),
            0,
        )
        observations = normalize_observations(payload.get("frame_segments") or [])
        observations = [_normalize_cached_observation_semantics(item) for item in observations]
        observations = _suppress_conflicting_screen_share_observations(observations)
        if not config.non_vlm_enabled:
            observations = [item for item in observations if item.source == "vlm"]
        sensitive_files = self._collect_sensitive_files(payload.get("sensitive_files") or [])
        # Log lineage is binding context for VLM evidence as well as deterministic
        # log evidence, so it must remain available in VLM-only runs.
        lineage = self._build_lineage(logs, sensitive_files)
        self._add_visual_lineage(observations, sensitive_files, lineage, logs=logs)
        observations = self._fuse_visual_evidence(
            logs,
            observations,
            sensitive_files,
            lineage,
            horizon_ms=config.visual_evidence_horizon_ms,
        )
        self._add_visual_lineage(observations, sensitive_files, lineage, logs=logs)
        self._bind_log_lineage_aliases(logs, sensitive_files, lineage)
        correlated = self._correlate(
            logs,
            observations,
            sensitive_files,
            lineage,
            config=config,
            recording_start_ms=visual_recording_start_ms,
        )
        correlated = _add_confirmed_git_push_events(correlated, logs)
        correlated = _suppress_redundant_cloud_sync_source_events(correlated, lineage)
        uploads = build_upload_candidates(correlated, default_confidence=config.upload_confidence)
        case_id = str(payload.get("case_id") or "case")
        facts = build_datalog_facts(correlated, uploads, lineage, case_id=case_id)

        return {
            "case_id": case_id,
            "session_id": str(payload.get("session_id") or payload.get("record_id") or "session"),
            "analysis_status": "success" if correlated or uploads else "no_match",
            "analysis_windows": self._analysis_windows(logs, sensitive_files),
            "correlated_events": [item.to_dict() for item in correlated],
            "operation_records": [operation_record(item) for item in correlated],
            "upload_candidates": [item.to_dict() for item in uploads],
            "file_lineage": lineage_payload(lineage),
            "landing_locations": landing_locations(lineage, correlated),
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
        seen: set[str] = set()
        for item in explicit:
            path = normalize_path(item)
            # The maintained catalog may contain same-named files from different
            # cases/users. Basename fallback is useful for a VLM reference, but
            # must not discard a distinct absolute source path here.
            key = path.lower()
            if path and key not in seen:
                sensitive.append(path)
                seen.add(key)
        return sensitive

    def _build_lineage(self, logs, sensitive_files: list[str]) -> Lineage:
        lineage = Lineage()
        self._lineage_artifact_times = {}
        known = list(sensitive_files)
        known_keys = {normalize_path(item).lower() for item in known if normalize_path(item)}
        known_stems = [_known_stem(item) for item in known]
        known_stems = [item for item in known_stems if item[0]]
        last_sensitive_by_process: dict[str, str] = {}
        recent_artifact_by_process: dict[str, tuple[str, int]] = {}
        recent_sensitive_contexts: list[tuple[str, int, str, str]] = []
        recent_document_titles: list[tuple[int, str]] = []
        active_clipboard: tuple[str, int, str, str, bool] | None = None

        for event in sorted(logs, key=lambda item: item.timestamp_ms):
            process_key = (event.process_name or event.app_name or "").lower()
            process_family = _lineage_process_family(event)
            original = original_file_from_metadata(event.raw)
            target = target_file_from_metadata(event.raw) or event.file_path
            event_time_ms = _lineage_event_time_ms(event)
            if event.window_title and any(
                _mentions_exact_filename(event.window_title, sensitive)
                for sensitive in sensitive_files
            ):
                recent_document_titles.append((event_time_ms, event.window_title))

            contextual_artifact = _event_sensitive_artifact(event, known, sensitive_files, lineage)
            if contextual_artifact and process_key:
                recent_artifact_by_process[process_key] = (contextual_artifact, event_time_ms)
            contextual_source = self._resolve_original(contextual_artifact, sensitive_files, lineage)
            if contextual_source and contextual_artifact:
                # Keep the concrete artifact, not just its root. This preserves
                # source -> renamed/exported file -> sink chains.
                recent_sensitive_contexts.append(
                    (normalize_path(contextual_artifact), event_time_ms, event.window_title, process_key)
                )

            clipboard_kind = _clipboard_event_kind(event)
            if clipboard_kind:
                active_clipboard = None
                recent = recent_artifact_by_process.get(process_key)
                if recent and 0 <= event_time_ms - recent[1] <= _CLIPBOARD_SOURCE_HORIZON_MS:
                    active_clipboard = (recent[0], event_time_ms, clipboard_kind, process_family, False)
                elif clipboard_kind == "image":
                    # Screenshot tools commonly own the clipboard, while the
                    # sensitive document remains open in another application.
                    # Keep this bridge narrowly bounded so a generic chat image
                    # cannot taint a later Office document.
                    source = _recent_sensitive_context_source(
                        recent_sensitive_contexts,
                        event_time_ms,
                    )
                    if source:
                        active_clipboard = (source, event_time_ms, clipboard_kind, process_family, True)
            elif active_clipboard:
                (
                    clipboard_source,
                    clipboard_time_ms,
                    clipboard_kind,
                    clipboard_process_family,
                    screenshot_bridge,
                ) = active_clipboard
                elapsed_ms = event_time_ms - clipboard_time_ms
                target_horizon_ms = (
                    _SCREENSHOT_CLIPBOARD_TARGET_HORIZON_MS
                    if clipboard_kind == "image"
                    else _CLIPBOARD_TARGET_HORIZON_MS
                )
                if elapsed_ms > target_horizon_ms:
                    active_clipboard = None
                elif (
                    0 <= elapsed_ms
                    and process_family
                    and (
                        process_family != clipboard_process_family
                        or (
                            clipboard_kind == "image"
                            and _is_screenshot_target(event, normalize_path(target).lower())
                        )
                    )
                    and _is_clipboard_derived_target(
                        event,
                        target,
                        clipboard_kind,
                        require_created_office_target=screenshot_bridge,
                    )
                ):
                    lineage.add(target, clipboard_source)
                    if self._resolve_original(target, sensitive_files, lineage):
                        _remember_known(target, known, known_keys, known_stems)
                        if process_key:
                            recent_artifact_by_process[process_key] = (normalize_path(target), event_time_ms)
                            last_sensitive_by_process[process_key] = self._resolve_original(
                                target,
                                sensitive_files,
                                lineage,
                            )
                        active_clipboard = None

            parent_inferred_source = _source_from_derived_parent_alias(target, sensitive_files)
            archive_inferred_source = _source_from_same_stem_archive(target, sensitive_files)
            title_inferred_source = _source_from_recent_title_context(
                event,
                target,
                recent_document_titles,
                sensitive_files,
            )
            recent_inferred_source = _source_from_recent_document_context(
                event,
                target,
                recent_sensitive_contexts,
                sensitive_files,
                lineage,
            )
            if (
                not parent_inferred_source
                and not archive_inferred_source
                and not title_inferred_source
                and not recent_inferred_source
                and not _may_contribute_lineage(event, original, target, known_keys, known_stems)
            ):
                continue
            text = _event_search_text(event)
            metadata_source = original if original and not _same_exact_file_path(original, target) else ""
            if metadata_source and self._resolve_original(metadata_source, sensitive_files, lineage):
                lineage.add(target, metadata_source)
                _remember_known(target, known, known_keys, known_stems)
            elif target:
                inferred_source = (
                    parent_inferred_source
                    or archive_inferred_source
                    or title_inferred_source
                    or _source_from_derived_filename(target, sensitive_files)
                    or recent_inferred_source
                )
                if inferred_source and (
                    parent_inferred_source
                    or archive_inferred_source
                    or title_inferred_source
                    or _has_derived_transfer_evidence(event, text, target)
                    or _is_recent_document_save_as(event, target, inferred_source, recent_sensitive_contexts)
                ):
                    lineage.add(target, inferred_source)
                    _remember_known(target, known, known_keys, known_stems)

            resolved = self._resolve_original(event.file_path, sensitive_files, lineage)
            if resolved and process_key:
                last_sensitive_by_process[process_key] = resolved
                recent_artifact_by_process[process_key] = (normalize_path(event.file_path), event_time_ms)

            if event.file_path and contains_any(text, TRANSFER_TOKENS):
                source = original or last_sensitive_by_process.get(process_key, "") or _guess_source_by_stem_from_index(target, known_stems)
                if source and not self._resolve_original(source, sensitive_files, lineage):
                    source = ""
                if source and (
                    original
                    or (
                        _is_generated_descendant_name(
                            Path(normalize_path(target)).stem.lower(),
                            Path(normalize_path(source)).stem.lower(),
                        )
                        and _has_derived_transfer_evidence(event, text, target)
                    )
                ):
                    lineage.add(target, source)
                    _remember_known(target, known, known_keys, known_stems)
        for event in sorted(logs, key=lambda item: item.timestamp_ms):
            artifact = normalize_path(target_file_from_metadata(event.raw) or event.file_path)
            resolved_artifact = lineage.resolve_artifact(artifact)
            if resolved_artifact in lineage.direct:
                self._lineage_artifact_times.setdefault(
                    resolved_artifact.lower(),
                    _lineage_event_time_ms(event),
                )
        return lineage

    def _correlate(
        self,
        logs,
        observations,
        sensitive_files: list[str],
        lineage: Lineage,
        *,
        config: EventCorrelatorConfig | None = None,
        recording_start_ms: int = 0,
    ) -> list[CorrelatedEvent]:
        config = config or self.config
        correlated: list[CorrelatedEvent] = []
        observation_time_mode = self._observation_time_mode(observations)
        original_cache: dict[str, str] = {}
        ordered_logs = sorted(logs, key=lambda item: item.timestamp_ms)
        clipboard_sink_ids = _contextual_clipboard_sink_event_ids(ordered_logs)

        for log in ordered_logs:
            if _is_internal_runtime_path(log.file_path):
                continue
            clipboard_sink = log.event_id in clipboard_sink_ids
            path_key = normalize_path(log.file_path).lower() or f"event:{log.event_id}"
            if path_key not in original_cache:
                original_cache[path_key] = self._resolve_log_original(log, ordered_logs, sensitive_files, lineage)
            original = original_cache[path_key]
            observation = self._best_observation_for_log(
                log,
                observations,
                observation_time_mode,
                original,
                sensitive_files,
                lineage,
                config.visual_evidence_horizon_ms,
                allow_external_sink=clipboard_sink,
            )

            if not original and observation:
                original = self._resolve_observation_original(observation, sensitive_files, lineage)
            if not config.non_vlm_enabled and observation is None:
                continue
            if (
                config.non_vlm_enabled
                and observation is None
                and not clipboard_sink
                and not _is_standalone_log_evidence(log)
            ):
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
            log_target = target_file_from_metadata(log.raw) or log.file_path
            cloud_sync_transfer = _is_cloud_sync_directory_transfer(log, log_target, original, lineage)
            confirmed_external_transfer = removable_transfer or clipboard_sink or cloud_sync_transfer
            behavior = "data_exfiltration_candidate" if confirmed_external_transfer else behavior_category(text)
            behavior = _behavior_with_action_status(observation, behavior)
            confidence = self.config.upload_confidence if behavior == "data_exfiltration_candidate" else 0.68
            if observation:
                confidence = max(confidence, observation.confidence)
            current_file = target_file_from_metadata(log.raw) or log.file_path or (
                observation.resource if observation and observation.resource else original
            )
            resolved_artifact = lineage.resolve_artifact(current_file)
            if resolved_artifact and resolved_artifact != normalize_path(current_file):
                current_file = resolved_artifact
            elif not _looks_like_absolute_path(current_file):
                mentioned_artifact = _lineage_artifact_mention(log.window_title, lineage)
                if mentioned_artifact:
                    current_file = mentioned_artifact
            if (
                original
                and current_file
                and (
                    _matches_sensitive_file_reference(current_file, original)
                    or (
                        Path(normalize_path(current_file)).name.lower() == Path(normalize_path(original)).name.lower()
                        and not _looks_like_absolute_path(current_file)
                    )
                )
            ):
                current_file = original
            if (
                observation
                and original
                and _observation_sink_type(observation) == "screen_share"
                and not _looks_like_absolute_path(normalize_path(current_file))
                and lineage.resolve_artifact(current_file) == normalize_path(current_file)
            ):
                # Screen sharing exposes an existing file or derived artifact;
                # a heading/OCR label from the document body is not a file in
                # the leak path.
                current_file = original

            join_reasons = self._join_reasons(log, observation, original, sensitive_files, lineage)
            if clipboard_sink:
                join_reasons.extend(("explicit_sink_log", "clipboard_to_external_app"))
                contextual_sink_type = _contextual_sink_type(log)
                if contextual_sink_type:
                    join_reasons.append(f"sink_type:{contextual_sink_type}")
                if contextual_sink_type == "virtual_machine":
                    join_reasons.append("virtual_machine_sink")
            if cloud_sync_transfer:
                join_reasons.extend(("explicit_sink_log", "cloud_sync_directory_transfer", "sink_type:cloud_sync"))
            correlated.append(
                CorrelatedEvent(
                    event_id=f"corr_{len(correlated)}",
                    timestamp=log.timestamp,
                    event_type=log.event_type,
                    app_name=_correlated_app_name(log, observation),
                    original_file=original,
                    current_file=current_file,
                    operation_type=(
                        "external_sink_interaction"
                        if confirmed_external_transfer
                        else _correlated_operation_type(log, observation, text)
                    ),
                    behavior_category=behavior,
                    confidence=round(min(confidence, 1.0), 3),
                    evidence_refs=tuple(
                        dict.fromkeys(
                            [f"log:{log.event_id}", *(_observation_evidence_refs(observation) if observation else ())]
                        )
                    ),
                    join_reasons=tuple(dict.fromkeys(join_reasons)),
                )
            )
        correlated.extend(
            self._correlate_visual_only(
                observations,
                sensitive_files,
                lineage,
                start_index=len(correlated),
                recording_start_ms=recording_start_ms,
                logs=ordered_logs,
            )
        )
        return correlated

    def _best_observation_for_log(
        self,
        log,
        observations,
        observation_time_mode: str,
        original: str,
        sensitive_files: list[str],
        lineage: Lineage,
        horizon_ms: int,
        *,
        allow_external_sink: bool = False,
    ):
        log_time = self._log_observation_time(log, observation_time_mode)
        best = None
        best_score = -1
        for observation in observations:
            if not _observation_allowed_for_log(log, observation, allow_external_sink=allow_external_sink):
                continue
            distance = abs(log_time - self._observation_center(observation))
            if distance > horizon_ms:
                continue
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
        elif original and _mentions_file(log.window_title, original):
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
        if original and _mentions_file(log.window_title, original):
            reasons.append("log_identity_window")
        if _is_sink_log(log):
            reasons.append("explicit_sink_log")
        if _is_screen_share_context(log):
            reasons.append("sink_type:screen_share")
        if observation is not None:
            resolved = self._resolve_observation_original(observation, sensitive_files, lineage)
            if resolved and original and same_file(resolved, original):
                reasons.append("visual_mentions_sensitive_file")
            text = _observation_search_text(observation)
            if observation.operation_type == "external_sink_interaction" or contains_any(text, SINK_TOKENS):
                reasons.append("visual_sink_context")
            if _is_transfer_observation(observation):
                reasons.append("visual_transfer_context")
            status = _observation_action_status(observation)
            if status != "unknown":
                reasons.append(f"action_status:{status}")
            sink_type = _observation_sink_type(observation)
            if sink_type:
                reasons.append(f"sink_type:{sink_type}")
            declared = _declared_visual_behavior(observation)
            if declared:
                reasons.append(f"visual_declared_behavior:{declared}")
            if _is_unexecuted_visual_preparation(observation):
                reasons.append("visual_unexecuted_preparation")
            if _is_unproven_cloud_folder_claim(observation):
                reasons.append("visual_unproven_cloud_folder")
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
        # Prefer an already-known concrete artifact/binding over a basename or
        # stem fallback. Generic names such as ``普通文件`` can otherwise bind
        # to an unrelated same-named catalog entry.
        normalized_file = normalize_path(file_path).lower()
        artifact = lineage.resolve_artifact(file_path)
        root = lineage.root(artifact)
        if root and normalize_path(root).lower() != normalized_file:
            bound = _lookup_sensitive_source(root, lookup, allow_stem_reference=False)
            if bound:
                return bound
        sensitive = lookup[0].get(normalized_file, "")
        if not sensitive and "/screenmonitor/winows_monitor/" in normalized_file:
            sensitive = lookup[0].get(
                normalized_file.replace("/screenmonitor/winows_monitor/", "/screenmonitor/windows_monitor/"),
                "",
            )
        if sensitive:
            return sensitive
        return _lookup_sensitive_source(file_path, lookup, allow_stem_reference=True)

    def _resolve_log_original(self, log, logs, sensitive_files: list[str], lineage: Lineage) -> str:
        """Resolve a log event from its path, title, and nearby identity context."""

        resolved = self._resolve_original(log.file_path, sensitive_files, lineage)
        if resolved:
            return resolved

        title = normalize_path(log.window_title).lower()
        title_matches = [
            sensitive
            for sensitive in sensitive_files
            if _mentions_exact_filename(title, sensitive)
        ]
        unique_title_matches = _dedupe_paths(title_matches)
        if len(unique_title_matches) == 1:
            return unique_title_matches[0]
        if len(unique_title_matches) > 1:
            nearby_titles = [
                normalize_path(other.window_title).lower()
                for other in logs
                if (
                    other is not log
                    and other.event_type == "app_switch"
                    and other.window_title
                    and abs(other.timestamp_ms - log.timestamp_ms) <= _CLIPBOARD_TARGET_HORIZON_MS
                )
            ]
            narrowed = [
                sensitive
                for sensitive in unique_title_matches
                if any(
                    _mentions_source_directory(nearby_title, sensitive)
                    for nearby_title in nearby_titles
                )
            ]
            unique_narrowed = _dedupe_paths(narrowed)
            if len(unique_narrowed) == 1:
                return unique_narrowed[0]

        if _is_screen_share_context(log):
            future_identity_logs = [
                other
                for other in logs
                if (
                    other is not log
                    and 0 <= other.timestamp_ms - log.timestamp_ms <= 30_000
                    and other.event_type == "app_switch"
                    and other.window_title
                )
            ]
            future_matches = [
                sensitive
                for sensitive in sensitive_files
                if any(_mentions_exact_filename(other.window_title, sensitive) for other in future_identity_logs)
            ]
            unique_future_matches = _dedupe_paths(future_matches)
            if len(unique_future_matches) == 1:
                return unique_future_matches[0]
            if len(unique_future_matches) > 1:
                future_titles = [normalize_path(other.window_title).lower() for other in future_identity_logs]
                narrowed = [
                    sensitive
                    for sensitive in unique_future_matches
                    if any(_mentions_source_directory(title, sensitive) for title in future_titles)
                ]
                unique_narrowed = _dedupe_paths(narrowed)
                if len(unique_narrowed) == 1:
                    return unique_narrowed[0]

        # A clipboard write followed by an external app switch may omit the
        # source path on the sink event. Bind only to a recent, explicit file
        # event rather than to arbitrary filesystem noise.
        if log.event_type == "app_switch":
            nearby_sources = []
            for other in logs:
                if other is log or other.timestamp_ms > log.timestamp_ms:
                    continue
                elapsed = log.timestamp_ms - other.timestamp_ms
                if elapsed < 0 or elapsed > _CLIPBOARD_TARGET_HORIZON_MS:
                    continue
                if other.event_type not in {
                    "file_access",
                    "opened",
                    "read",
                    "clipboard_text",
                    "clipboard_write",
                    "copy",
                    "copied",
                    "file_copied",
                }:
                    continue
                source = self._resolve_original(other.file_path, sensitive_files, lineage)
                if not source:
                    title_matches = [
                        sensitive
                        for sensitive in sensitive_files
                        if _mentions_file(other.window_title, sensitive)
                    ]
                    unique_title_matches = _dedupe_paths(title_matches)
                    if len(unique_title_matches) == 1:
                        source = unique_title_matches[0]
                if source:
                    nearby_sources.append(source)
            unique_sources = _dedupe_paths(nearby_sources)
            if len(unique_sources) == 1:
                return unique_sources[0]

            # The sink may follow a clipboard write by more than the target
            # horizon while the source was opened earlier. In that case use
            # the last clipboard-associated identity window within the source
            # horizon, still requiring a unique sensitive path.
            clipboard_times = [
                other.timestamp_ms
                for other in logs
                if other.timestamp_ms <= log.timestamp_ms
                and log.timestamp_ms - other.timestamp_ms <= _CLIPBOARD_TARGET_HORIZON_MS
                and _clipboard_event_kind(other)
            ]
            if clipboard_times:
                clipboard_time = max(clipboard_times)
                contextual_sources = []
                for other in logs:
                    if other.timestamp_ms > clipboard_time:
                        continue
                    elapsed = clipboard_time - other.timestamp_ms
                    if elapsed < 0 or elapsed > _CLIPBOARD_SOURCE_HORIZON_MS:
                        continue
                    source = self._resolve_original(other.file_path, sensitive_files, lineage)
                    if not source:
                        title_matches = [
                            sensitive
                            for sensitive in sensitive_files
                            if _mentions_file(other.window_title, sensitive)
                        ]
                        unique_title_matches = _dedupe_paths(title_matches)
                        if len(unique_title_matches) == 1:
                            source = unique_title_matches[0]
                    if source:
                        contextual_sources.append(source)
                unique_contextual_sources = _dedupe_paths(contextual_sources)
                if len(unique_contextual_sources) == 1:
                    return unique_contextual_sources[0]
        return ""

    def _resolve_observation_original(self, observation, sensitive_files: list[str], lineage: Lineage) -> str:
        candidates = [observation.resource, *observation.related_resources, observation.description]
        resolved_mentions = [
            (candidate, resolved)
            for candidate in candidates
            if (resolved := self._resolve_original(candidate, sensitive_files, lineage))
        ]
        if resolved_mentions:
            # Prefer an exact/full-path identity; otherwise use the complete
            # visual description so folder context (for example OneDrive >
            # Desktop) can disambiguate same-named sources.
            exact = [
                resolved
                for candidate, resolved in resolved_mentions
                if _looks_like_absolute_path(normalize_path(candidate))
            ]
            if len(set(exact)) == 1:
                return exact[0]
            unique_resolved = _dedupe_paths([resolved for _, resolved in resolved_mentions])
            if len(unique_resolved) == 1:
                return unique_resolved[0]
        return _best_sensitive_mention(" ".join(candidates), sensitive_files)

    def _fuse_visual_evidence(
        self,
        logs,
        observations,
        sensitive_files: list[str],
        lineage: Lineage,
        *,
        horizon_ms: int,
    ):
        """Bind a sink/result observation to nearby file-identity evidence."""

        if not observations or not sensitive_files or horizon_ms <= 0:
            return observations
        time_mode = self._observation_time_mode(observations)
        fused = []
        for sink in observations:
            resolved_sink = self._resolve_observation_original(sink, sensitive_files, lineage)
            ambiguous_sink = _has_ambiguous_sensitive_label(sink, sensitive_files)
            if sink.source == "log_anchored" or (resolved_sink and not ambiguous_sink):
                fused.append(sink)
                continue
            sink_text = _observation_search_text(sink)
            sink_status = _observation_action_status(sink)
            is_actionable = (
                _is_unbound_visual_risk(sink, sink_text)
                and "unknown_risk" not in sink_text.lower()
            ) or sink_status in {
                "submitted",
                "in_progress",
                "completed",
                "failed",
            }
            if not is_actionable or not (_is_external_observation(sink) or _is_transfer_observation(sink)):
                fused.append(sink)
                continue

            sink_time = self._observation_center(sink)
            bindings: list[tuple[int, int, str, str, Any]] = []
            if ambiguous_sink:
                explicit_log_identity = self._unique_explicit_session_log_identity(
                    sink,
                    logs,
                    sensitive_files,
                    lineage,
                )
                if explicit_log_identity:
                    original, identity_log = explicit_log_identity
                    # An exact path emitted by this session is stronger identity
                    # evidence than a nearby title containing only a common
                    # basename. This also covers VLM events timestamped at a
                    # terminal/result frame long after the identity frame.
                    bindings.append((-1_000_000, -1, original, "log", identity_log))
            for identity in observations:
                if identity is sink or identity.source == "log_anchored":
                    continue
                original = self._resolve_observation_original(identity, sensitive_files, lineage)
                if not original:
                    continue
                identity_time = self._observation_center(identity)
                distance = abs(sink_time - identity_time)
                if distance > horizon_ms:
                    continue
                adjusted_distance = distance + (5_000 if identity_time > sink_time else 0)
                app_rank = _app_compatibility_rank(sink.app_name, identity.app_name)
                bindings.append((adjusted_distance + app_rank * 10_000, 0, original, "visual", identity))

            for log in logs:
                original = self._resolve_log_original(log, logs, sensitive_files, lineage)
                if not original:
                    continue
                log_time = self._log_observation_time(log, time_mode)
                distance = abs(sink_time - log_time)
                if distance > horizon_ms:
                    continue
                adjusted_distance = distance + (5_000 if log_time > sink_time else 0)
                app_rank = _app_compatibility_rank(sink.app_name, log.app_name or log.process_name)
                bindings.append((adjusted_distance + app_rank * 10_000, 1, original, "log", log))

            binding = _choose_identity_binding(bindings)
            if binding is None:
                fused.append(sink)
                continue
            _, _, original, source_kind, identity = binding
            if source_kind == "visual":
                identity_id = identity.observation_id
                identity_resource = identity.resource or original
                related = [*sink.related_resources, identity.resource, *identity.related_resources, original]
                marker = f"visual_identity={identity_id}"
                identity_frames = _description_evidence_frame_ids(identity.description)
                if identity_frames:
                    marker += " visual_identity_frame_ids=" + "|".join(identity_frames)
                confidence = max(sink.confidence, min(identity.confidence, 0.95))
            else:
                identity_id = identity.event_id
                identity_resource = identity.file_path or original
                related = [*sink.related_resources, identity.file_path, original]
                marker = f"log_identity={identity_id}"
                confidence = sink.confidence
            fused.append(
                replace(
                    sink,
                    resource=normalize_path(identity_resource),
                    related_resources=tuple(
                        dict.fromkeys(normalize_path(item) for item in related if normalize_path(item))
                    ),
                    description=f"{sink.description} {marker}.",
                    confidence=confidence,
                )
            )
        return fused

    def _unique_explicit_session_log_identity(
        self,
        observation,
        logs,
        sensitive_files: list[str],
        lineage: Lineage,
    ):
        """Resolve a basename-only observation from unique exact session paths."""

        labels = {
            Path(normalize_path(reference)).name.lower()
            for reference in (observation.resource, *observation.related_resources)
            if normalize_path(reference) and not _looks_like_absolute_path(normalize_path(reference))
        }
        if not labels:
            return None

        matches: list[tuple[str, Any]] = []
        for log in logs:
            candidates = (
                log.file_path,
                original_file_from_metadata(log.raw),
                target_file_from_metadata(log.raw),
            )
            for candidate in candidates:
                normalized = normalize_path(candidate)
                if not _looks_like_absolute_path(normalized):
                    continue
                original = self._resolve_original(normalized, sensitive_files, lineage)
                if not original:
                    continue
                candidate_names = {
                    Path(normalized).name.lower(),
                    Path(normalize_path(original)).name.lower(),
                }
                if labels.isdisjoint(candidate_names):
                    continue
                matches.append((original, log))
                break

        unique_sources = _dedupe_paths([original for original, _ in matches])
        if len(unique_sources) != 1:
            return None
        source = unique_sources[0]
        identity_log = next(log for original, log in matches if same_file(original, source))
        return source, identity_log

    def _add_visual_lineage(self, observations, sensitive_files: list[str], lineage: Lineage, *, logs=None) -> None:
        for observation in observations:
            if observation.source == "log_anchored":
                continue
            explicit_derivation = _explicit_visual_derivation(
                observation,
                logs or [],
                sensitive_files,
                lineage,
            )
            if explicit_derivation:
                derived, source = explicit_derivation
                lineage.add(derived, source, replace_existing=True)
                self._lineage_artifact_times.setdefault(
                    normalize_path(derived).lower(),
                    self._observation_center(observation),
                )
            declared_behavior = _declared_visual_behavior(observation)
            if declared_behavior and declared_behavior != "hidden_transfer":
                continue
            original = self._resolve_visual_original_without_lineage(observation, sensitive_files)
            if not original:
                continue
            text = _observation_search_text(observation)
            # Outbound observations describe what reached a sink, not how a new
            # local artifact was created. Only transformation observations may
            # extend lineage; otherwise a hallucinated attachment name can
            # become a fake derived file.
            if not _is_transfer_observation(observation):
                continue
            for candidate in [observation.resource, *observation.related_resources]:
                derived = lineage.resolve_artifact(normalize_path(candidate))
                if _is_visual_derived_candidate(derived, original):
                    lineage.add(derived, original)
                    self._lineage_artifact_times.setdefault(
                        normalize_path(derived).lower(),
                        self._observation_center(observation),
                    )

    def _bind_log_lineage_aliases(self, logs, sensitive_files: list[str], lineage: Lineage) -> None:
        """Bind a VLM basename-only derivative to its unique absolute log path."""

        for log in logs:
            target = normalize_path(target_file_from_metadata(log.raw) or log.file_path)
            if not _looks_like_absolute_path(target) or lineage.root(target) != target:
                continue
            name = Path(target).name.lower()
            aliases = [artifact for artifact in lineage.direct if Path(normalize_path(artifact)).name.lower() == name]
            sources = _dedupe_paths(
                self._resolve_original(alias, sensitive_files, lineage)
                for alias in aliases
            )
            if len(sources) != 1:
                continue
            lineage.add(target, sources[0])
            self._lineage_artifact_times.setdefault(target.lower(), _lineage_event_time_ms(log))

    def _resolve_visual_original_without_lineage(self, observation, sensitive_files: list[str]) -> str:
        candidates = [observation.resource, *observation.related_resources, observation.description]
        return _best_sensitive_mention(" ".join(candidates), sensitive_files)

    def _correlate_visual_only(
        self,
        observations,
        sensitive_files: list[str],
        lineage: Lineage,
        start_index: int,
        recording_start_ms: int,
        logs=None,
    ) -> list[CorrelatedEvent]:
        visual_events: list[CorrelatedEvent] = []
        for observation in observations:
            if observation.source == "log_anchored":
                continue
            log_binding = self._visual_log_binding(
                observation,
                logs or [],
                sensitive_files,
                lineage,
            )
            original = log_binding[0] if log_binding else self._resolve_observation_original(
                observation,
                sensitive_files,
                lineage,
            )
            if not original:
                original = _partial_visual_filename_identity(
                    observation,
                    logs or [],
                    sensitive_files,
                )
            text = f"{observation.description} {observation.operation_type} {observation.resource} {' '.join(observation.related_resources)}"
            if not (_is_external_observation(observation) or _is_transfer_observation(observation)):
                continue
            if (
                not original
                and _declared_visual_behavior(observation) == "unknown_risk"
                and _observation_action_status(observation) == "unknown"
            ):
                continue
            if not original and not _is_unbound_visual_risk(observation, text):
                continue
            declared_behavior = _declared_visual_behavior(observation)
            if declared_behavior == "unknown_risk" and _observation_action_status(observation) == "unknown":
                behavior = "unknown_risk"
            else:
                behavior = behavior_category(text) if original else "unknown_risk"
            behavior = _behavior_with_action_status(observation, behavior)
            current_file = (
                log_binding[1]
                if log_binding and log_binding[1]
                else self._visual_current_file(observation, original, sensitive_files, lineage)
            )
            visual_event = CorrelatedEvent(
                event_id=f"corr_{start_index + len(visual_events)}",
                timestamp=_observation_timestamp(observation, recording_start_ms),
                event_type="visual_observation",
                app_name=observation.app_name,
                original_file=original,
                current_file=current_file,
                operation_type=_effective_visual_operation_type(observation),
                behavior_category=behavior,
                confidence=round(min(max(observation.confidence, 0.70), 1.0), 3),
                evidence_refs=_observation_evidence_refs(observation),
                join_reasons=tuple(
                    _visual_join_reasons(observation, original)
                    + _screen_share_state_reasons(observation, logs or [])
                ),
            )
            visual_events.append(visual_event)
            if _is_visual_file_split(observation) and current_file and not same_file(original, current_file):
                # A split followed by cloud/mail/chat delivery contains two
                # semantically useful actions. Keep the confirmed external
                # event above, and emit the local derivation independently.
                visual_events.append(
                    CorrelatedEvent(
                        event_id=f"corr_{start_index + len(visual_events)}",
                        timestamp=visual_event.timestamp,
                        event_type="visual_transformation",
                        app_name=observation.app_name,
                        original_file=original,
                        current_file=current_file,
                        operation_type="file_or_content_transfer",
                        behavior_category="hidden_transformation_candidate",
                        confidence=visual_event.confidence,
                        evidence_refs=visual_event.evidence_refs,
                        join_reasons=(*visual_event.join_reasons, "visual_file_split"),
                    )
                )
        return visual_events

    def _visual_log_binding(self, observation, logs, sensitive_files: list[str], lineage: Lineage):
        """Use a nearby explicit sink log to disambiguate a generic VLM filename."""

        center = self._observation_center(observation)
        observation_sink = _observation_sink_type(observation)
        candidates = []
        for log in logs:
            distance = abs(log.timestamp_ms - center)
            if distance > _CLIPBOARD_SOURCE_HORIZON_MS:
                continue
            contextual_sink = _contextual_sink_type(log)
            cloud_sync_result = (
                observation_sink == "cloud_sync"
                and log.event_type.lower()
                in {"created", "modified", "renamed", "copied", "file_created", "file_modified", "file_renamed"}
                and _is_cloud_sync_user_path(log.file_path)
                and _is_user_visible_lineage_target(log.file_path)
            )
            if not (
                _is_sink_log(log)
                or log.event_type in {"file_selected", "file_upload", "upload", "uploaded"}
                or (contextual_sink and (not observation_sink or contextual_sink == observation_sink))
                or cloud_sync_result
            ):
                continue
            original = self._resolve_log_original(log, logs, sensitive_files, lineage)
            if not original:
                continue
            observed_artifact = lineage.resolve_artifact(observation.resource)
            if (
                observed_artifact in lineage.direct
                and same_file(lineage.root(observed_artifact), original)
            ):
                current = observed_artifact
            else:
                current = lineage.resolve_artifact(log.file_path)
            if not current or current == normalize_path(log.file_path):
                current = _lineage_artifact_mention(log.window_title, lineage) or current
            if _is_source_label_only(current, original):
                current = original
            if current:
                current_root = lineage.root(lineage.resolve_artifact(current))
                if not (
                    _same_exact_file_path(current, original)
                    or _same_exact_file_path(current_root, original)
                ):
                    continue
            artifact_rank = 1
            if (
                observation_sink == "cloud_sync"
                and current
                and not _same_exact_file_path(current, original)
            ):
                artifact_rank = 0
            candidates.append((artifact_rank, distance, original, current))
        if not candidates:
            return None
        candidates.sort(key=lambda item: item[:2])
        best_score = candidates[0][:2]
        best = [item for item in candidates if item[:2] == best_score]
        roots = _dedupe_paths([item[2] for item in best])
        return (best[0][2], best[0][3]) if len(roots) == 1 else None

    def _visual_current_file(
        self,
        observation,
        original: str,
        sensitive_files: list[str],
        lineage: Lineage,
    ) -> str:
        current_file = observation.resource or original
        if normalize_path(current_file).strip().lower() in {"unknown", "none", "null", "n/a", "na", "未知"}:
            current_file = original
        if not current_file:
            return original
        removable_destination = _visual_removable_destination(observation, original)
        if removable_destination:
            return removable_destination
        distinct_remote_output = _is_explicit_remote_output(observation, current_file, original)
        if original and _is_source_label_only(current_file, original) and not distinct_remote_output:
            current_file = original
        action = _declared_visual_action(observation)
        if (
            original
            and not distinct_remote_output
            and (same_file(current_file, original) or _matches_sensitive_file_reference(current_file, original))
        ):
            current_file = original
        if original and action == "chat_upload" and _mentions_exact_filename(observation.description, original):
            return original
        if original and action in _OUTBOUND_SOURCE_OBJECT_ACTIONS and _mentions_exact_filename(observation.description, original):
            derived = self._latest_visible_descendant_before(original, observation, lineage)
            return derived or original
        resolved_artifact = lineage.resolve_artifact(current_file)
        if original:
            root = lineage.root(resolved_artifact)
            if root and same_file(root, original) and not same_file(resolved_artifact, original):
                return resolved_artifact
            if action == "cloud_sync" and same_file(resolved_artifact, original):
                derived = _unique_cloud_sync_descendant(original, lineage)
                if derived:
                    return derived
            if (
                action in _DECLARED_VISUAL_OUTBOUND_ACTIONS
                or _is_external_observation(observation)
            ) and (
                same_file(resolved_artifact, original)
                or _matches_sensitive_file_reference(resolved_artifact, original)
                or "..." in normalize_path(current_file)
                or "…" in normalize_path(current_file)
            ):
                derived = self._latest_visible_descendant_before(original, observation, lineage)
                if derived:
                    return derived
        if normalize_path(resolved_artifact).lower() != normalize_path(current_file).lower():
            return resolved_artifact
        if (
            original
            and action in _DECLARED_VISUAL_OUTBOUND_ACTIONS
            and _mentions_file(observation.description, original)
            and not _mentions_file(current_file, original)
        ):
            return original
        return current_file

    def _latest_visible_descendant_before(self, original: str, observation, lineage: Lineage) -> str:
        center = self._observation_center(observation)
        candidates = []
        for artifact in lineage.direct:
            normalized = normalize_path(artifact)
            if _same_exact_file_path(normalized, original) or lineage.root(normalized) != lineage.root(original):
                continue
            if not _is_user_visible_lineage_target(normalized):
                continue
            timestamp = self._lineage_artifact_times.get(normalized.lower(), 0)
            if timestamp and center and timestamp > center:
                continue
            candidates.append((timestamp, normalized))
        if not candidates:
            return ""
        candidates.sort(key=lambda item: (item[0], item[1].lower()), reverse=True)
        return candidates[0][1]

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


def _normalize_cached_observation_semantics(observation):
    """Reapply deterministic semantics to observations loaded from older caches."""

    text = observation.description.lower()
    declared = _declared_visual_behavior(observation)
    status = _observation_action_status(observation)
    submitted_to_web = (
        declared == "hidden_transfer"
        and status in {"submitted", "in_progress", "completed"}
        and (
            bool(re.search(r"\bpasted\b.{0,100}\binto\b", text))
            or any(
            marker in text
            for marker in ("pasted into", "pasted to", "pasted the content", "submitted to", "粘贴到", "提交到")
            )
        )
        and (
            bool(re.search(r"\b[a-z0-9-]+\.(?:com|cn|net|org|io|ai)\b", text))
            or any(
                marker in text
                for marker in ("web-based", "online tool", "online service", "third-party tool", "在线工具", "第三方工具")
            )
        )
    )
    if submitted_to_web:
        description = re.sub(
            r"^\s*hidden_transfer\s*:\s*[^.]+",
            "direct_leak: paste_to_web",
            observation.description,
            count=1,
            flags=re.IGNORECASE,
        )
        description = re.sub(
            r"\bsink_type=unknown\b",
            "sink_type=network_upload",
            description,
            count=1,
            flags=re.IGNORECASE,
        )
        return replace(
            observation,
            operation_type="external_sink_interaction",
            description=description,
        )

    passive_local_preview = (
        any(marker in text for marker in ("file explorer", "windows explorer", "资源管理器"))
        and any(marker in text for marker in ("preview pane", "preview panel", "预览窗格", "预览栏"))
        and any(marker in text for marker in ("ai summary", "ai public document", "ai 公文", "ai公文"))
        and not any(marker in text for marker in ("clicked ai", "submitted to ai", "uploaded to ai", "点击ai", "提交到ai"))
    )
    inferred_toolbar_upload = (
        any(marker in text for marker in ("is displayed in", "is visible in", "显示在", "可见于"))
        and any(marker in text for marker in ("toolbar shows", "toolbar contains", "工具栏显示", "工具栏包含"))
        and any(
            marker in text
            for marker in (
                "implying the file was uploaded",
                "suggesting the file was uploaded",
                "therefore it was uploaded",
                "由此推断文件已上传",
                "暗示文件已上传",
            )
        )
    )
    monitoring_log_only_claim = _is_monitoring_log_only_claim(text)
    inferred_recording_attachment = _is_inferred_recording_attachment(text, status=status)
    if declared == "direct_leak" and (
        passive_local_preview
        or inferred_toolbar_upload
        or monitoring_log_only_claim
        or inferred_recording_attachment
    ):
        downgraded_action = (
            "monitoring_log_claim"
            if monitoring_log_only_claim
            else "recording_attachment_inference"
            if inferred_recording_attachment
            else "local_preview"
        )
        description = re.sub(
            r"^\s*direct_leak\s*:\s*[^.]+",
            f"unknown_risk: {downgraded_action}",
            observation.description,
            count=1,
            flags=re.IGNORECASE,
        )
        description = re.sub(r"\bsink_type=[a-z_]+\b", "sink_type=unknown", description, count=1, flags=re.IGNORECASE)
        description = re.sub(
            r"\baction_status=(?:selected|submitted|in_progress|completed|failed)\b",
            "action_status=unknown",
            description,
            count=1,
            flags=re.IGNORECASE,
        )
        return replace(observation, operation_type="file_or_content_transfer", description=description)

    separate_upload_not_started = (
        status == "selected"
        and (
            (
                any(marker in text for marker in ("upload button is visible", "上传按钮可见", "显示上传按钮"))
                and any(marker in text for marker in ("no upload progress", "upload has not started", "未开始上传", "没有上传进度"))
            )
            or (
                any(marker in text for marker in ("staged in the upload area", "ready for upload", "已进入上传区域", "等待上传"))
                and ("upload progress" not in text or "no upload progress" in text)
                and not any(marker in text for marker in ("upload completed", "was uploaded", "上传完成", "已上传"))
            )
        )
    )
    local_recording_preview = (
        status == "selected"
        and any(marker in text for marker in ("screen recording preview", "recording preview", "录屏预览"))
        and any(marker in text for marker in ("send to", "发送到"))
        and any(marker in text for marker in ("button", "按钮", "staged", "待发送"))
        and not any(marker in text for marker in ("was sent", "sent successfully", "点击发送", "发送成功", "已发送"))
    )
    if declared == "direct_leak" and (separate_upload_not_started or local_recording_preview):
        description = re.sub(
            r"^\s*direct_leak\s*:\s*[^.]+",
            "unknown_risk: preparation",
            observation.description,
            count=1,
            flags=re.IGNORECASE,
        )
        description = re.sub(
            r"\baction_status=selected\b",
            "action_status=unknown",
            description,
            count=1,
            flags=re.IGNORECASE,
        )
        return replace(
            observation,
            operation_type="file_or_content_transfer",
            description=description,
        )
    return observation


def _is_monitoring_log_only_claim(text: str) -> bool:
    monitoring = any(
        marker in text
        for marker in (
            "monitoring logs",
            "monitor log",
            "powershell log",
            "logs in powershell",
            "监控日志",
            "powershell 日志",
        )
    )
    inferred_from_log = any(
        marker in text
        for marker in (
            "log confirms",
            "logs explicitly record",
            "log explicitly records",
            "based on the log",
            "日志确认",
            "日志表明",
        )
    )
    direct_ui_evidence = any(
        marker in text
        for marker in (
            "attachment card is visible",
            "upload progress is visible",
            "send confirmation is visible",
            "generated answer is visible",
            "上传进度可见",
            "附件卡片可见",
            "发送确认可见",
        )
    )
    return monitoring and inferred_from_log and not direct_ui_evidence


def _is_inferred_recording_attachment(text: str, *, status: str) -> bool:
    if status != "selected":
        return False
    recording = any(marker in text for marker in ("screen recording", "screenshot", "recording mp4", "录屏", "截图"))
    attachment = any(marker in text for marker in ("attachment", "staged", "thumbnail", "附件", "待发送", "缩略图"))
    inferred = any(
        marker in text
        for marker in (
            "likely the screen recording",
            "likely a screen recording",
            "likely a screenshot",
            "given the dark thumbnail",
            "appears to contain the recording",
            "可能是录屏",
            "可能是截图",
        )
    )
    clearly_identified = any(
        marker in text
        for marker in (
            "attachment card is visible",
            "identified attachment card",
            "clearly identified screenshot",
            "明确的截图附件",
        )
    )
    completed = any(marker in text for marker in ("was sent", "sent successfully", "发送成功", "已发送"))
    return recording and attachment and inferred and not clearly_identified and not completed


def _is_external_observation(observation) -> bool:
    text = _observation_search_text(observation)
    return _effective_visual_operation_type(observation) == "external_sink_interaction" or contains_any(text, SINK_TOKENS)


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
    if original and _mentions_file(log.window_title, original):
        return True
    return bool(log.file_path and _mentions_file(text, log.file_path))


def _is_transfer_observation(observation) -> bool:
    if _effective_visual_operation_type(observation) == "external_sink_interaction":
        return False
    text = _observation_search_text(observation)
    return observation.operation_type == "file_or_content_transfer" or contains_any(text, TRANSFER_TOKENS)


def _observation_allowed_for_log(log, observation, *, allow_external_sink: bool = False) -> bool:
    if observation.source == "log_anchored":
        return False
    return not _is_external_observation(observation) or allow_external_sink or _is_sink_log(log)


def _is_sink_log(log) -> bool:
    if _is_screen_share_context(log):
        return True
    if log.event_type in {"file_selected", "file_upload", "upload", "uploaded", "upload_complete", "send_click"}:
        return True
    if (log.process_name or log.app_name or "").lower() == "fsquirt.exe":
        return True
    extra = log.raw.get("extra") if isinstance(log.raw.get("extra"), dict) else {}
    raw_operation = str(log.raw.get("operation") or extra.get("raw_operation") or "")
    category = str(extra.get("category") or "")
    return raw_operation in {"file_selected", "file_upload", "upload", "send_click"} or contains_any(category, ("文件上传", "直接外发"))


def _is_screen_share_context(log) -> bool:
    if log.event_type not in {"app_switch", "screen_share"}:
        return False
    text = normalize_path(f"{log.app_name} {log.window_title} {_event_search_text(log)}").lower()
    has_share_action = contains_any(
        text,
        (
            "正在共享你的屏幕",
            "正在共享屏幕",
            "screen is being shared",
            "sharing your screen",
            "screen sharing",
            "share screen",
            "屏幕共享中",
            "屏幕共享会议控件",
        ),
    )
    return has_share_action and contains_any(text, ("teams", "zoom", "meeting", "会议", "共享"))


def _add_confirmed_git_push_events(
    correlated: list[CorrelatedEvent],
    logs,
) -> list[CorrelatedEvent]:
    """Promote a terminal Git staging chain plus an observed push command.

    Clipboard monitoring can attribute the copied ``git push`` command to a
    browser. It becomes external-transfer evidence only when a nearby visual
    terminal event has already bound a sensitive file to Git preparation.
    """

    result = list(correlated)
    for log in logs:
        text = _event_search_text(log).lower()
        if not re.search(r"\bgit\s+push\b", text):
            continue
        timestamp_ms = log.timestamp_ms
        candidates = [
            event
            for event in result
            if event.original_file
            and event.app_name.lower() in {"cmd", "cmd.exe", "windowsterminal", "powershell", "powershell.exe"}
            and any(ref.startswith("frame:") for ref in event.evidence_refs)
            and 0 <= timestamp_ms - parse_timestamp_ms(event.timestamp) <= 30_000
        ]
        if not candidates:
            continue
        source = max(candidates, key=lambda event: parse_timestamp_ms(event.timestamp))
        if not _has_git_stage_evidence(logs, source.original_file, timestamp_ms):
            continue
        result.append(
            CorrelatedEvent(
                event_id=f"corr_{len(result)}",
                timestamp=log.timestamp,
                event_type="git_push",
                app_name="WindowsTerminal",
                original_file=source.original_file,
                current_file=source.current_file or source.original_file,
                operation_type="external_sink_interaction",
                behavior_category="data_exfiltration_candidate",
                confidence=0.95,
                evidence_refs=tuple(dict.fromkeys((*source.evidence_refs, f"log:{log.event_id}"))),
                join_reasons=(
                    "terminal_git_push",
                    "explicit_sink_log",
                    "sink_type:network_upload",
                    "git_staged_sensitive_file",
                ),
            )
        )
    return result


def _has_git_stage_evidence(logs, original_file: str, push_timestamp_ms: int) -> bool:
    filename = Path(normalize_path(original_file)).name.lower()
    if not filename:
        return False
    for log in logs:
        if not 0 <= push_timestamp_ms - log.timestamp_ms <= 30_000:
            continue
        text = _event_search_text(log).lower()
        if re.search(r"\bgit\s+(?:add|commit)\b", text) and filename in text:
            return True
    return False


def _contextual_clipboard_sink_event_ids(logs) -> set[str]:
    """Find sensitive-file app switches immediately following a clipboard write."""

    result: set[str] = set()
    last_clipboard_ms = -1
    for log in sorted(logs, key=lambda item: item.timestamp_ms):
        if _clipboard_event_kind(log):
            last_clipboard_ms = log.timestamp_ms
            continue
        if last_clipboard_ms < 0 or log.event_type != "app_switch":
            continue
        elapsed = log.timestamp_ms - last_clipboard_ms
        if elapsed < 0 or elapsed > _CLIPBOARD_TARGET_HORIZON_MS:
            continue
        if _is_preparation_only_external_window(log):
            continue
        identity = identify_frontend_app(log.app_name, log.window_title)
        if (
            identity.category in RISKY_APP_CATEGORIES
            and identity.category != "browser"
        ) or identity.category == "external_sink" or _is_virtual_machine_context(log):
            result.add(log.event_id)
    return result


def _is_virtual_machine_context(log) -> bool:
    text = normalize_path(
        " ".join(
            (
                log.app_name,
                log.process_name,
                log.window_title,
                _event_search_text(log),
            )
        )
    ).lower()
    return contains_any(text, ("vmware", "virtualbox", "parallels", "虚拟机", "virtual machine"))


def _is_preparation_only_external_window(log) -> bool:
    title = normalize_path(log.window_title).lower()
    if contains_any(title, ("消息", "聊天", "chat", "conversation", "editing", "编辑", "发送", "send")):
        return False
    return contains_any(
        title,
        (
            "未命名文档",
            "untitled document",
            "飞书云文档",
            "lark docs",
            "ai工作平台",
            "ai 工作平台",
            "飞书官网",
        ),
    )


def _is_internal_runtime_path(file_path: str) -> bool:
    """Exclude monitor artifacts and application internals from sensitive flows."""

    path = normalize_path(file_path).lower()
    if not path:
        return False
    markers = (
        "/recordings/session_",
        "/screenmonitor/windows_monitor/recordings/",
        "/screenmonitor/winows_monitor/recordings/",
        "/appdata/",
        "/cache/",
        "/cachedata/",
        "/cacheddata/",
        "/program files/",
        "/programdata/",
        "/temp/",
    )
    return any(marker in path for marker in markers)


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
    raw = log.raw if isinstance(log.raw, dict) else {}
    disk = raw.get("disk_info") if isinstance(raw.get("disk_info"), dict) else {}
    extra = raw.get("extra") if isinstance(raw.get("extra"), dict) else {}
    operation = " ".join(
        [
            log.event_type,
            str(raw.get("operation") or ""),
            str(extra.get("raw_operation") or ""),
            str(extra.get("category") or ""),
        ]
    )
    destination_context = " ".join(
        [
            log.file_path,
            log.window_title,
            str(raw.get("destination_path") or ""),
            str(raw.get("destination_name") or ""),
            str(disk.get("drive_letter") or ""),
            str(disk.get("disk_type") or ""),
            observation.resource if observation else "",
            " ".join(observation.related_resources) if observation else "",
            observation.description if observation else "",
        ]
    )
    # Arbitrary clipboard text can contain strings such as a different case ID
    # with "USB" in its name. Only destination/window/device context may prove
    # removable media; clipboard content alone is not a transfer destination.
    if not _is_removable_media_context(destination_context):
        return False
    return contains_any(f"{operation} {destination_context}", TRANSFER_TOKENS + SINK_TOKENS) or log.event_type in {
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


def _lineage_event_time_ms(event) -> int:
    if event.timestamp_ms:
        return int(event.timestamp_ms)
    return max(int(event.video_time_ms), 0)


def _lineage_process_family(event) -> str:
    process = (event.process_name or event.app_name or "").strip().lower().removesuffix(".exe")
    if process in {"et", "wps", "wpp"}:
        return "wps_office"
    if process in {"excel", "powerpnt", "winword"}:
        return "microsoft_office"
    return process


def _clipboard_event_kind(event) -> str:
    extra = event.raw.get("extra") if isinstance(event.raw.get("extra"), dict) else {}
    values = {
        event.event_type.lower(),
        str(event.raw.get("operation") or "").lower(),
        str(extra.get("raw_operation") or "").lower(),
    }
    if "clipboard_image" in values:
        return "image"
    if values & {"clipboard_copy", "clipboard_text", "clipboard_write"}:
        return "text"
    return ""


def _recent_sensitive_context_source(
    contexts: list[tuple[str, int, str, str]],
    timestamp_ms: int,
) -> str:
    for artifact, context_ms, _title, _process in reversed(contexts):
        if timestamp_ms - context_ms > _SCREENSHOT_CLIPBOARD_CONTEXT_HORIZON_MS:
            break
        if context_ms <= timestamp_ms:
            return artifact
    return ""


def _event_sensitive_artifact(event, known: list[str], sensitive_files: list[str], lineage: Lineage) -> str:
    for candidate in (
        event.file_path,
        original_file_from_metadata(event.raw),
        target_file_from_metadata(event.raw),
    ):
        artifact = normalize_path(candidate)
        if artifact and _artifact_has_sensitive_root(artifact, sensitive_files, lineage):
            return artifact
    return _unique_known_artifact_mention(_event_search_text(event), known, sensitive_files, lineage)


def _artifact_has_sensitive_root(artifact: str, sensitive_files: list[str], lineage: Lineage) -> bool:
    lookup = _sensitive_lookup(tuple(sensitive_files))
    if _lookup_sensitive_source(artifact, lookup, allow_stem_reference=True):
        return True
    return bool(_lookup_sensitive_source(lineage.root(artifact), lookup, allow_stem_reference=False))


def _unique_known_artifact_mention(
    text: str,
    known: list[str],
    sensitive_files: list[str],
    lineage: Lineage,
) -> str:
    normalized_text = normalize_path(text).lower()
    if not normalized_text:
        return ""
    full_matches = {
        normalize_path(artifact)
        for artifact in known
        if normalize_path(artifact).lower() in normalized_text
        and _artifact_has_sensitive_root(artifact, sensitive_files, lineage)
    }
    if len(full_matches) == 1:
        return next(iter(full_matches))

    name_matches = {
        normalize_path(artifact)
        for artifact in known
        if len(Path(normalize_path(artifact)).name) >= 4
        and Path(normalize_path(artifact)).name.lower() in normalized_text
        and _artifact_has_sensitive_root(artifact, sensitive_files, lineage)
    }
    return next(iter(name_matches)) if len(name_matches) == 1 else ""


def _is_clipboard_derived_target(
    event,
    target: str,
    clipboard_kind: str,
    *,
    require_created_office_target: bool = False,
) -> bool:
    if event.event_type.lower() not in _CLIPBOARD_TARGET_EVENTS:
        return False
    normalized = normalize_path(target)
    lowered = normalized.lower()
    if (
        not _looks_like_absolute_path(normalized)
        or Path(lowered).suffix not in _CLIPBOARD_DOCUMENT_EXTENSIONS
        or any(marker in lowered for marker in _CLIPBOARD_HIDDEN_PATH_MARKERS)
    ):
        return False

    name = Path(lowered).name
    title = normalize_path(event.window_title).lower()
    if name and name in title:
        if require_created_office_target:
            return event.event_type.lower() in {"created", "file_created"}
        return True
    if clipboard_kind != "image":
        return False
    if _is_screenshot_target(event, lowered):
        return True
    return (
        (not require_created_office_target or event.event_type.lower() in {"created", "file_created"})
        and _is_office_document_target(event, lowered)
    )


def _is_office_document_target(event, target: str) -> bool:
    suffix = Path(target).suffix.lower()
    if suffix not in _CLIPBOARD_DOCUMENT_EXTENSIONS or suffix == ".txt":
        return False
    process_family = _lineage_process_family(event)
    if process_family not in {"wps_office", "microsoft_office"}:
        return False
    title = normalize_path(event.window_title).lower()
    name = Path(target).name.lower()
    return (
        name in title
        or (suffix == ".pdf" and any(marker in title for marker in ("输出为pdf", "成功输出")))
        or (event.event_type.lower() in {"created", "file_created"} and not title)
    )


def _is_screenshot_target(event, target: str) -> bool:
    if Path(target).suffix not in {".bmp", ".jpeg", ".jpg", ".png"}:
        return False
    name = Path(target).name
    extra = event.raw.get("extra") if isinstance(event.raw.get("extra"), dict) else {}
    operation = str(event.raw.get("operation") or extra.get("raw_operation") or "").lower()
    process = (event.process_name or event.app_name or "").lower()
    return (
        "/pictures/screenshots/" in target
        or "screenshot" in name
        or "屏幕截图" in name
        or "截图" in name
        or "snippingtool" in process
        or operation in {"screen_capture", "screenshot"}
    )


def _has_derived_transfer_evidence(event, text: str, target: str) -> bool:
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
    if _is_removable_media_context(context):
        return True
    return _is_user_visible_lineage_target(target) and contains_any(context, TRANSFER_TOKENS)


def _is_user_visible_lineage_target(file_path: str) -> bool:
    normalized = normalize_path(file_path).lower()
    return bool(
        normalized
        and Path(normalized).suffix in _CLIPBOARD_DOCUMENT_EXTENSIONS
        and not any(marker in normalized for marker in _CLIPBOARD_HIDDEN_PATH_MARKERS)
    )


def _is_cloud_sync_directory_transfer(log, target: str, original: str, lineage: Lineage) -> bool:
    if log.event_type.lower() not in {
        "created",
        "renamed",
        "moved",
        "copied",
        "file_created",
        "file_renamed",
        "file_moved",
        "file_copied",
    }:
        return False
    normalized_target = normalize_path(target)
    lowered_target = normalized_target.lower()
    if not _is_user_visible_lineage_target(normalized_target):
        return False
    if not any(marker in lowered_target for marker in ("/onedrive/", "/wpsdrive/")):
        return False
    if _same_exact_file_path(normalized_target, original):
        return False
    root = lineage.root(lineage.resolve_artifact(normalized_target))
    return bool(root and same_file(root, original))


def _is_recent_document_save_as(
    event,
    target: str,
    source: str,
    recent_contexts: list[tuple[str, int, str, str]],
) -> bool:
    if event.event_type.lower() not in {"created", "modified", "renamed", "copied", "file_created", "file_modified"}:
        return False
    target_path = Path(normalize_path(target))
    source_path = Path(normalize_path(source))
    same_name = target_path.name.lower() == source_path.name.lower()
    cloud_sync_copy = same_name and _is_cloud_sync_user_path(target) and not _same_exact_file_path(target, source)
    if not target_path.name or not source_path.name or (same_name and not cloud_sync_copy):
        return False
    if target_path.suffix.lower() not in _CLIPBOARD_DOCUMENT_EXTENSIONS:
        return False
    event_time = _lineage_event_time_ms(event)
    nearby = [
        (context_source, context_time, title, process)
        for context_source, context_time, title, process in recent_contexts
        if context_source == source and 0 <= event_time - context_time <= _CLIPBOARD_SOURCE_HORIZON_MS
    ]
    if not nearby:
        return False
    title_text = normalize_path(event.window_title).lower()
    target_name = target_path.name.lower()
    source_name = source_path.name.lower()
    visible_target = target_name in title_text or target_path.stem.lower() in title_text
    visible_source = any(source_name in normalize_path(title).lower() for _, _, title, _ in nearby if title)
    same_parent = (
        _is_save_as_context(event)
        and target_path.parent.name.lower() == source_path.parent.name.lower()
    )
    same_stem = target_path.stem.lower() == source_path.stem.lower()
    event_process = (event.process_name or event.app_name or "").lower()
    same_process = bool(
        event_process
        and target_path.suffix.lower() == source_path.suffix.lower()
        and _is_lineage_authoring_process(event_process)
        and any(
            process == event_process and event_time - context_time <= 30_000
            for _, context_time, _, process in nearby
        )
    )
    return bool(
        visible_target
        or same_parent
        or same_stem
        or same_process
        or (visible_source and event_time - nearby[-1][1] <= 30_000)
    )


def _source_from_recent_document_context(
    event,
    target: str,
    recent_contexts: list[tuple[str, int, str, str]],
    sensitive_files: list[str],
    lineage: Lineage,
) -> str:
    if not _is_user_visible_lineage_target(target):
        return ""
    event_time = _lineage_event_time_ms(event)
    target_path = Path(normalize_path(target))
    candidates: list[tuple[int, int, str, str]] = []
    event_process = (event.process_name or event.app_name or "").lower()
    explicit_copy_context = contains_any(_event_search_text(event), ("copy", "copied", "复制"))
    for source, source_time, title, process in reversed(recent_contexts):
        elapsed = event_time - source_time
        horizon = _DOCUMENT_TITLE_SOURCE_HORIZON_MS if explicit_copy_context else _CLIPBOARD_SOURCE_HORIZON_MS
        if elapsed < 0 or elapsed > horizon:
            continue
        root = lineage.root(source) or source
        sensitive_root = _lookup_sensitive_source(
            root,
            _sensitive_lookup(tuple(sensitive_files)),
            allow_stem_reference=False,
        )
        if not sensitive_root:
            continue
        source_path = Path(normalize_path(source))
        if _same_exact_file_path(source, target):
            continue
        same_name = source_path.name.lower() == target_path.name.lower()
        explicit_copy = (
            event.event_type.lower() in {"created", "copied", "file_created", "file_copied"}
            and source_path.parent != target_path.parent
            and explicit_copy_context
        )
        if same_name and not (
            _is_cloud_sync_user_path(target)
            and _is_user_visible_lineage_target(source)
        ) and not explicit_copy:
            continue
        title_text = normalize_path(title).lower()
        same_parent = (
            event.event_type.lower() in {"created", "file_created", "renamed"}
            and _is_save_as_context(event)
            and source_path.parent.name.lower() == target_path.parent.name.lower()
        )
        visible_source = source_path.name.lower() in title_text
        same_stem = source_path.stem.lower() == target_path.stem.lower()
        same_process = bool(
            event_process
            and process == event_process
            and target_path.suffix.lower() == source_path.suffix.lower()
            and _is_lineage_authoring_process(event_process)
            and elapsed <= 30_000
        )
        if same_parent or same_stem or same_process:
            # Same-process/save-as and same-stem evidence is stronger than a
            # stale title-only mention from another candidate file.
            strength = 0 if (same_parent or same_stem or same_process) else 1
            candidates.append((strength, elapsed, source, sensitive_root))
    ordered = sorted(candidates)
    if not ordered:
        return ""
    best_score = ordered[0][:2]
    best = [item for item in ordered if item[:2] == best_score]
    unique_roots = _dedupe_paths([item[3] for item in best])
    return best[0][2] if len(unique_roots) == 1 else ""


def _is_cloud_sync_user_path(file_path: str) -> bool:
    lowered = normalize_path(file_path).lower()
    return any(marker in lowered for marker in ("/onedrive/", "/wpsdrive/"))


def _same_exact_file_path(left: str, right: str) -> bool:
    return bool(left and right and normalize_path(left).lower() == normalize_path(right).lower())


def _is_save_as_context(event) -> bool:
    return contains_any(
        _event_search_text(event).lower(),
        ("save_as", "save as", "另存为", "export", "导出"),
    )


def _is_lineage_authoring_process(process: str) -> bool:
    return contains_any(
        process.lower(),
        ("wps", "winword", "excel", "powerpnt", "notepad", "explorer"),
    )


def _lineage_artifact_mention(text: str, lineage: Lineage) -> str:
    """Resolve a concrete derived artifact named by a sink window/title."""

    normalized_text = normalize_path(text).lower()
    if not normalized_text:
        return ""
    matches = [
        normalize_path(artifact)
        for artifact in lineage.direct
        if Path(normalize_path(artifact)).name.lower()
        and Path(normalize_path(artifact)).name.lower() in normalized_text
    ]
    unique = _dedupe_paths(matches)
    return unique[0] if len(unique) == 1 else ""


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


def _source_from_recent_title_context(
    event,
    target: str,
    recent_titles: list[tuple[int, str]],
    sensitive_files: list[str],
) -> str:
    """Combine a recent exact title with a same-directory derived target."""

    normalized_target = normalize_path(target)
    if not _is_user_visible_lineage_target(normalized_target):
        return ""
    target_path = Path(normalized_target)
    event_time = _lineage_event_time_ms(event)
    matches: list[tuple[int, str]] = []
    for title_time, title in reversed(recent_titles):
        elapsed = event_time - title_time
        if elapsed < 0 or elapsed > _DOCUMENT_TITLE_SOURCE_HORIZON_MS:
            continue
        for sensitive in sensitive_files:
            source_path = Path(normalize_path(sensitive))
            if (
                source_path.parent.as_posix().lower() != target_path.parent.as_posix().lower()
                or not _mentions_exact_filename(title, sensitive)
                or not _is_generated_descendant_name(
                    target_path.stem.lower(),
                    source_path.stem.lower(),
                )
            ):
                continue
            matches.append((elapsed, sensitive))
    if not matches:
        return ""
    best_elapsed = min(elapsed for elapsed, _ in matches)
    best_sources = _dedupe_paths([source for elapsed, source in matches if elapsed == best_elapsed])
    return best_sources[0] if len(best_sources) == 1 else ""


def _source_from_derived_parent_alias(file_path: str, sensitive_files: list[str]) -> str:
    """Infer wrapped split outputs stored beside a source-named directory."""

    normalized = normalize_path(file_path)
    if not _looks_like_absolute_path(normalized):
        return ""
    candidate = Path(normalized)
    candidate_suffixes = [suffix.lower() for suffix in candidate.suffixes]
    if not candidate.name or not candidate_suffixes:
        return ""

    matches: list[str] = []
    for sensitive in sensitive_files:
        source = Path(normalize_path(sensitive))
        source_suffix = source.suffix.lower()
        if not source_suffix or source_suffix not in candidate_suffixes:
            continue
        expected_parent = normalize_path(source.with_suffix("")).lower()
        if normalize_path(candidate.parent).lower() == expected_parent:
            matches.append(sensitive)
    unique = _dedupe_paths(matches)
    return unique[0] if len(unique) == 1 else ""


def _source_from_same_stem_archive(file_path: str, sensitive_files: list[str]) -> str:
    """Bind an archive beside a sensitive file to that source conservatively."""

    normalized = normalize_path(file_path)
    if not _looks_like_absolute_path(normalized):
        return ""
    candidate = Path(normalized)
    if candidate.suffix.lower() not in {".zip", ".rar", ".7z"}:
        return ""
    matches = [
        sensitive
        for sensitive in sensitive_files
        if (
            Path(normalize_path(sensitive)).parent.as_posix().lower() == candidate.parent.as_posix().lower()
            and Path(normalize_path(sensitive)).stem.lower() == candidate.stem.lower()
        )
    ]
    unique = _dedupe_paths(matches)
    return unique[0] if len(unique) == 1 else ""


def _is_generated_descendant_name(candidate_stem: str, source_stem: str) -> bool:
    if not source_stem or candidate_stem == source_stem:
        return False
    return (
        candidate_stem.startswith(f"{source_stem}_")
        or candidate_stem.startswith(f"{source_stem} (")
        or candidate_stem.endswith(source_stem)
    )


def _correlated_operation_type(log, observation, text: str) -> str:
    if _is_sink_log(log):
        return "external_sink_interaction"
    if observation is not None and observation.operation_type:
        return _effective_visual_operation_type(observation)
    return operation_from_text(text, log.event_type)


def _correlated_app_name(log, observation) -> str:
    if _is_screen_share_context(log):
        text = normalize_path(f"{log.app_name} {log.window_title} {_event_search_text(log)}").lower()
        if "teams" in text:
            return "Microsoft Teams"
        if "zoom" in text:
            return "Zoom"
    frontend_name = _contextual_frontend_name(log)
    if frontend_name:
        return frontend_name
    return log.app_name or log.process_name or (observation.app_name if observation else "")


def _contextual_frontend_name(log) -> str:
    text = normalize_path(f"{log.app_name} {log.window_title}").lower()
    for markers, label in (
        (("飞书", "feishu", "lark"), "Feishu"),
        (("wechat", "微信"), "WeChat"),
        (("dingtalk", "钉钉"), "DingTalk"),
        (("poe",), "Poe"),
        (("chatgpt",), "ChatGPT"),
        (("gemini",), "Google Gemini"),
        (("claude",), "Claude"),
        (("豆包", "doubao"), "Doubao"),
    ):
        if contains_any(text, markers):
            return label
    return ""


def _contextual_sink_type(log) -> str:
    if _is_virtual_machine_context(log):
        return "virtual_machine"
    text = normalize_path(f"{log.app_name} {log.window_title}").lower()
    if contains_any(
        text,
        ("quark", "baidunetdisk", "baidu netdisk", "夸克网盘", "百度网盘", "网盘", "cloud drive", "cloud storage"),
    ):
        return "cloud_sync"
    if contains_any(text, ("飞书", "feishu", "lark", "wechat", "微信", "dingtalk", "钉钉", "qq(浏览)")):
        return "chat_upload"
    if contains_any(text, ("poe", "chatgpt", "gemini", "claude", "deepseek", "kimi", "豆包")):
        return "ai_chat"
    return ""


def _visual_join_reasons(observation, original: str) -> list[str]:
    reasons = ["visual_only"]
    if original:
        reasons.append("visual_mentions_sensitive_file")
    text = _observation_search_text(observation)
    if observation.operation_type == "external_sink_interaction" or contains_any(text, SINK_TOKENS):
        reasons.append("visual_sink_context")
    if _is_transfer_observation(observation):
        reasons.append("visual_transfer_context")
    status = _observation_action_status(observation)
    if status != "unknown":
        reasons.append(f"action_status:{status}")
    sink_type = _observation_sink_type(observation)
    if sink_type:
        reasons.append(f"sink_type:{sink_type}")
    declared = _declared_visual_behavior(observation)
    if declared:
        reasons.append(f"visual_declared_behavior:{declared}")
    if _is_unexecuted_visual_preparation(observation):
        reasons.append("visual_unexecuted_preparation")
    if _is_unproven_cloud_folder_claim(observation):
        reasons.append("visual_unproven_cloud_folder")
    return reasons


def _is_visual_file_split(observation) -> bool:
    text = _observation_search_text(observation).lower()
    return any(marker in text for marker in ("pdf split", "split pdf", "file split", "拆分pdf", "pdf拆分", "文件拆分"))


def _observation_timestamp(observation, recording_start_ms: int) -> str:
    timestamp_ms = int(observation.start_ms or 0)
    if timestamp_ms > 10_000_000_000:
        return str(timestamp_ms)
    if recording_start_ms:
        return str(recording_start_ms + timestamp_ms)
    return ""


def _screen_share_state_reasons(observation, logs) -> list[str]:
    """Attach the observed start of an active sharing state to visual evidence."""

    if _observation_sink_type(observation) != "screen_share":
        return []
    center = observation.start_ms if not observation.end_ms else (observation.start_ms + observation.end_ms) // 2
    app = normalize_path(observation.app_name).lower()
    starts = [
        log.timestamp_ms
        for log in logs
        if log.timestamp_ms
        and log.timestamp_ms <= center
        and _is_screen_share_context(log)
        and (not app or app in normalize_path(_correlated_app_name(log, None)).lower())
    ]
    return [f"screen_share_started_at:{min(starts)}"] if starts else []


def _choose_identity_binding(bindings: list[tuple[int, int, str, str, Any]]):
    if not bindings:
        return None
    ordered = sorted(bindings, key=lambda item: (item[0], item[1], -float(getattr(item[4], "confidence", 0.0))))
    best = ordered[0]
    for candidate in ordered[1:]:
        if same_file(candidate[2], best[2]):
            continue
        if candidate[0] - best[0] < 1_500:
            return None
        break
    return best


def _app_compatibility_rank(left: str, right: str) -> int:
    left_text = str(left or "").strip().lower()
    right_text = str(right or "").strip().lower()
    if left_text and right_text and left_text == right_text:
        return 0
    left_category = identify_frontend_app(app_name=left, visual_text=left).category
    right_category = identify_frontend_app(app_name=right, visual_text=right).category
    if left_category == right_category and left_category != "unknown":
        return 0
    generic = {"unknown", "browser", "document_editor"}
    if left_category in generic or right_category in generic:
        return 1
    return 2


def _observation_action_status(observation) -> str:
    match = re.search(r"\baction_status=(selected|submitted|in_progress|completed|failed|unknown)\b", observation.description.lower())
    return match.group(1) if match else "unknown"


def _declared_visual_behavior(observation) -> str:
    match = re.match(
        r"\s*(normal|direct_leak|hidden_transfer|unknown_risk)\s*:",
        observation.description.lower(),
    )
    return match.group(1) if match else ""


def _is_unexecuted_visual_preparation(observation) -> bool:
    if (
        _declared_visual_behavior(observation) != "hidden_transfer"
        or _observation_action_status(observation) != "unknown"
    ):
        return False
    text = _observation_search_text(observation).lower()
    return any(
        marker in text
        for marker in (
            "context_menu",
            "not confirmed",
            "not executed",
            "preparation",
            "仅准备",
            "未确认",
            "未执行",
        )
    )


def _declared_visual_action(observation) -> str:
    match = re.match(r"\s*(?:normal|direct_leak|hidden_transfer|unknown_risk)\s*:\s*([^.]+)", observation.description.lower())
    if not match:
        return ""
    return re.sub(r"[\s-]+", "_", match.group(1).strip())


def _effective_visual_operation_type(observation) -> str:
    if _is_confirmed_external_observation(observation):
        return "external_sink_interaction"
    if _declared_visual_behavior(observation) == "direct_leak":
        if (
            observation.operation_type == "external_sink_interaction"
            and _observation_action_status(observation) in {"submitted", "in_progress", "completed"}
            and _observation_sink_type(observation) not in {"", "unknown", "none", "null", "n/a", "na"}
        ):
            # The parser has already normalized the structured VLM event. Do
            # not downgrade it merely because the model used a novel action
            # label such as "git push upload" or "文件上传/导入".
            return "external_sink_interaction"
        action = _declared_visual_action(observation)
        if action in _DECLARED_VISUAL_OUTBOUND_ACTIONS or _is_explicit_declared_outbound_action(action):
            return "external_sink_interaction"
        return "file_or_content_transfer"
    return observation.operation_type


def _is_confirmed_external_observation(observation) -> bool:
    if _observation_action_status(observation) not in {"submitted", "in_progress", "completed"}:
        return False
    if _observation_sink_type(observation) in {"", "unknown", "none", "null", "n/a", "na"}:
        return False
    action = _declared_visual_action(observation)
    if action in _DECLARED_VISUAL_OUTBOUND_ACTIONS or _is_explicit_declared_outbound_action(action):
        return True
    text = observation.description.lower()
    return any(
        marker in text
        for marker in (
            "pasted into",
            "pasted to",
            "sends it to",
            "sent it to",
            "submitted to",
            "uploaded to",
            "sent to",
            "synced to",
            "transferring sensitive data to",
            "粘贴到",
            "提交到",
            "上传到",
            "发送到",
            "同步到",
        )
    )


def _suppress_conflicting_screen_share_observations(observations):
    local_recordings = [
        observation
        for observation in observations
        if _declared_visual_action(observation) in {"screen_recording", "screen_record", "record_screen"}
        and any(
            marker in observation.description.lower()
            for marker in ("mp4", "screen recording", "录屏", "屏幕录制")
        )
    ]
    if not local_recordings:
        return observations
    result = []
    for observation in observations:
        action = _declared_visual_action(observation)
        text = observation.description.lower()
        resource_names = _observation_resource_stems(observation)
        conflicted = (
            action in {"screen_share", "share_screen"}
            and any(marker in text for marker in ("mp4", "screen recording", "录屏", "屏幕录制"))
            and not _has_independent_screen_share_evidence(text)
            and any(
                abs(recording.start_ms - observation.start_ms) <= 10_000
                and (
                    not resource_names
                    or bool(resource_names & _observation_resource_stems(recording))
                )
                for recording in local_recordings
            )
        )
        if not conflicted:
            result.append(observation)
    return result


def _suppress_redundant_cloud_sync_source_events(events, lineage: Lineage):
    explicit_targets = [
        event
        for event in events
        if "cloud_sync_directory_transfer" in event.join_reasons
        and event.original_file
        and event.current_file
        and not _same_exact_file_path(event.original_file, event.current_file)
    ]
    if not explicit_targets:
        return events
    result = []
    for event in events:
        redundant = (
            event.event_type == "visual_observation"
            and "sink_type:cloud_sync" in event.join_reasons
            and _same_exact_file_path(event.original_file, event.current_file)
            and any(
                _same_exact_file_path(target.original_file, event.original_file)
                and _same_exact_file_path(lineage.root(target.current_file), event.original_file)
                and abs(parse_timestamp_ms(target.timestamp) - parse_timestamp_ms(event.timestamp)) <= 60_000
                for target in explicit_targets
            )
        )
        if not redundant:
            result.append(event)
    return result


def _has_independent_screen_share_evidence(text: str) -> bool:
    return any(
        marker in text
        for marker in (
            "sharing toolbar",
            "share toolbar",
            "sharing banner",
            "share banner",
            "active share indicator",
            "remote participant",
            "共享工具栏",
            "共享横幅",
            "正在共享",
            "远端参会者",
        )
    )


def _observation_resource_stems(observation) -> set[str]:
    return {
        stem
        for resource in (observation.resource, *observation.related_resources)
        if (stem := Path(normalize_path(resource)).stem.lower()) not in {"", "unknown", "未知"}
    }


def _is_explicit_declared_outbound_action(action: str) -> bool:
    """Accept common status suffixes without accepting vague AI-processing claims."""

    if not action or "processing" in action or "transform" in action:
        return False
    return action.startswith(
        (
            "upload_",
            "email_",
            "file_",
            "chat_",
            "send_",
            "web_",
            "network_",
            "http_",
            "copy_",
            "paste_",
            "publish_",
            "article_publish_",
            "folder_sync_",
            "ai_prompt_",
            "ai_chat_",
            "cloud_",
            "screen_",
            "share_",
            "commit_",
        )
    )


def _observation_sink_type(observation) -> str:
    match = re.search(r"\bsink_type=([a-z_]+)\b", observation.description.lower())
    return match.group(1) if match else ""


def _is_unproven_cloud_folder_claim(observation) -> bool:
    if _observation_sink_type(observation) != "cloud_sync":
        return False
    text = observation.description.lower()
    membership_claim = any(
        marker in text
        for marker in (
            "automatically synced",
            "automatically sync",
            "files in the onedrive folder",
            "visible in file explorer within a onedrive",
            "within a onedrive synced folder",
            "onedrive synced folder path",
            "showing sync status icons",
            "位于onedrive",
            "位于 onedrive",
            "同步目录中的文件会自动",
        )
    )
    dynamic_status = any(
        marker in text
        for marker in (
            "upload progress",
            "sync progress",
            "sync completed",
            "syncing",
            "同步进度",
            "同步完成",
            "正在同步",
        )
    )
    if any(
        marker in text
        for marker in (
            "no file-specific sync status",
            "no sync status",
            "without a sync status",
            "未显示同步状态",
            "没有同步状态",
        )
    ):
        dynamic_status = False
    return membership_claim and not dynamic_status


def _behavior_with_action_status(observation, fallback: str) -> str:
    if observation is None:
        return fallback
    status = _observation_action_status(observation)
    if status == "failed":
        return "failed_external_attempt"
    if status == "selected":
        return "selected_external_attempt"
    return fallback


def _description_evidence_frame_ids(description: str) -> tuple[str, ...]:
    values: list[str] = []
    for match in re.finditer(r"(?:evidence_frame_ids|visual_identity_frame_ids)=([^\s.]+)", description):
        values.extend(item for item in match.group(1).split("|") if item)
    return tuple(dict.fromkeys(values))


def _observation_evidence_refs(observation) -> tuple[str, ...]:
    refs = [f"frame:{observation.observation_id}"]
    refs.extend(f"frame:{frame_id}" for frame_id in _description_evidence_frame_ids(observation.description))
    for marker, prefix in (("visual_identity", "frame"), ("log_identity", "log")):
        match = re.search(rf"\b{marker}=([A-Za-z0-9_.:-]+)", observation.description)
        if match:
            refs.append(f"{prefix}:{match.group(1).rstrip('.')}")
    return tuple(dict.fromkeys(refs))


def _mentions_file(text: str, file_path: str) -> bool:
    normalized_text = normalize_path(text).lower()
    normalized_file = normalize_path(file_path).lower()
    if not normalized_text or not normalized_file:
        return False
    name = Path(normalized_file).name.lower()
    stem = Path(name).stem.lower()
    return normalized_file in normalized_text or (name and name in normalized_text) or (len(stem) >= 4 and stem in normalized_text)


def _mentions_exact_filename(text: str, file_path: str) -> bool:
    normalized_text = normalize_path(text).lower()
    name = Path(normalize_path(file_path)).name.lower()
    return bool(name and Path(name).suffix and name in normalized_text)


def _mentions_source_directory(text: str, file_path: str) -> bool:
    """Match a parent folder as a token, not as a filename extension."""

    normalized_text = normalize_path(text).lower()
    directory = Path(normalize_path(file_path)).parent.name.lower()
    suffix = Path(normalize_path(file_path)).suffix.lower().lstrip(".")
    if not normalized_text or not directory or directory == suffix:
        return False
    return bool(re.search(rf"(?<![\w.]){re.escape(directory)}(?![\w.])", normalized_text))


def _best_sensitive_mention(text: str, sensitive_files: list[str]) -> str:
    best = ""
    best_score = (0, 0, 0)
    normalized_text = normalize_path(text).lower()
    for sensitive in sensitive_files:
        normalized_sensitive = normalize_path(sensitive).lower()
        if not normalized_sensitive:
            continue
        name = Path(normalized_sensitive).name.lower()
        stem = Path(name).stem.lower()
        if same_file(text, sensitive):
            strength = 5
        elif _matches_sensitive_file_reference(text, sensitive):
            strength = 4
        elif normalized_sensitive in normalized_text:
            strength = 3
        elif name and name in normalized_text:
            strength = 2
        elif len(stem) >= 4 and stem in normalized_text and not stem.isascii():
            # An ASCII stem such as ``customer`` or ``secret`` commonly
            # appears as prose or as a VLM translation. Without the extension
            # or an exact path it is not file identity. Non-ASCII stems retain
            # their literal screen text value (for example 客户联系方式).
            strength = 1
        else:
            continue
        context_score = _source_context_score(normalized_text, normalized_sensitive)
        score = (strength + context_score, len(stem), len(name))
        if score > best_score:
            best = sensitive
            best_score = score
    return best


def _partial_visual_filename_identity(observation, logs, sensitive_files: list[str]) -> str:
    """Resolve a visibly truncated filename only within a unique user scope."""

    text = normalize_path(observation.description).lower()
    partial_match = re.search(
        r"(?:partial text|filename (?:starts? with|prefix)|部分文字|文件名前缀)\s*[`'\"]([^`'\"]+?)(?:\.\.\.|\u2026)[`'\"]",
        text,
    )
    extension_match = re.search(r"(?<![\w])\.(docx?|pdf|xlsx?|pptx?|txt|sql|zip|rar)(?![\w])", text)
    if not partial_match or not extension_match:
        return ""
    prefix = partial_match.group(1).strip().lower()
    extension = f".{extension_match.group(1).lower()}"
    if len(prefix) < 2:
        return ""

    usernames = {
        str((getattr(log, "raw", {}) or {}).get("user_info", {}).get("username", "")).strip().lower()
        for log in logs
    }
    usernames.discard("")
    if not usernames:
        return ""

    matches = []
    for sensitive in sensitive_files:
        normalized = normalize_path(sensitive)
        lowered = normalized.lower()
        name = Path(normalized).name
        if Path(name).suffix.lower() != extension or not Path(name).stem.lower().startswith(prefix):
            continue
        if not any(f"/users/{username}/" in lowered for username in usernames):
            continue
        matches.append(normalized)
    unique = _dedupe_paths(matches)
    return unique[0] if len(unique) == 1 else ""


def _source_context_score(text: str, sensitive: str) -> int:
    """Use explicit folder/user cues to disambiguate same-named sources."""

    score = 0
    pairs = (("onedrive", "onedrive"), ("wps cloud", "wps cloud"), ("wpsdrive", "wpsdrive"))
    for text_marker, path_marker in pairs:
        if text_marker in text and path_marker in sensitive:
            score += 2
    if ("桌面" in text or "desktop" in text) and "/desktop/" in sensitive:
        score += 2
    if "图片" in text and "/图片/" in sensitive:
        score += 1
    source = Path(sensitive)
    if source.suffix and source.stem:
        descendant_with_same_type = re.search(
            rf"{re.escape(source.stem)}(?:[_ (][^/\\\s]*)?{re.escape(source.suffix)}\b",
            text,
        )
        if descendant_with_same_type:
            score += 2
    for username in re.findall(r"(?:users|用户)[/：: ]+([^/\\ >]+)", text):
        if f"/users/{username.lower()}/" in sensitive:
            score += 1
    return score


def _looks_like_absolute_path(value: str) -> bool:
    normalized = normalize_path(value)
    return bool(re.match(r"^[A-Za-z]:/", normalized) or normalized.startswith("/"))


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


def _is_source_label_only(candidate: str, original: str) -> bool:
    normalized = normalize_path(candidate)
    if not normalized or _looks_like_absolute_path(normalized) or "/" in normalized:
        return False
    label = Path(normalized).name.lower()
    source_name = Path(normalize_path(original)).name.lower()
    return bool(label and source_name and label in {source_name, Path(source_name).stem.lower()})


def _is_explicit_remote_output(observation, candidate: str, original: str) -> bool:
    if not candidate or not original or not _is_source_label_only(candidate, original):
        return False
    action = _declared_visual_action(observation)
    text = observation.description.lower()
    remote_context = any(
        marker in text
        for marker in ("web ide", "web repository", "remote repository", "gitlab", "github", "jihulab")
    )
    created_output = any(
        marker in text
        for marker in ("created it as file", "created as file", "pasted/created", "pasted into a new file")
    )
    return remote_context and created_output and action.startswith(("copy_", "paste_", "commit", "upload_"))


def _visual_removable_destination(observation, original: str) -> str:
    if not original or _observation_sink_type(observation) != "removable_media":
        return ""
    if _observation_action_status(observation) != "completed":
        return ""
    text = observation.description.lower()
    if not any(marker in text for marker in ("usb", "removable", "u 盘", "u盘", "移动存储")):
        return ""
    if not any(marker in text for marker in ("copied to", "appears in the destination", "复制到", "目标目录")):
        return ""
    source_match = re.match(r"^([a-z]):/", normalize_path(original).lower())
    source_drive = source_match.group(1) if source_match else ""
    drives = [match.lower() for match in re.findall(r"(?<![a-z0-9])([a-z]):", text)]
    targets = [drive for drive in drives if drive != source_drive]
    if len(set(targets)) != 1:
        return ""
    return f"{targets[0].upper()}:/{Path(normalize_path(original)).name}"


def _explicit_visual_derivation(
    observation,
    logs,
    sensitive_files: list[str],
    lineage: Lineage,
) -> tuple[str, str] | None:
    text = observation.description.lower()
    extracted_save = (
        any(marker in text for marker in ("extracted sensitive text", "extracted text", "提取的敏感文本", "提取文字"))
        and any(marker in text for marker in ("save as", "saved as", "being saved", "另存为", "保存为"))
    )
    explicit_source_derivation = any(
        marker in text
        for marker in ("derived file", "derived from", "screenshot of", "源自", "派生文件", "截图来自")
    )
    if not explicit_source_derivation and not extracted_save:
        return None
    target_name = Path(normalize_path(observation.resource)).name.lower()
    if not target_name or target_name in {"unknown", "未知"}:
        return None
    target_path = Path(target_name)
    targets = _dedupe_paths([
        log.file_path
        for log in logs
        if _looks_like_absolute_path(normalize_path(log.file_path))
        and (
            Path(normalize_path(log.file_path)).name.lower() == target_name
            or (
                extracted_save
                and target_path.suffix
                and Path(normalize_path(log.file_path)).suffix.lower() == target_path.suffix.lower()
                and Path(normalize_path(log.file_path)).stem.lower().startswith(target_path.stem.lower())
                and _is_cloud_sync_user_path(log.file_path)
            )
        )
    ])
    if len(targets) != 1:
        return None
    target = targets[0]
    if extracted_save:
        source_artifacts = _dedupe_paths([
            lineage.resolve_artifact(reference)
            for reference in observation.related_resources
            if normalize_path(reference)
            and not _same_exact_file_path(reference, observation.resource)
            and lineage.root(lineage.resolve_artifact(reference)) != lineage.resolve_artifact(reference)
        ])
        if len(source_artifacts) == 1:
            return target, source_artifacts[0]
    mentioned_sources = _dedupe_paths(
        [
            sensitive
            for sensitive in sensitive_files
            if not same_file(sensitive, target) and _mentions_exact_filename(observation.description, sensitive)
        ]
    )
    same_parent_sources = [
        source
        for source in mentioned_sources
        if Path(normalize_path(source)).parent == Path(normalize_path(target)).parent
    ]
    sources = same_parent_sources or mentioned_sources
    if sources:
        longest_name = max(len(Path(normalize_path(source)).name) for source in sources)
        sources = [source for source in sources if len(Path(normalize_path(source)).name) == longest_name]
    return (target, sources[0]) if len(sources) == 1 else None


def _has_ambiguous_sensitive_label(observation, sensitive_files: list[str]) -> bool:
    references = [observation.resource, *observation.related_resources]
    if any(_looks_like_absolute_path(normalize_path(reference)) for reference in references):
        return False
    matches = {
        normalize_path(sensitive).lower()
        for reference in references
        for sensitive in sensitive_files
        if normalize_path(reference)
        and Path(normalize_path(reference)).name.lower()
        in {
            Path(normalize_path(sensitive)).name.lower(),
            Path(normalize_path(sensitive)).stem.lower(),
        }
    }
    return len(matches) > 1


def _is_unbound_visual_risk(observation, text: str) -> bool:
    normalized = text.lower()
    return (
        observation.operation_type in {"external_sink_interaction", "file_or_content_transfer"}
        and any(marker in normalized for marker in ("direct_leak", "hidden_transfer", "unknown_risk"))
    )


def _is_visual_derived_candidate(file_path: str, original: str) -> bool:
    normalized = normalize_path(file_path).strip().strip("\"'")
    lowered = normalized.lower()
    if (
        not lowered
        or lowered in {"unknown", "未知", "n/a", "na", "none", "null", "-"}
        or "..." in lowered
        or "…" in lowered
    ):
        return False
    if same_file(normalized, original) or _matches_sensitive_file_reference(normalized, original):
        return False
    return True


def _unique_cloud_sync_descendant(original: str, lineage: Lineage) -> str:
    original_key = normalize_path(original).lower()
    candidates = {
        normalize_path(derived)
        for derived in lineage.direct
        if "/" in normalize_path(derived)
        and normalize_path(lineage.root(derived)).lower() == original_key
        and any(
            marker in normalize_path(derived).lower()
            for marker in ("/onedrive/", "/wps cloud files/", "/wpsdrive/")
        )
    }
    return next(iter(candidates)) if len(candidates) == 1 else ""


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
    if not matched and "/screenmonitor/winows_monitor/" in normalized:
        matched = exact.get(normalized.replace("/screenmonitor/winows_monitor/", "/screenmonitor/windows_monitor/"))
    if matched:
        return matched
    if _looks_like_absolute_path(normalized) or "/" in normalized:
        return ""
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


