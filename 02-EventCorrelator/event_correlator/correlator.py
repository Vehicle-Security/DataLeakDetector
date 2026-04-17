from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta

from .config import EventCorrelatorConfig
from .dedup import deduplicate_correlated_events, deduplicate_upload_candidates
from .lineage import LineageBuilder, LineageState
from .schema import CorrelationBundle, CorrelatedEvent, EventCorrelatorInput, UploadCandidate
from .timeline import TimelineNormalizer
from .utils import (
    choose_non_empty,
    extract_resource_tokens,
    get_path_basename,
    is_unknown_resource,
    normalize_app_name,
    normalize_file_path,
    parse_timestamp,
)


EXFILTRATION_KEYWORDS = (
    "upload",
    "send",
    "share",
    "mail",
    "email",
    "attachment",
    "上传",
    "发送",
    "分享",
    "附件",
    "外发",
    "邮箱",
    "邮件",
)

TRANSFORM_KEYWORDS = (
    "rename",
    "convert",
    "split",
    "compress",
    "archive",
    "分片",
    "转换",
    "压缩",
)

SCENE_MARKERS = [
    "share",
    "screen",
    "meeting",
    "upload",
    "send",
    "mail",
    "共享",
    "屏幕",
    "会议",
    "上传",
    "发送",
    "外发",
    "腾讯会议",
    "共享屏幕",
    "屏幕共享",
]

EXPLICIT_SINK_TERMS = (
    "send",
    "upload",
    "attachment",
    "share",
    "screen",
    "meeting",
    "mail",
    "email",
    "发送",
    "上传",
    "附件",
    "共享",
    "屏幕",
    "会议",
    "外发",
    "邮件",
    "邮箱",
)

SCREEN_SHARE_MARKERS = ["screen", "share", "meeting", "共享", "屏幕", "会议", "腾讯会议", "共享屏幕", "屏幕共享"]
MAIL_MARKERS = ["mail", "email", "attachment", "邮箱", "邮件", "附件", "qq邮箱", "qqmail"]
CHAT_MARKERS = ["chat", "qq", "wechat", "聊天", "微信"]
CLOUD_MARKERS = ["cloud", "drive", "sync", "云盘"]
PRECURSOR_BROWSER_APPS = ("edge", "chrome", "browser", "qq邮箱", "qqmail")


@dataclass
class MatchResult:
    correlation_score: float
    matched: bool
    ambiguous: bool = False


class EventCorrelator:
    def __init__(self, config: EventCorrelatorConfig | None = None):
        self.config = config or EventCorrelatorConfig()
        self.timeline_normalizer = TimelineNormalizer()
        self.lineage_builder = LineageBuilder(max_depth=self.config.max_lineage_depth)

    def run(self, payload: EventCorrelatorInput) -> CorrelationBundle:
        correlation_config = self._merge_runtime_config(payload.get("correlation_config", {}))
        context = self.timeline_normalizer.normalize(payload, correlation_config)
        lineage_state = self.lineage_builder.build(context)

        correlated_events = self._build_correlated_events(context, lineage_state)
        correlated_events = deduplicate_correlated_events(correlated_events)
        correlated_events = self._suppress_precursor_ambiguous_events(correlated_events)

        upload_candidates = self._build_upload_candidates(context, lineage_state, correlated_events)
        upload_candidates = deduplicate_upload_candidates(upload_candidates)

        operation_records = self._build_operation_records(correlated_events)
        file_lineage = lineage_state.export()

        analysis_status = "success"
        if context.errors:
            analysis_status = "partial_success"
        if not correlated_events and not upload_candidates:
            analysis_status = "no_match" if not context.errors else "partial_success"

        return CorrelationBundle(
            session_id=context.session_id,
            analysis_status=analysis_status,
            correlated_events=correlated_events,
            operation_records=operation_records,
            upload_candidates=upload_candidates,
            file_lineage=file_lineage,
            statistics={
                "log_events_input": len(context.normalized_logs),
                "frame_segments_input": len(context.normalized_segments),
                "correlated_events_output": len(correlated_events),
                "upload_candidates_output": len(upload_candidates),
                "lineage_direct_mappings": len(file_lineage["direct_file_mappings"]),
                "lineage_full_chains": len(file_lineage["full_file_mapping_chains"]),
            },
            errors=context.errors,
        )

    def _merge_runtime_config(self, runtime_config: dict) -> dict:
        merged = self.config.as_dict()
        if isinstance(runtime_config, dict):
            merged.update(runtime_config)
        return merged

    def _build_correlated_events(self, context, lineage_state: LineageState) -> list[CorrelatedEvent]:
        correlated_events: list[CorrelatedEvent] = []
        tolerance = timedelta(seconds=self.config.time_window_tolerance_seconds)

        for index, log_event in enumerate(context.normalized_logs):
            matching_segments = []
            for segment in context.normalized_segments:
                match_result = self._score_log_segment_match(log_event, segment, tolerance)
                if match_result.matched:
                    matching_segments.append((segment, match_result))

            if matching_segments:
                matching_segments.sort(key=lambda item: item[1].correlation_score, reverse=True)
                segment, match_result = matching_segments[0]
            else:
                segment, match_result = None, MatchResult(correlation_score=0.0, matched=False)

            original_file = lineage_state.resolve_root(log_event.file_path, self.config.max_lineage_depth)
            if not original_file and log_event.file_path in context.sensitive_files:
                original_file = log_event.file_path

            if not original_file and not segment:
                continue

            correlated_events.append(
                CorrelatedEvent(
                    event_id=f"corr_log_{index}",
                    session_id=context.session_id,
                    timestamp=log_event.timestamp_text,
                    event_type=log_event.event_type,
                    source_type="log+segment" if segment else "log",
                    original_file=original_file,
                    current_file=log_event.file_path,
                    app_name=choose_non_empty(
                        segment.app_name if segment else "",
                        log_event.app_name,
                        log_event.process_name,
                    ),
                    operation_type=choose_non_empty(
                        segment.operation_type if segment else "",
                        log_event.event_type,
                    ),
                    behavior_category=self._infer_behavior_category(
                        log_event.event_type,
                        segment.operation_type if segment else "",
                    ),
                    evidence_refs=self._build_evidence_refs(log_event.event_id, segment.segment_id if segment else ""),
                    confidence=self._compute_confidence(segment.confidence if segment else 0.0, match_result),
                    correlation_score=match_result.correlation_score,
                    status="ambiguous" if match_result.ambiguous else "linked",
                    object_binding=self._build_event_object_binding(
                        context=context,
                        log_event=log_event,
                        segment=segment,
                        original_file=original_file,
                        lineage_state=lineage_state,
                    ),
                )
            )

        for index, segment in enumerate(context.normalized_segments):
            segment_ref = f"segment:{segment.segment_id}"
            already_matched = any(segment_ref in event["evidence_refs"] for event in correlated_events)
            if already_matched:
                continue

            original_file = self._resolve_segment_original_file(segment, lineage_state, context.sensitive_files)
            if not original_file:
                continue

            timestamp = segment.supporting_timestamps[0] if segment.supporting_timestamps else segment.time_range
            correlated_events.append(
                CorrelatedEvent(
                    event_id=f"corr_segment_{index}",
                    session_id=context.session_id,
                    timestamp=timestamp,
                    event_type="frame_segment",
                    source_type="segment",
                    original_file=original_file,
                    current_file=choose_non_empty(segment.primary_resource, original_file),
                    app_name=segment.app_name,
                    operation_type=segment.operation_type,
                    behavior_category=self._infer_behavior_category("frame_segment", segment.operation_type),
                    evidence_refs=[segment_ref],
                    confidence=max(0.0, min(segment.confidence, 1.0)),
                    correlation_score=max(0.0, min(segment.confidence, 1.0)),
                    status="segment_only",
                    object_binding=self._build_segment_object_binding(
                        context=context,
                        segment=segment,
                        original_file=original_file,
                        lineage_state=lineage_state,
                    ),
                )
            )

        return correlated_events

    def _suppress_precursor_ambiguous_events(self, events: list[CorrelatedEvent]) -> list[CorrelatedEvent]:
        if not events:
            return events

        strong_future_keys: set[tuple[str, str, str]] = set()
        for event in events:
            event_type = str(event["event_type"] or "").strip().lower()
            if event_type not in {"file_upload", "file_selected", "upload_detected"}:
                continue
            segment_refs = tuple(
                sorted(ref for ref in event.get("evidence_refs", []) or [] if str(ref).startswith("segment:"))
            )
            strong_future_keys.add(
                (
                    normalize_file_path(str(event["original_file"] or "")).lower(),
                    str(
                        event.get(
                            "sink_type",
                            self._infer_sink_type(event["operation_type"], event["app_name"]),
                        )
                        or ""
                    ).strip().lower(),
                    "|".join(segment_refs),
                )
            )

        filtered: list[CorrelatedEvent] = []
        for event in events:
            if event.get("status") != "ambiguous":
                filtered.append(event)
                continue

            event_type = str(event["event_type"] or "").strip().lower()
            if event_type not in {"created", "modified"}:
                filtered.append(event)
                continue

            segment_refs = tuple(
                sorted(ref for ref in event.get("evidence_refs", []) or [] if str(ref).startswith("segment:"))
            )
            suppression_key = (
                normalize_file_path(str(event["original_file"] or "")).lower(),
                self._infer_sink_type(event["operation_type"], event["app_name"]),
                "|".join(segment_refs),
            )
            if suppression_key in strong_future_keys:
                continue

            filtered.append(event)

        return filtered

    def _score_log_segment_match(self, log_event, segment, tolerance: timedelta) -> MatchResult:
        score = 0.0
        matched_axes = 0
        matched_features = set()

        if segment.start_time and segment.end_time:
            lower = segment.start_time - tolerance
            upper = segment.end_time + tolerance
            if lower <= log_event.timestamp <= upper:
                score += 0.35
                matched_axes += 1
                matched_features.add("time")

        log_tokens = extract_resource_tokens(log_event.file_path, log_event.file_name)
        segment_tokens = extract_resource_tokens(
            segment.primary_resource,
            segment.related_resources,
            segment.visible_evidence,
        )
        if log_tokens and segment_tokens and log_tokens.intersection(segment_tokens):
            score += 0.35
            matched_axes += 1
            matched_features.add("resource")

        log_app = normalize_app_name(log_event.app_name).lower()
        segment_app = normalize_app_name(segment.app_name).lower()
        if log_app and segment_app and (log_app in segment_app or segment_app in log_app):
            score += 0.15
            matched_axes += 1
            matched_features.add("app")

        operation_text = f"{segment.operation_type} {segment.action_description}".lower()
        if log_event.event_type and log_event.event_type.lower() in operation_text:
            score += 0.05
            matched_axes += 1
            matched_features.add("operation")

        score += max(0.0, min(segment.confidence, 1.0)) * 0.1

        if matched_axes >= 2:
            return MatchResult(correlation_score=score, matched=True, ambiguous=False)

        if (
            matched_axes == 1
            and self.config.allow_ambiguous_candidates
            and score >= 0.3
            and ("resource" in matched_features or "app" in matched_features)
        ):
            return MatchResult(correlation_score=score, matched=True, ambiguous=True)

        return MatchResult(correlation_score=score, matched=False, ambiguous=False)

    def _compute_confidence(self, segment_confidence: float, match_result: MatchResult) -> float:
        base = max(0.0, min(segment_confidence, 1.0))
        if match_result.matched:
            blended = (base + match_result.correlation_score) / 2.0
        else:
            blended = base
        if match_result.ambiguous:
            blended *= 0.7
        return max(0.0, min(blended, 1.0))

    def _infer_behavior_category(self, event_type: str, operation_type: str) -> str:
        combined = f"{event_type} {operation_type}".lower()
        if any(keyword in combined for keyword in EXFILTRATION_KEYWORDS):
            return "data_exfiltration_candidate"
        if any(keyword in combined for keyword in TRANSFORM_KEYWORDS):
            return "hidden_transformation_candidate"
        return "activity_observation"

    def _build_evidence_refs(self, log_event_id: str, segment_id: str) -> list[str]:
        refs = [f"log:{log_event_id}"]
        if segment_id:
            refs.append(f"segment:{segment_id}")
        return refs

    def _resolve_segment_original_file(self, segment, lineage_state: LineageState, sensitive_files: list[str]) -> str:
        resource_candidates = [segment.primary_resource, *segment.related_resources]
        for resource in resource_candidates:
            normalized = normalize_file_path(resource)
            if not normalized or is_unknown_resource(normalized):
                continue
            resolved = lineage_state.resolve_root(normalized, self.config.max_lineage_depth)
            if resolved:
                return resolved

            resource_name = get_path_basename(normalized).lower()
            for sensitive_file in sensitive_files:
                if resource_name == get_path_basename(sensitive_file).lower():
                    return sensitive_file

        if len(sensitive_files) == 1:
            combined = f"{segment.app_name} {segment.operation_type} {segment.action_description}".lower()
            if any(marker in combined for marker in SCENE_MARKERS):
                return sensitive_files[0]
        return ""

    def _build_upload_candidates(
        self,
        context,
        lineage_state: LineageState,
        correlated_events: list[CorrelatedEvent],
    ) -> list[UploadCandidate]:
        candidates: list[UploadCandidate] = []
        upload_keywords = tuple(keyword.lower() for keyword in self.config.upload_operation_keywords)
        upload_event_types = {item.lower() for item in self.config.upload_event_types}
        segment_lookup = {segment.segment_id: segment for segment in context.normalized_segments}

        for index, event in enumerate(correlated_events):
            combined = f"{event['event_type']} {event['operation_type']} {event['behavior_category']}".lower()
            if not self._event_represents_actual_sink_interaction(event):
                continue
            if not any(keyword in combined for keyword in upload_keywords) and event["event_type"].lower() not in upload_event_types:
                continue

            current_files = self._collect_candidate_files(event, segment_lookup, lineage_state)
            current_files = [
                item
                for item in current_files
                if normalize_file_path(item) and not is_unknown_resource(normalize_file_path(item))
            ]
            mapping_links = []
            for file_path in current_files:
                mapping_chain = lineage_state.build_full_chain(file_path, self.config.max_lineage_depth)
                if mapping_chain and mapping_chain not in mapping_links:
                    mapping_links.append(mapping_chain)

            candidates.append(
                UploadCandidate(
                    candidate_id=f"upload_{index}",
                    session_id=context.session_id,
                    timestamp=event["timestamp"],
                    original_file=event["original_file"],
                    current_files=current_files,
                    app_name=event["app_name"],
                    operation_type=event["operation_type"],
                    sink_type=self._infer_sink_type(event["operation_type"], event["app_name"]),
                    evidence_refs=list(event["evidence_refs"]),
                    mapping_links=mapping_links,
                    confidence=event["confidence"],
                    status=event["status"],
                    object_binding=dict(event.get("object_binding", {}) or {}),
                )
            )

        if not candidates:
            for index, segment in enumerate(context.normalized_segments):
                combined = f"{segment.app_name} {segment.operation_type} {segment.action_description}".lower()
                segment_signal = any(keyword in combined for keyword in upload_keywords) or any(
                    marker in combined
                    for marker in (
                        "share",
                        "screen",
                        "meeting",
                        "tencent meeting",
                        "共享",
                        "屏幕",
                        "会议",
                        "腾讯会议",
                        "共享屏幕",
                        "屏幕共享",
                        "上传",
                        "发送",
                        "外发",
                    )
                )
                if not segment_signal:
                    continue

                original_file = self._resolve_segment_original_file(segment, lineage_state, context.sensitive_files)
                if not original_file:
                    continue
                if self._segment_is_covered_by_existing_candidate(segment, original_file, candidates):
                    continue

                current_files = self._collect_segment_files(segment, lineage_state)
                current_files = [
                    item
                    for item in current_files
                    if normalize_file_path(item) and not is_unknown_resource(normalize_file_path(item))
                ]
                mapping_links = []
                for file_path in current_files:
                    mapping_chain = lineage_state.build_full_chain(file_path, self.config.max_lineage_depth)
                    if mapping_chain and mapping_chain not in mapping_links:
                        mapping_links.append(mapping_chain)

                if not mapping_links and original_file:
                    mapping_links = [original_file]

                candidates.append(
                    UploadCandidate(
                        candidate_id=f"upload_segment_{index}",
                        session_id=context.session_id,
                        timestamp=segment.supporting_timestamps[0] if segment.supporting_timestamps else segment.time_range,
                        original_file=original_file,
                        current_files=current_files or [original_file],
                        app_name=segment.app_name,
                        operation_type=segment.operation_type,
                        sink_type=self._infer_sink_type(segment.operation_type, segment.app_name),
                        evidence_refs=[f"segment:{segment.segment_id}"],
                        mapping_links=mapping_links,
                        confidence=max(0.0, min(segment.confidence, 1.0)),
                        status="segment_only",
                        object_binding=self._build_segment_object_binding(
                            context=context,
                            segment=segment,
                            original_file=original_file,
                            lineage_state=lineage_state,
                        ),
                    )
                )

        return candidates

    def _build_event_object_binding(self, context, log_event, segment, original_file: str, lineage_state: LineageState) -> dict:
        if segment is None:
            return {
                "binding_type": "log_only",
                "binding_confidence": 1.0 if original_file else 0.0,
                "evidence": ["sensitive_log_path"] if original_file else [],
            }

        return self._build_segment_object_binding(
            context=context,
            segment=segment,
            original_file=original_file,
            lineage_state=lineage_state,
        )

    def _build_segment_object_binding(self, context, segment, original_file: str, lineage_state: LineageState) -> dict:
        evidence: list[str] = []
        confidence = max(0.0, min(segment.confidence, 1.0))
        binding_type = "heuristic"

        resource_candidates = [segment.primary_resource, *segment.related_resources]
        normalized_candidates = [normalize_file_path(item) for item in resource_candidates if normalize_file_path(item)]
        candidate_basenames = {get_path_basename(item).lower() for item in normalized_candidates if item}

        if normalized_candidates:
            for candidate in normalized_candidates:
                candidate_for_resolution = candidate
                if ":" not in candidate_for_resolution:
                    candidate_for_resolution = lineage_state.resolve_by_basename(candidate_for_resolution) or candidate_for_resolution
                resolved = lineage_state.resolve_root(candidate_for_resolution, self.config.max_lineage_depth)
                if resolved and resolved == original_file:
                    evidence.append("lineage_resolved_resource")
                    binding_type = "lineage"
                    confidence = max(confidence, 0.95)
                    break
            else:
                if any(
                    get_path_basename(original_file).lower() == basename
                    for basename in candidate_basenames
                    if basename
                ):
                    evidence.append("basename_matched_sensitive_file")
                    binding_type = "basename_match"
                    confidence = max(confidence, 0.85)

        if not evidence:
            temporal_binding = self._infer_temporal_sensitive_binding(context, segment, original_file)
            if temporal_binding is not None:
                evidence.extend(temporal_binding["evidence"])
                binding_type = temporal_binding["binding_type"]
                confidence = max(confidence, temporal_binding["binding_confidence"])

        if not evidence and original_file:
            original_basename = get_path_basename(original_file).lower()
            evidence_blob = " ".join(
                [
                    segment.app_name,
                    segment.operation_type,
                    segment.action_description,
                    *segment.visible_evidence,
                ]
            ).lower()
            if original_basename and original_basename in evidence_blob:
                evidence.append("basename_visible_in_segment")
                binding_type = "basename_match"
                confidence = max(confidence, 0.88)

        if not evidence and len(context.sensitive_files) == 1:
            evidence.append("single_sensitive_file_fallback")
            binding_type = "single_sensitive_file_fallback"
            confidence = max(confidence, 0.6)

        return {
            "binding_type": binding_type,
            "binding_confidence": round(confidence, 3),
            "bound_asset": original_file,
            "evidence": evidence,
        }

    def _infer_temporal_sensitive_binding(self, context, segment, original_file: str) -> dict | None:
        if not original_file:
            return None

        segment_time = None
        if segment.supporting_timestamps:
            segment_time = segment.supporting_timestamps[0]
        elif segment.start_time:
            segment_time = segment.start_time.strftime("%Y-%m-%d %H:%M:%S")
        if not segment_time:
            return None

        segment_dt = parse_timestamp(segment_time)
        if segment_dt is None:
            return None

        time_delta_seconds = None
        nearest_sensitive_open = None
        for log_event in context.normalized_logs:
            if log_event.file_path != original_file:
                continue
            if log_event.event_type.lower() != "file_open":
                continue
            delta = abs((segment_dt - log_event.timestamp).total_seconds())
            if time_delta_seconds is None or delta < time_delta_seconds:
                time_delta_seconds = delta
                nearest_sensitive_open = log_event

        if nearest_sensitive_open is None or time_delta_seconds is None:
            return None

        sink_type = self._infer_sink_type(segment.operation_type, segment.app_name)
        if sink_type == "screen_share" and time_delta_seconds <= 60:
            return {
                "binding_type": "temporal_screen_share_binding",
                "binding_confidence": 0.82,
                "evidence": [
                    "recent_sensitive_file_open",
                    f"open_to_segment_delta_seconds:{int(time_delta_seconds)}",
                    "screen_share_context",
                ],
            }

        return None

    def _segment_is_covered_by_existing_candidate(
        self,
        segment,
        original_file: str,
        candidates: list[UploadCandidate],
    ) -> bool:
        segment_sink = self._infer_sink_type(segment.operation_type, segment.app_name)
        segment_files = {
            normalize_file_path(item)
            for item in self._collect_segment_files(segment, LineageState(set()))
            if normalize_file_path(item) and not is_unknown_resource(normalize_file_path(item))
        }
        segment_time = segment.supporting_timestamps[0] if segment.supporting_timestamps else segment.time_range
        segment_bucket = segment_time[:16]

        for candidate in candidates:
            if normalize_file_path(candidate["original_file"]) != normalize_file_path(original_file):
                continue
            if str(candidate["sink_type"] or "").strip().lower() != str(segment_sink or "").strip().lower():
                continue
            candidate_bucket = str(candidate["timestamp"] or "")[:16]
            if candidate_bucket and segment_bucket and candidate_bucket != segment_bucket:
                continue

            candidate_files = {
                normalize_file_path(item)
                for item in candidate.get("current_files", []) or []
                if normalize_file_path(item)
            }
            candidate_file_basenames = {get_path_basename(item).lower() for item in candidate_files if item}
            segment_file_basenames = {get_path_basename(item).lower() for item in segment_files if item}
            if (
                segment_files
                and candidate_files
                and not segment_files.issubset(candidate_files)
                and not segment_file_basenames.issubset(candidate_file_basenames)
            ):
                continue
            return True

        return False

    def _event_represents_actual_sink_interaction(self, event: CorrelatedEvent) -> bool:
        event_type = str(event["event_type"] or "").strip().lower()
        operation_type = str(event["operation_type"] or "").strip().lower()
        app_name = str(event["app_name"] or "").strip().lower()

        if event_type in {"file_upload", "file_selected", "upload_detected"}:
            return True

        if any(term in operation_type for term in EXPLICIT_SINK_TERMS):
            return True

        if event["source_type"] == "segment" and self._infer_sink_type(event["operation_type"], event["app_name"]) in {
            "screen_share",
            "mail_attachment",
            "chat_upload",
            "cloud_sync",
        }:
            return True

        if event_type in {"created", "modified"} and any(term in app_name for term in PRECURSOR_BROWSER_APPS):
            return False

        return False

    def _infer_sink_type(self, operation_type: str, app_name: str) -> str:
        combined = f"{operation_type} {app_name}".lower()
        if any(marker in combined for marker in SCREEN_SHARE_MARKERS):
            return "screen_share"
        if any(marker in combined for marker in MAIL_MARKERS):
            return "mail_attachment"
        if any(marker in combined for marker in CHAT_MARKERS):
            return "chat_upload"
        if any(marker in combined for marker in CLOUD_MARKERS):
            return "cloud_sync"
        return "web_post"

    def _collect_candidate_files(self, event: CorrelatedEvent, segment_lookup: dict, lineage_state: LineageState) -> list[str]:
        current_files: list[str] = []

        direct_file = normalize_file_path(event["current_file"])
        if direct_file:
            current_files.append(direct_file)

        for evidence_ref in event["evidence_refs"]:
            if not evidence_ref.startswith("segment:"):
                continue
            segment_id = evidence_ref.split(":", 1)[1]
            segment = segment_lookup.get(segment_id)
            if not segment:
                continue

            for candidate in [segment.primary_resource, *segment.related_resources]:
                normalized = normalize_file_path(candidate)
                if normalized and ":" not in normalized:
                    resolved = lineage_state.resolve_by_basename(normalized)
                    normalized = resolved or normalized
                if normalized and normalized not in current_files:
                    current_files.append(normalized)

        deduped: list[str] = []
        seen = set()
        for item in current_files:
            normalized = normalize_file_path(item)
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            deduped.append(normalized)
        return deduped

    def _collect_segment_files(self, segment, lineage_state: LineageState) -> list[str]:
        current_files: list[str] = []
        for candidate in [segment.primary_resource, *segment.related_resources]:
            normalized = normalize_file_path(candidate)
            if normalized and ":" not in normalized:
                resolved = lineage_state.resolve_by_basename(normalized)
                normalized = resolved or normalized
            if normalized and normalized not in current_files and not is_unknown_resource(normalized):
                current_files.append(normalized)
        return current_files

    def _build_operation_records(self, correlated_events: list[CorrelatedEvent]) -> list[dict]:
        operation_records: list[dict] = []
        for event in sorted(correlated_events, key=lambda item: item["timestamp"]):
            operation_records.append(
                {
                    "operation_time": event["timestamp"],
                    "sensitive_file_path": event["original_file"],
                    "current_file": event["current_file"],
                    "app_name": event["app_name"],
                    "operation": event["operation_type"],
                    "behavior_category": event["behavior_category"],
                    "evidence_refs": list(event["evidence_refs"]),
                    "status": event["status"],
                }
            )
        return operation_records
