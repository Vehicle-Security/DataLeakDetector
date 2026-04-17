from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional, TypedDict


class FrameSegment(TypedDict, total=False):
    segment_id: str
    time_range: str
    app_name: str
    operation_type: str
    primary_resource: str
    related_resources: list[str]
    action_description: str
    visible_evidence: list[str]
    supporting_timestamps: list[str]
    confidence: float


class EventCorrelatorInput(TypedDict, total=False):
    session_id: str
    record_id: str
    log_events: list[dict[str, Any]]
    frame_segments: list[FrameSegment]
    sensitive_files: list[str]
    recording_start_time: str
    session_metadata: dict[str, Any]
    correlation_config: dict[str, Any]


class CorrelatedEvent(TypedDict):
    event_id: str
    session_id: str
    timestamp: str
    event_type: str
    source_type: str
    original_file: str
    current_file: str
    app_name: str
    operation_type: str
    behavior_category: str
    evidence_refs: list[str]
    confidence: float
    correlation_score: float
    status: str
    object_binding: dict[str, Any]


class UploadCandidate(TypedDict):
    candidate_id: str
    session_id: str
    timestamp: str
    original_file: str
    current_files: list[str]
    app_name: str
    operation_type: str
    sink_type: str
    evidence_refs: list[str]
    mapping_links: list[str]
    confidence: float
    status: str
    object_binding: dict[str, Any]


class FileLineage(TypedDict):
    direct_file_mappings: dict[str, str]
    full_file_mapping_chains: dict[str, str]


class CorrelationBundle(TypedDict):
    session_id: str
    analysis_status: str
    correlated_events: list[CorrelatedEvent]
    operation_records: list[dict[str, Any]]
    upload_candidates: list[UploadCandidate]
    file_lineage: FileLineage
    statistics: dict[str, Any]
    errors: list[dict[str, Any]]


@dataclass(frozen=True)
class NormalizedLogEvent:
    event_id: str
    timestamp: datetime
    timestamp_text: str
    event_type: str
    file_path: str
    file_name: str
    process_name: str
    app_name: str
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class NormalizedFrameSegment:
    segment_id: str
    start_time: Optional[datetime]
    end_time: Optional[datetime]
    time_range: str
    app_name: str
    operation_type: str
    primary_resource: str
    related_resources: list[str]
    action_description: str
    visible_evidence: list[str]
    supporting_timestamps: list[str]
    confidence: float
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass
class CorrelatorContext:
    session_id: str
    record_id: str
    sensitive_files: list[str]
    recording_start_time: str
    session_metadata: dict[str, Any]
    correlation_config: dict[str, Any]
    normalized_logs: list[NormalizedLogEvent]
    normalized_segments: list[NormalizedFrameSegment]
    errors: list[dict[str, Any]] = field(default_factory=list)
