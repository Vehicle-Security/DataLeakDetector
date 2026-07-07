from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class LogEvent:
    """Normalized event emitted by a desktop monitor or imported log file."""

    event_id: str
    timestamp: str
    timestamp_ms: int
    event_type: str
    file_path: str = ""
    file_name: str = ""
    process_name: str = ""
    app_name: str = ""
    window_title: str = ""
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class FrameObservation:
    """Human-readable behavior observation from OCR/VLM or a deterministic fallback."""

    observation_id: str
    start_ms: int
    end_ms: int
    app_name: str
    operation_type: str
    resource: str = ""
    related_resources: list[str] = field(default_factory=list)
    description: str = ""
    confidence: float = 0.0
    source: str = "frame_analyzer"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CorrelatedEvent:
    """A log event bound to a sensitive object, a visual observation, or both."""

    event_id: str
    timestamp: str
    event_type: str
    original_file: str
    current_file: str
    app_name: str
    operation_type: str
    behavior_category: str
    confidence: float
    evidence_refs: list[str] = field(default_factory=list)
    status: str = "linked"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class UploadCandidate:
    """Potential external sink interaction involving sensitive data."""

    candidate_id: str
    timestamp: str
    original_file: str
    current_files: list[str]
    app_name: str
    operation_type: str
    sink_type: str
    confidence: float
    evidence_refs: list[str] = field(default_factory=list)
    status: str = "candidate"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DatalogFact:
    """Symbolic fact used by the leak reasoner."""

    relation: str
    args: tuple[Any, ...]

    def to_dict(self) -> dict[str, Any]:
        return {"relation": self.relation, "args": list(self.args)}


@dataclass(frozen=True)
class LeakPath:
    """Reasoned path from sensitive source to external sink."""

    start_op: str
    end_op: str
    leaking_proc: str
    leaked_file: str
    full_path: str
    leak_channel: str
    leak_timestamp: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "start_op": self.start_op,
            "end_op": self.end_op,
            "leaking_proc": self.leaking_proc,
            "leaked_file": self.leaked_file,
            "full_path": self.full_path,
            "leak_channel": self.leak_channel,
            "leak_timestamp": self.leak_timestamp,
            "path_steps": self.full_path.split(" -> ") if self.full_path else [],
        }


@dataclass
class DetectionReport:
    """Final report object returned by the E2E pipeline."""

    report_id: str
    generated_at: str
    input: dict[str, str]
    summary: dict[str, Any]
    event_correlator: dict[str, Any]
    frame_analyzer: dict[str, Any]
    leak_reasoner: dict[str, Any]
    conclusion: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
