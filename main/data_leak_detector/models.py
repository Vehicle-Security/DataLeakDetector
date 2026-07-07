"""Shared dataclasses that define the pipeline's internal data contract.

Every stage imports these models instead of passing loosely shaped dictionaries
around. This keeps the single-package rewrite modular without reintroducing the
old duplicated stage directories.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class LogEvent:
    """A normalized event from a desktop monitor or imported audit log."""

    event_id: str
    timestamp: str
    timestamp_ms: int
    event_type: str
    file_path: str = ""
    process_name: str = ""
    app_name: str = ""
    window_title: str = ""
    description: str = ""
    raw: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class FrameObservation:
    """Structured visual or log-anchored behavior evidence."""

    observation_id: str
    start_ms: int
    end_ms: int
    app_name: str
    operation_type: str
    resource: str = ""
    related_resources: tuple[str, ...] = ()
    description: str = ""
    confidence: float = 0.0
    source: str = "frame_analyzer"

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["related_resources"] = list(self.related_resources)
        return payload


@dataclass(frozen=True)
class CorrelatedEvent:
    """A sensitive object bound to log and optional frame evidence."""

    event_id: str
    timestamp: str
    event_type: str
    app_name: str
    original_file: str
    current_file: str
    operation_type: str
    behavior_category: str
    confidence: float
    evidence_refs: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["evidence_refs"] = list(self.evidence_refs)
        return payload


@dataclass(frozen=True)
class UploadCandidate:
    """Potential external sink interaction involving sensitive data."""

    candidate_id: str
    timestamp: str
    app_name: str
    original_file: str
    current_file: str
    sink_type: str
    risk_level: str
    confidence: float
    evidence_refs: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["evidence_refs"] = list(self.evidence_refs)
        return payload


@dataclass(frozen=True)
class DatalogFact:
    """Symbolic fact consumed by the leak reasoner."""

    relation: str
    args: tuple[Any, ...]

    def to_dict(self) -> dict[str, Any]:
        return {"relation": self.relation, "args": list(self.args)}


@dataclass(frozen=True)
class LeakPath:
    """Reasoned taint path from sensitive source to external sink."""

    start_op: str
    end_op: str
    leaking_proc: str
    leaked_file: str
    leak_channel: str
    leak_timestamp: int
    full_path: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "start_op": self.start_op,
            "end_op": self.end_op,
            "leaking_proc": self.leaking_proc,
            "leaked_file": self.leaked_file,
            "leak_channel": self.leak_channel,
            "leak_timestamp": self.leak_timestamp,
            "full_path": self.full_path,
            "path_steps": self.full_path.split(" -> ") if self.full_path else [],
        }


@dataclass(frozen=True)
class DetectionReport:
    """Final report returned by the end-to-end pipeline."""

    report_id: str
    generated_at: str
    input: dict[str, str]
    summary: dict[str, Any]
    frame_analyzer: dict[str, Any]
    event_correlator: dict[str, Any]
    leak_reasoner: dict[str, Any]
    conclusion: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
