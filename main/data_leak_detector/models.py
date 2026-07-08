"""定义流水线内部数据契约的共享数据类。

每个阶段都导入这些模型，而不是到处传递形状松散的字典。这样可以让单包重写保持模块化，
同时避免重新引入旧的重复阶段目录。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class LogEvent:
    """来自桌面监控或导入审计日志的规范化事件。"""

    event_id: str
    timestamp: str
    timestamp_ms: int
    video_time_ms: int
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
    """结构化的视觉证据或日志锚定行为证据。"""

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
    """与日志以及可选帧证据绑定的敏感对象。"""

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
    """涉及敏感数据的潜在外部汇聚点交互。"""

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
    """泄露推理器消费的符号事实。"""

    relation: str
    args: tuple[Any, ...]

    def to_dict(self) -> dict[str, Any]:
        return {"relation": self.relation, "args": list(self.args)}


@dataclass(frozen=True)
class LeakPath:
    """从敏感源到外部汇聚点的推理污点路径。"""

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
    """端到端流水线返回的最终报告。"""

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
