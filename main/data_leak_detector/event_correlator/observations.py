"""帧观察的规范化与时间窗口匹配。

本文件是 FrameAnalyzer 输出与 EventCorrelator 输入之间的适配器。它接受字典或模型对象，
并筛选邻近证据，这样关联器就不需要了解每一种观察形状。
"""

from __future__ import annotations

from bisect import bisect_left
from dataclasses import dataclass
from typing import Any

from ..io import normalize_path
from ..models import FrameObservation


def normalize_observations(items: list[Any]) -> list[FrameObservation]:
    """将帧分段字典强制转换为 FrameObservation 对象。"""

    observations: list[FrameObservation] = []
    for index, item in enumerate(items):
        if isinstance(item, FrameObservation):
            observations.append(item)
            continue
        if not isinstance(item, dict):
            continue
        observations.append(
            FrameObservation(
                observation_id=str(item.get("observation_id") or item.get("segment_id") or f"obs_{index}"),
                start_ms=int(item.get("start_ms") or 0),
                end_ms=int(item.get("end_ms") or item.get("start_ms") or 0),
                app_name=str(item.get("app_name") or ""),
                operation_type=str(item.get("operation_type") or item.get("operation") or ""),
                resource=normalize_path(item.get("resource") or item.get("file_path") or ""),
                related_resources=tuple(normalize_path(value) for value in item.get("related_resources") or ()),
                description=str(item.get("description") or ""),
                confidence=float(item.get("confidence") or 0.0),
                source=str(item.get("source") or "frame_analyzer"),
            )
        )
    return observations


def nearest_observation(
    timestamp_ms: int,
    observations: list[FrameObservation],
    tolerance_ms: int,
) -> FrameObservation | None:
    """返回配置窗口内最近的视觉观察。"""

    if not timestamp_ms:
        return None
    best: tuple[int, FrameObservation] | None = None
    for observation in observations:
        center = observation.start_ms if not observation.end_ms else (observation.start_ms + observation.end_ms) // 2
        distance = abs(timestamp_ms - center)
        if distance <= tolerance_ms and (best is None or distance < best[0]):
            best = (distance, observation)
    return best[1] if best else None


@dataclass(frozen=True)
class ObservationIndex:
    centers: tuple[int, ...]
    observations: tuple[FrameObservation, ...]

    @classmethod
    def from_observations(cls, observations: list[FrameObservation]) -> "ObservationIndex":
        indexed = sorted(
            (
                observation.start_ms if not observation.end_ms else (observation.start_ms + observation.end_ms) // 2,
                index,
                observation,
            )
            for index, observation in enumerate(observations)
        )
        return cls(
            centers=tuple(item[0] for item in indexed),
            observations=tuple(item[2] for item in indexed),
        )

    def nearest(self, timestamp_ms: int, tolerance_ms: int) -> FrameObservation | None:
        if not timestamp_ms:
            return None
        position = bisect_left(self.centers, timestamp_ms)
        best: tuple[int, FrameObservation] | None = None
        for index in (position - 1, position):
            if index < 0 or index >= len(self.centers):
                continue
            distance = abs(timestamp_ms - self.centers[index])
            if distance <= tolerance_ms and (best is None or distance < best[0]):
                best = (distance, self.observations[index])
        return best[1] if best else None
