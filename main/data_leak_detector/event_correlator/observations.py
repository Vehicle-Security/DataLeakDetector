"""Frame-observation normalization and time-window matching.

This file is the adapter between FrameAnalyzer output and EventCorrelator
input. It accepts dictionaries or model objects and selects nearby evidence so
the correlator does not need to know every observation shape.
"""

from __future__ import annotations

from typing import Any

from ..io import normalize_path
from ..models import FrameObservation


def normalize_observations(items: list[Any]) -> list[FrameObservation]:
    """Coerce frame segment dictionaries into FrameObservation objects."""

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
    """Return the closest visual observation within the configured window."""

    if not timestamp_ms:
        return None
    best: tuple[int, FrameObservation] | None = None
    for observation in observations:
        center = observation.start_ms if not observation.end_ms else (observation.start_ms + observation.end_ms) // 2
        distance = abs(timestamp_ms - center)
        if distance <= tolerance_ms and (best is None or distance < best[0]):
            best = (distance, observation)
    return best[1] if best else None
