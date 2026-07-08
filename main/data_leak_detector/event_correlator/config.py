"""Configuration for deterministic event correlation."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EventCorrelatorConfig:
    """Small knobs for evidence binding.

    Initial sensitive sources should normally come from groundtruth.json or
    explicit CLI arguments. Derived files are inferred through lineage and
    reasoning, not inserted into the initial source set.
    """

    nearby_window_ms: int = 5 * 60 * 1000
    upload_confidence: float = 0.86
    transfer_confidence: float = 0.72
    infer_sensitive_from_logs: bool = False
