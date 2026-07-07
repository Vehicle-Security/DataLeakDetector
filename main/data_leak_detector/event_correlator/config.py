"""Configuration object for deterministic event correlation.

These knobs are separated from the correlator so tests and future deployments
can tune time windows and confidence defaults without editing workflow code.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EventCorrelatorConfig:
    """Small set of knobs for deterministic evidence binding."""

    nearby_window_ms: int = 5 * 60 * 1000
    upload_confidence: float = 0.86
    transfer_confidence: float = 0.72
