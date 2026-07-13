"""Configuration for deterministic event correlation."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EventCorrelatorConfig:
    """Small knobs for evidence binding.

    Initial sensitive sources come from the sensitive-files configuration,
    which may be maintained from verified log evidence. Groundtruth never
    supplies sources. Derived files are inferred through lineage and reasoning,
    not inserted into the initial source set.
    """

    upload_confidence: float = 0.86
    transfer_confidence: float = 0.72
    non_vlm_enabled: bool = True
