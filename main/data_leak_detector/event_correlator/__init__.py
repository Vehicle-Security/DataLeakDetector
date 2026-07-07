"""EventCorrelator package boundary.

This file exposes only the correlator and its small support contract. The
individual modules below keep classification, lineage, candidate extraction,
and fact generation separate enough to evolve independently.
"""

from __future__ import annotations

from .classification import classify_frontend_app
from .config import EventCorrelatorConfig
from .correlator import EventCorrelator
from .lineage import Lineage

__all__ = ["EventCorrelator", "EventCorrelatorConfig", "Lineage", "classify_frontend_app"]
