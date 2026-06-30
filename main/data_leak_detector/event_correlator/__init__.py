from __future__ import annotations

from data_leak_detector.legacy_paths import EVENT_CORRELATOR_IMPL, add_legacy_import_paths


add_legacy_import_paths(EVENT_CORRELATOR_IMPL)

from event_correlator import EventCorrelator, EventCorrelatorConfig, classify_frontend_app  # noqa: E402


__all__ = ["EventCorrelator", "EventCorrelatorConfig", "classify_frontend_app"]
