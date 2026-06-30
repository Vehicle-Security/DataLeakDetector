from __future__ import annotations

from data_leak_detector.legacy_paths import LEAK_REASONER_IMPL, add_legacy_import_paths


add_legacy_import_paths(LEAK_REASONER_IMPL)

from datalog.datalog_engine import DatalogEngine  # noqa: E402
from threat_prompts import PromptTemplates  # noqa: E402


__all__ = ["DatalogEngine", "PromptTemplates"]

