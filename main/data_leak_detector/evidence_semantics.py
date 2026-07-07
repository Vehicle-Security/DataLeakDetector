"""Outcome semantics for risk signals versus confirmed leak conclusions.

The project distinguishes suspicious behavior from confirmed taint-to-sink
paths. This file keeps that decision policy explicit so tests, reports, and
future UI code can agree on what "positive" means.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .policy import CONFIRMED_RISK_LEVELS, RISK_LEVELS


@dataclass(frozen=True)
class EvidenceDecision:
    risk_positive: bool
    confirmed_leak: bool
    final_positive: bool
    risk_reasoning_source: str
    reasoning_source: str
    risk_semantics: str = "attempt_or_exposure_or_completed"
    final_semantics: str = "confirmed_leak"


def normalize_risk_level(value: object) -> str:
    text = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    return text if text in RISK_LEVELS or text in {"none", "preparation"} else ""


def is_confirmed_risk_level(value: object) -> bool:
    return normalize_risk_level(value) in CONFIRMED_RISK_LEVELS


def decide_evidence_outcome(
    *,
    datalog_risk_positive: bool,
    datalog_confirmed: bool,
    log_rule_positive: bool,
    log_rule_rules: Iterable[object],
) -> EvidenceDecision:
    # Log rules are risk signals. Confirmation should come from symbolic leak
    # reasoning or explicit future confirmation rules.
    _ = list(log_rule_rules)
    risk_positive = bool(datalog_risk_positive or log_rule_positive)
    confirmed = bool(datalog_confirmed)
    return EvidenceDecision(
        risk_positive=risk_positive,
        confirmed_leak=confirmed,
        final_positive=confirmed,
        risk_reasoning_source="datalog" if datalog_risk_positive else ("log_rule" if log_rule_positive else "none"),
        reasoning_source="datalog" if confirmed else "none",
    )
