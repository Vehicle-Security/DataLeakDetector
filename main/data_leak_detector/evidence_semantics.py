from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


RISK_LEVELS = {
    "attempted",
    "selected_or_attached",
    "in_progress",
    "content_exposed",
    "completed",
}

CONFIRMED_RISK_LEVELS = {"content_exposed", "completed"}

CONFIRMED_LOG_RULES = {"upload_event", "screen_share"}


@dataclass(frozen=True)
class EvidenceDecision:
    risk_positive: bool
    confirmed_leak: bool
    final_positive: bool
    risk_reasoning_source: str
    reasoning_source: str
    final_semantics: str = "confirmed_leak"
    risk_semantics: str = "staging_or_attempted_or_confirmed"


def normalize_risk_level(value: object) -> str:
    text = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    return text if text in RISK_LEVELS or text in {"none", "preparation"} else ""


def is_confirmed_risk_level(value: object) -> bool:
    return normalize_risk_level(value) in CONFIRMED_RISK_LEVELS


def log_rules_confirm_leak(rules: Iterable[object]) -> bool:
    return any(str(rule or "") in CONFIRMED_LOG_RULES for rule in rules)


def decide_evidence_outcome(
    *,
    datalog_risk_positive: bool,
    datalog_confirmed: bool,
    log_rule_positive: bool,
    log_rule_rules: Iterable[object],
) -> EvidenceDecision:
    log_rule_confirmed = log_rules_confirm_leak(log_rule_rules)
    risk_positive = bool(datalog_risk_positive) or bool(log_rule_positive)
    confirmed_leak = bool(datalog_confirmed) or log_rule_confirmed
    return EvidenceDecision(
        risk_positive=risk_positive,
        confirmed_leak=confirmed_leak,
        final_positive=confirmed_leak,
        risk_reasoning_source=(
            "log_rule"
            if log_rule_positive and not datalog_risk_positive
            else ("datalog" if risk_positive else "none")
        ),
        reasoning_source=(
            "log_rule"
            if log_rule_confirmed and not datalog_confirmed
            else ("datalog" if datalog_confirmed else "none")
        ),
    )
