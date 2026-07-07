"""风险信号与已确认泄露结论的结果语义。

项目需要区分可疑行为和已确认的污点到泄露路径。本文件将该决策策略显式化，
以便测试、报告和未来的 UI 代码对“阳性”的含义保持一致。
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
    # 日志规则只代表风险信号。确认应来自符号泄露推理或未来显式的确认规则。
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
