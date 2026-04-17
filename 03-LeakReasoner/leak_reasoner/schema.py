from __future__ import annotations

from typing import Any, TypedDict


class LeakReasonerInput(TypedDict, total=False):
    session_id: str
    correlation_bundle: dict[str, Any]
    policy_snapshot: dict[str, Any]
    session_metadata: dict[str, Any]


class RiskCase(TypedDict):
    case_id: str
    session_id: str
    severity: str
    score: int
    confidence: float
    disposition: str
    primary_asset_id: str
    asset_lineage: list[str]
    sink_type: str
    leak_channel: str
    sink_target: str
    actor: dict[str, Any]
    reasons: list[str]
    evidence_bundle_id: str
    recommended_actions: list[str]
    created_at: str


class LeakReasonerOutput(TypedDict):
    session_id: str
    analysis_status: str
    risk_cases: list[RiskCase]
    evidence_bundles: list[dict[str, Any]]
    metrics: dict[str, Any]
    errors: list[dict[str, Any]]
