from __future__ import annotations

from datetime import datetime
from typing import Any

from .config import LeakReasonerConfig
from .datalog_bridge import LeakDatalogBridge
from .fact_builder import CorrelationFactBuilder
from .schema import LeakReasonerInput, LeakReasonerOutput, RiskCase
from .scorer import LeakRiskScorer


class LeakReasoner:
    def __init__(self, config: LeakReasonerConfig | None = None):
        self.config = config or LeakReasonerConfig()
        self.fact_builder = CorrelationFactBuilder()
        self.scorer = LeakRiskScorer(self.config)
        self.datalog_bridge = LeakDatalogBridge()

    def run(self, payload: LeakReasonerInput) -> LeakReasonerOutput:
        session_id = str(payload.get("session_id", "") or "").strip()
        if not session_id:
            raise ValueError("session_id is required")

        correlation_bundle = dict(payload.get("correlation_bundle", {}) or {})
        facts = self.fact_builder.build(correlation_bundle)
        leak_paths = self.datalog_bridge.run(facts)
        upload_facts = [fact for fact in facts if fact.get("fact_type") == "upload_candidate"]

        risk_cases: list[RiskCase] = []
        evidence_bundles: list[dict[str, Any]] = []
        errors: list[dict[str, Any]] = []

        for index, candidate_fact in enumerate(upload_facts):
            confidence = float(candidate_fact.get("confidence", 0.0) or 0.0)
            if confidence < self.config.min_confidence_for_case:
                continue

            matched_paths = self._match_leak_paths(candidate_fact, leak_paths)
            if not matched_paths:
                continue

            score, severity, disposition, reasons = self.scorer.score(candidate_fact)
            reasons = self._augment_reasons_from_paths(reasons, matched_paths)
            leak_channel = self._resolve_leak_channel(candidate_fact, matched_paths)

            case_id = f"case_{session_id}_{index}"
            evidence_bundle_id = f"evidence_{session_id}_{index}"

            evidence_bundles.append(
                {
                    "evidence_bundle_id": evidence_bundle_id,
                    "session_id": session_id,
                    "fact_type": candidate_fact.get("fact_type", ""),
                    "evidence_refs": list(candidate_fact.get("evidence_refs", []) or []),
                    "mapping_links": list(candidate_fact.get("mapping_links", []) or []),
                    "leak_paths": matched_paths,
                }
            )

            risk_cases.append(
                RiskCase(
                    case_id=case_id,
                    session_id=session_id,
                    severity=severity,
                    score=score,
                    confidence=confidence,
                    disposition=disposition,
                    primary_asset_id=str(candidate_fact.get("original_file", "") or ""),
                    asset_lineage=list(candidate_fact.get("mapping_links", []) or []),
                    sink_type=str(candidate_fact.get("sink_type", "") or ""),
                    leak_channel=leak_channel,
                    sink_target=str(candidate_fact.get("app_name", "") or ""),
                    actor={
                        "app_name": str(candidate_fact.get("app_name", "") or ""),
                        "leaking_processes": sorted(
                            {
                                str(path.get("leaking_proc", "") or "")
                                for path in matched_paths
                                if str(path.get("leaking_proc", "") or "").strip()
                            }
                        ),
                    },
                    reasons=reasons,
                    evidence_bundle_id=evidence_bundle_id,
                    recommended_actions=self._build_recommended_actions(severity),
                    created_at=datetime.now().isoformat(),
                )
            )

        analysis_status = "success"
        if not risk_cases:
            analysis_status = "no_case"
        if errors:
            analysis_status = "partial_success"

        return LeakReasonerOutput(
            session_id=session_id,
            analysis_status=analysis_status,
            risk_cases=risk_cases,
            evidence_bundles=evidence_bundles,
            metrics={
                "facts_input": len(facts),
                "upload_candidates_input": len(upload_facts),
                "risk_cases_output": len(risk_cases),
                "leak_paths_output": len(leak_paths),
            },
            errors=errors,
        )

    def _match_leak_paths(
        self,
        candidate_fact: dict[str, Any],
        leak_paths: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        original_file = str(candidate_fact.get("original_file", "") or "").strip()
        current_files = {
            str(item or "").strip()
            for item in list(candidate_fact.get("current_files", []) or [])
            if str(item or "").strip()
        }
        sink_type = str(candidate_fact.get("sink_type", "") or "").strip().lower()

        matched_paths: list[dict[str, Any]] = []
        for path in leak_paths:
            leaked_file = str(path.get("leaked_file", "") or "").strip()
            leak_channel = str(path.get("leak_channel", "") or "").strip().lower()
            if sink_type and leak_channel and leak_channel != sink_type:
                continue
            if leaked_file and leaked_file in current_files:
                matched_paths.append(path)
                continue
            if leaked_file and original_file and leaked_file == original_file and not current_files:
                matched_paths.append(path)

        return matched_paths

    def _resolve_leak_channel(
        self,
        candidate_fact: dict[str, Any],
        matched_paths: list[dict[str, Any]],
    ) -> str:
        for path in matched_paths:
            channel = str(path.get("leak_channel", "") or "").strip()
            if channel:
                return channel
        return str(candidate_fact.get("sink_type", "") or "").strip()

    def _augment_reasons_from_paths(
        self,
        reasons: list[str],
        matched_paths: list[dict[str, Any]],
    ) -> list[str]:
        merged = list(reasons)
        if matched_paths:
            merged.append("datalog_leak_path_confirmed")

        unique_channels = sorted(
            {
                str(path.get("leak_channel", "") or "").strip()
                for path in matched_paths
                if str(path.get("leak_channel", "") or "").strip()
            }
        )
        for channel in unique_channels:
            merged.append(f"leak_channel:{channel}")

        unique_processes = sorted(
            {
                str(path.get("leaking_proc", "") or "").strip()
                for path in matched_paths
                if str(path.get("leaking_proc", "") or "").strip()
            }
        )
        for process in unique_processes:
            merged.append(f"leaking_process:{process}")

        deduped: list[str] = []
        seen = set()
        for reason in merged:
            if reason in seen:
                continue
            seen.add(reason)
            deduped.append(reason)
        return deduped

    def _build_recommended_actions(self, severity: str) -> list[str]:
        if severity == "high":
            return ["raise_alert", "manual_review", "preserve_evidence"]
        if severity == "medium":
            return ["manual_review", "preserve_evidence"]
        return ["record_only"]
