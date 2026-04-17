from __future__ import annotations

from .config import LeakReasonerConfig


class LeakRiskScorer:
    def __init__(self, config: LeakReasonerConfig):
        self.config = config

    def score(self, fact: dict) -> tuple[int, str, str, list[str]]:
        score = 0
        reasons: list[str] = []

        sink_type = str(fact.get("sink_type", "") or "").strip()
        mapping_links = fact.get("mapping_links", []) or []
        current_files = fact.get("current_files", []) or []
        confidence = float(fact.get("confidence", 0.0) or 0.0)

        if sink_type in self.config.high_risk_sink_types:
            score += 45
            reasons.append(f"high_risk_sink:{sink_type}")

        if len(current_files) >= 2:
            score += 15
            reasons.append("multiple_files_uploaded")
        elif len(current_files) == 1:
            score += 8
            reasons.append("single_file_uploaded")

        if mapping_links:
            score += 20
            reasons.append("lineage_available")

        score += int(min(confidence, 1.0) * 20)
        if confidence >= 0.8:
            reasons.append("high_confidence")

        if score >= self.config.high_risk_score:
            return score, "high", "alert", reasons
        if score >= self.config.medium_risk_score:
            return score, "medium", "review", reasons
        return score, "low", "inform", reasons
