"""EventCorrelator 输出的报告整形辅助函数。

关联器内部使用的是强类型对象。这些辅助函数会把这些对象转换成稳定的 JSON 分区，供 CLI 输出、
测试和 Neo4j 写入使用。
"""

from __future__ import annotations

from pathlib import Path
import re
from typing import Any

from ..models import CorrelatedEvent
from .lineage import Lineage


def operation_record(event: CorrelatedEvent) -> dict[str, Any]:
    return {
        "operation_time": event.timestamp,
        "sensitive_file_path": event.original_file,
        "current_file": event.current_file,
        "app_name": event.app_name,
        "operation": event.operation_type,
        "behavior_category": event.behavior_category,
        "evidence_refs": list(event.evidence_refs),
        "join_reasons": list(event.join_reasons),
    }


def lineage_payload(lineage: Lineage) -> dict[str, Any]:
    return {
        "direct_file_mappings": dict(lineage.direct),
        "full_file_mapping_chains": {path: lineage.chain(path) for path in lineage.direct},
        "artifact_instances": [
            {"artifact_id": Path(dst).name, "current_file": dst, "source_file": src}
            for dst, src in lineage.direct.items()
        ],
    }


def landing_locations(lineage: Lineage, events: list[CorrelatedEvent]) -> list[dict[str, Any]]:
    """Return user-visible derived artifacts that can serve as carrier locations."""

    locations: dict[str, dict[str, Any]] = {}
    for derived, source in lineage.direct.items():
        _add_landing_location(
            locations,
            path=derived,
            source_file=source,
            location_type="local_file",
            confidence=0.95,
            provenance="file_lineage",
            evidence_refs=(),
        )
    for event in events:
        if not _is_event_landing_location(event):
            continue
        _add_landing_location(
            locations,
            path=event.current_file,
            source_file=event.original_file,
            location_type=_location_type(event),
            confidence=event.confidence,
            provenance="correlated_event",
            evidence_refs=event.evidence_refs,
        )
    return sorted(locations.values(), key=lambda item: item["path"].lower())


def _add_landing_location(
    locations: dict[str, dict[str, Any]],
    *,
    path: str,
    source_file: str,
    location_type: str,
    confidence: float,
    provenance: str,
    evidence_refs: tuple[str, ...],
) -> None:
    normalized = str(path).replace("\\", "/").strip()
    if not _is_user_visible_path(normalized):
        return
    key = normalized.lower()
    candidate = {
        "path": normalized,
        "source_file": str(source_file).replace("\\", "/").strip(),
        "location_type": location_type,
        "confidence": round(float(confidence), 3),
        "provenance": provenance,
        "evidence_refs": list(evidence_refs),
    }
    existing = locations.get(key)
    if existing is None:
        locations[key] = candidate
        return
    if candidate["location_type"] != "local_file" and existing["location_type"] == "local_file":
        existing["location_type"] = candidate["location_type"]
    existing["evidence_refs"] = list(dict.fromkeys([*existing["evidence_refs"], *candidate["evidence_refs"]]))
    if candidate["provenance"] not in existing["provenance"].split("+"):
        existing["provenance"] = f"{existing['provenance']}+{candidate['provenance']}"
    if candidate["confidence"] > existing["confidence"]:
        existing["confidence"] = candidate["confidence"]
        if candidate["source_file"]:
            existing["source_file"] = candidate["source_file"]


def _is_user_visible_path(path: str) -> bool:
    if not re.match(r"^(?:[a-zA-Z]:/|/)", path):
        return False
    lowered = path.lower()
    hidden_markers = (
        "/appdata/",
        "/cache/",
        "/cachedata/",
        "/cacheddata/",
        "/program files/",
        "/programdata/",
        "/temp/",
        "/windows/",
    )
    return not any(marker in lowered for marker in hidden_markers)


def _location_type(event: CorrelatedEvent) -> str:
    reasons = " ".join(event.join_reasons).lower()
    if "sink_type:cloud_sync" in reasons:
        return "cloud_sync_file"
    if "sink_type:removable_media" in reasons or "removable" in reasons:
        return "removable_media_file"
    return "local_file"


def _is_event_landing_location(event: CorrelatedEvent) -> bool:
    if not event.current_file or _same_path(event.current_file, event.original_file):
        return False
    reasons = " ".join(event.join_reasons).lower()
    # Log identity binding may associate an opened sensitive document with many
    # unrelated application files. Only visual transfer evidence or explicit
    # outbound evidence can promote an event path to a carrier location.
    return any(
        marker in reasons
        for marker in (
            "visual_transfer_context",
            "visual_declared_behavior:direct_leak",
            "visual_declared_behavior:hidden_transfer",
            "explicit_sink_log",
            "cloud_sync_directory_transfer",
            "removable_media",
        )
    )


def _same_path(left: str, right: str) -> bool:
    return str(left).replace("\\", "/").strip().lower() == str(right).replace("\\", "/").strip().lower()
