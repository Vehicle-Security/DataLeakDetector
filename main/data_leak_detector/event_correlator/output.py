"""Report-shaping helpers for EventCorrelator output.

The correlator works with typed objects internally. These helpers convert those
objects into stable JSON sections used by CLI output, tests, and Neo4j writes.
"""

from __future__ import annotations

from pathlib import Path
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
