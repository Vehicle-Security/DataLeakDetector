"""Neo4j log graph import, reuse fingerprints, and schema setup."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..frame_analyzer.apps import identify_frontend_app
from ..io import flatten_text, normalize_path
from ..models import LogEvent
from ..policy import SENSITIVE_TOKENS, SINK_TOKENS, TRANSFER_TOKENS, contains_any
from .config import Neo4jConfig


@dataclass(frozen=True)
class ImportSummary:
    imported: bool
    reused: bool
    log_hash: str
    imported_events: int = 0
    import_batches: int = 0


class Neo4jLogImporter:
    def __init__(self, config: Neo4jConfig):
        self.config = config

    def ensure_import(
        self,
        session,
        *,
        case_id: str,
        log_file: str | Path,
        records: list[dict[str, Any]],
        logs: list[LogEvent],
        sensitive_files: list[str],
    ) -> ImportSummary:
        log_hash = fingerprint_records(records)
        session.execute_write(self._ensure_schema_tx)
        reusable = bool(
            session.execute_read(
                self._is_reusable_import_tx,
                case_id,
                log_hash,
                len(records),
                self.config.log_miner_schema_version,
            )
        )
        if reusable and self.config.reuse_import:
            return ImportSummary(imported=False, reused=True, log_hash=log_hash)

        graph_events = records_to_graph_events(case_id, logs, sensitive_files)
        session.execute_write(self._clear_case_import_tx, case_id)
        imported_events = 0
        batches = 0
        for batch in chunks(graph_events, max(1, self.config.log_miner_batch_size)):
            session.execute_write(self._write_event_batch_tx, case_id, batch)
            imported_events += len(batch)
            batches += 1
        session.execute_write(
            self._mark_case_import_tx,
            case_id,
            str(log_file),
            log_hash,
            len(records),
            self.config.log_miner_schema_version,
            imported_events,
            batches,
        )
        return ImportSummary(
            imported=True,
            reused=False,
            log_hash=log_hash,
            imported_events=imported_events,
            import_batches=batches,
        )

    @staticmethod
    def _ensure_schema_tx(tx) -> None:
        tx.run("CREATE CONSTRAINT dld_case_import_id IF NOT EXISTS FOR (c:DLDCaseImport) REQUIRE c.case_id IS UNIQUE")
        tx.run("CREATE CONSTRAINT dld_log_event_id IF NOT EXISTS FOR (e:DLDLogEvent) REQUIRE e.id IS UNIQUE")
        tx.run("CREATE CONSTRAINT dld_file_id IF NOT EXISTS FOR (f:DLDFile) REQUIRE f.id IS UNIQUE")
        tx.run("CREATE CONSTRAINT dld_process_id IF NOT EXISTS FOR (p:DLDProcess) REQUIRE p.id IS UNIQUE")
        tx.run("CREATE CONSTRAINT dld_app_id IF NOT EXISTS FOR (a:DLDApp) REQUIRE a.id IS UNIQUE")
        tx.run("CREATE INDEX dld_log_event_case_time IF NOT EXISTS FOR (e:DLDLogEvent) ON (e.case_id, e.video_time_ms)")
        tx.run("CREATE INDEX dld_log_event_case_file IF NOT EXISTS FOR (e:DLDLogEvent) ON (e.case_id, e.file_path_lower)")
        tx.run("CREATE INDEX dld_log_event_case_candidate IF NOT EXISTS FOR (e:DLDLogEvent) ON (e.case_id, e.is_candidate)")
        tx.run("CREATE INDEX dld_log_event_case_risky_app IF NOT EXISTS FOR (e:DLDLogEvent) ON (e.case_id, e.is_risky_app)")
        tx.run("CREATE INDEX dld_log_event_case_sensitive IF NOT EXISTS FOR (e:DLDLogEvent) ON (e.case_id, e.is_sensitive_related)")

    @staticmethod
    def _is_reusable_import_tx(
        tx,
        case_id: str,
        log_hash: str,
        records_count: int,
        schema_version: int,
    ) -> bool:
        row = tx.run(
            """
            MATCH (c:DLDCaseImport {case_id: $case_id})
            RETURN c["log_hash"] = $log_hash
               AND c["records_count"] = $records_count
               AND c["schema_version"] = $schema_version
               AND c["import_status"] = "ready" AS reusable
            """,
            {
                "case_id": case_id,
                "log_hash": log_hash,
                "records_count": records_count,
                "schema_version": schema_version,
            },
        ).single()
        return bool(row and row.get("reusable"))

    @staticmethod
    def _clear_case_import_tx(tx, case_id: str) -> None:
        tx.run(
            """
            MATCH (c:DLDCaseImport {case_id: $case_id})
            OPTIONAL MATCH (c)-[*0..2]-(n)
            DETACH DELETE n
            """,
            {"case_id": case_id},
        )
        tx.run(
            """
            MERGE (c:DLDCaseImport {case_id: $case_id})
            SET c.import_status = "importing",
                c.import_started_at = datetime()
            """,
            {"case_id": case_id},
        )

    @staticmethod
    def _mark_case_import_tx(
        tx,
        case_id: str,
        log_file: str,
        log_hash: str,
        records_count: int,
        schema_version: int,
        imported_events: int,
        import_batches: int,
    ) -> None:
        tx.run(
            """
            MERGE (c:DLDCaseImport {case_id: $case_id})
            SET c.log_file = $log_file,
                c.log_hash = $log_hash,
                c.records_count = $records_count,
                c.schema_version = $schema_version,
                c.imported_events = $imported_events,
                c.import_batches = $import_batches,
                c.import_status = "ready",
                c.imported_at = datetime()
            """,
            {
                "case_id": case_id,
                "log_file": log_file,
                "log_hash": log_hash,
                "records_count": records_count,
                "schema_version": schema_version,
                "imported_events": imported_events,
                "import_batches": import_batches,
            },
        )

    @staticmethod
    def _write_event_batch_tx(tx, case_id: str, events: list[dict[str, Any]]) -> None:
        tx.run(
            """
            MATCH (c:DLDCaseImport {case_id: $case_id})
            UNWIND $events AS item
            MERGE (e:DLDLogEvent {id: item.id})
            SET e.case_id = $case_id,
                e.event_id = item.event_id,
                e.timestamp = item.timestamp,
                e.timestamp_ms = item.timestamp_ms,
                e.video_time_ms = item.video_time_ms,
                e.event_type = item.event_type,
                e.file_path = item.file_path,
                e.file_path_lower = item.file_path_lower,
                e.process_name = item.process_name,
                e.app_name = item.app_name,
                e.window_title = item.window_title,
                e.raw_text = item.raw_text,
                e.is_sensitive_related = item.is_sensitive_related,
                e.is_transfer_action = item.is_transfer_action,
                e.is_sink_action = item.is_sink_action,
                e.is_explicit_upload = item.is_explicit_upload,
                e.is_candidate = item.is_candidate,
                e.app_category = item.app_category,
                e.app_known = item.app_known,
                e.app_risk_hint = item.app_risk_hint,
                e.is_risky_app = item.is_risky_app
            MERGE (c)-[:HAS_EVENT]->(e)
            FOREACH (_ IN CASE WHEN item.file_path = "" THEN [] ELSE [1] END |
                MERGE (f:DLDFile {id: item.file_id})
                SET f.case_id = $case_id, f.path = item.file_path, f.path_lower = item.file_path_lower
                MERGE (e)-[:TOUCHES_FILE]->(f)
            )
            FOREACH (_ IN CASE WHEN item.process_name = "" THEN [] ELSE [1] END |
                MERGE (p:DLDProcess {id: item.process_id})
                SET p.case_id = $case_id, p.name = item.process_name
                MERGE (e)-[:BY_PROCESS]->(p)
            )
            FOREACH (_ IN CASE WHEN item.app_name = "" THEN [] ELSE [1] END |
                MERGE (a:DLDApp {id: item.app_id})
                SET a.case_id = $case_id, a.name = item.app_name
                MERGE (e)-[:IN_APP]->(a)
            )
            """,
            {"case_id": case_id, "events": events},
        )


def records_to_graph_events(case_id: str, logs: list[LogEvent], sensitive_files: list[str]) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    sensitive_paths = normalized_terms(sensitive_files)
    for event in logs:
        file_path = normalize_path(event.file_path)
        process_name = event.process_name or ""
        app_name = event.app_name or process_name
        flags = event_flags(event, sensitive_paths)
        events.append(
            {
                "id": f"{case_id}:event:{event.event_id}",
                "event_id": event.event_id,
                "timestamp": event.timestamp,
                "timestamp_ms": event.timestamp_ms,
                "video_time_ms": event.video_time_ms,
                "event_type": event.event_type,
                "file_path": file_path,
                "file_path_lower": file_path.lower(),
                "process_name": process_name,
                "app_name": app_name,
                "window_title": event.window_title,
                "raw_text": flatten_text(event.raw),
                **flags,
                "file_id": f"{case_id}:file:{file_path.lower()}",
                "process_id": f"{case_id}:process:{process_name.lower()}",
                "app_id": f"{case_id}:app:{app_name.lower()}",
            }
        )
    return events


def fingerprint_records(records: list[dict[str, Any]]) -> str:
    payload = json.dumps(records, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def normalized_terms(values: Any) -> list[str]:
    if not isinstance(values, (list, tuple, set)):
        values = [values]
    return sorted({normalize_path(value).lower() for value in values if str(value or "").strip()})


def event_flags(event: LogEvent, sensitive_paths: list[str]) -> dict[str, Any]:
    file_path = normalize_path(event.file_path).lower()
    raw_text = flatten_text(event.raw)
    text = f"{raw_text} {event.file_path} {event.process_name} {event.app_name} {event.window_title}"
    event_type = event.event_type.lower()
    extra = event.raw.get("extra") if isinstance(event.raw.get("extra"), dict) else {}
    raw_operation = str(extra.get("raw_operation") or "").lower()

    sensitive_related = (
        any(path and (path in file_path or path in raw_text.lower()) for path in sensitive_paths)
        or contains_any(text, SENSITIVE_TOKENS)
    )
    transfer_action = contains_any(text, TRANSFER_TOKENS)
    sink_action = contains_any(text, SINK_TOKENS)
    explicit_upload = event_type in {"file_selected", "file_upload", "upload", "uploaded", "upload_complete"} or raw_operation in {
        "file_selected",
        "file_upload",
        "upload",
        "send_click",
    }
    app = identify_frontend_app(event.app_name or event.process_name, event.window_title, raw_text)
    risky_app = app.risk_hint in {"external_capable", "external_capable_inferred", "unknown_external_sink"}
    return {
        "is_sensitive_related": bool(sensitive_related),
        "is_transfer_action": bool(transfer_action),
        "is_sink_action": bool(sink_action),
        "is_explicit_upload": bool(explicit_upload),
        "is_candidate": bool(sensitive_related or transfer_action or sink_action or explicit_upload),
        "app_category": app.category,
        "app_known": app.known,
        "app_risk_hint": app.risk_hint,
        "is_risky_app": bool(risky_app),
    }


def chunks(items: list[dict[str, Any]], size: int) -> list[list[dict[str, Any]]]:
    return [items[index : index + size] for index in range(0, len(items), size)]
