"""用于检测报告的 Neo4j 持久化适配器。

图存储刻意放在检测阶段之外。它会在推理完成后把一份 JSON 报告转换成节点和关系，
这样除非请求严格模式，否则图写入失败不会污染核心分析路径。
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Protocol

from .config import Neo4jConfig


class TransactionLike(Protocol):
    def run(self, query: str, parameters: dict[str, Any] | None = None, **kwargs: Any) -> Any:
        ...


class Neo4jGraphStore:
    """将流水线证据持久化到 Neo4j 图中。"""

    def __init__(self, config: Neo4jConfig):
        self.config = config

    def write_report(self, report: dict[str, Any]) -> dict[str, Any]:
        """写入报告图并返回一份简要写入摘要。"""

        from neo4j import GraphDatabase

        driver = GraphDatabase.driver(
            self.config.uri,
            auth=(self.config.user, self.config.password),
            connection_timeout=2.0,
        )
        try:
            with driver.session(database=self.config.database) as session:
                with session.begin_transaction() as tx:
                    self._write_report_tx(tx, report, self.config.clear_session)
                    tx.commit()
        finally:
            driver.close()

        return {
            "enabled": True,
            "status": "written",
            "uri": self.config.uri,
            "database": self.config.database,
            "report_id": report.get("report_id", ""),
        }

    @staticmethod
    def _write_report_tx(tx: TransactionLike, report: dict[str, Any], clear_session: bool) -> None:
        report_id = str(report.get("report_id", "report"))
        session_id = str(report.get("event_correlator", {}).get("session_id") or report_id)

        if clear_session:
            tx.run(
                """
                MATCH (r:DLDReport {id: $report_id})
                OPTIONAL MATCH (r)-[*0..2]-(n)
                DETACH DELETE n
                """,
                {"report_id": report_id},
            )

        tx.run(
            """
            MERGE (s:DLDSession {id: $session_id})
            MERGE (r:DLDReport {id: $report_id})
            SET r.generated_at = $generated_at,
                r.conclusion = $conclusion,
                r.summary_json = $summary_json,
                r.input_json = $input_json
            MERGE (r)-[:FOR_SESSION]->(s)
            """,
            {
                "session_id": session_id,
                "report_id": report_id,
                "generated_at": report.get("generated_at", ""),
                "conclusion": report.get("conclusion", ""),
                "summary_json": _json(report.get("summary", {})),
                "input_json": _json(report.get("input", {})),
            },
        )

        _write_logs(tx, report_id, report)
        _write_observations(tx, report_id, report)
        _write_correlated_events(tx, report_id, report)
        _write_upload_candidates(tx, report_id, report)
        _write_lineage(tx, report_id, report)
        _write_datalog_facts(tx, report_id, report)
        _write_leak_paths(tx, report_id, report)


def write_report_to_neo4j(report: dict[str, Any], config: Neo4jConfig | None = None) -> dict[str, Any]:
    """如果启用则将报告写入 Neo4j；否则返回跳过摘要。"""

    config = config or Neo4jConfig.from_env()
    if not config.enabled:
        return {"enabled": False, "status": "skipped"}
    return Neo4jGraphStore(config).write_report(report)


def _write_logs(tx: TransactionLike, report_id: str, report: dict[str, Any]) -> None:
    logs = report.get("event_correlator", {}).get("raw_log_events", [])
    for index, item in enumerate(logs):
        event_id = str(item.get("event_id") or f"log_{index}")
        tx.run(
            """
            MATCH (r:DLDReport {id: $report_id})
            MERGE (e:DLDLogEvent {id: $id})
            SET e.timestamp = $timestamp,
                e.event_type = $event_type,
                e.file_path = $file_path,
                e.app_name = $app_name,
                e.raw_json = $raw_json
            MERGE (r)-[:HAS_LOG_EVENT]->(e)
            """,
            {
                "report_id": report_id,
                "id": event_id,
                "timestamp": item.get("timestamp", ""),
                "event_type": item.get("event_type", ""),
                "file_path": item.get("file_path", ""),
                "app_name": _app_name(item),
                "raw_json": _json(item),
            },
        )
        _merge_file_edge(tx, "DLDLogEvent", event_id, item.get("file_path", ""), "TOUCHES_FILE")


def _write_observations(tx: TransactionLike, report_id: str, report: dict[str, Any]) -> None:
    observations = report.get("frame_analyzer", {}).get("observations", [])
    for item in observations:
        observation_id = str(item.get("observation_id", ""))
        tx.run(
            """
            MATCH (r:DLDReport {id: $report_id})
            MERGE (o:DLDFrameObservation {id: $id})
            SET o.start_ms = $start_ms,
                o.end_ms = $end_ms,
                o.app_name = $app_name,
                o.operation_type = $operation_type,
                o.resource = $resource,
                o.confidence = $confidence,
                o.description = $description
            MERGE (r)-[:HAS_FRAME_OBSERVATION]->(o)
            """,
            {
                "report_id": report_id,
                "id": observation_id,
                "start_ms": item.get("start_ms", 0),
                "end_ms": item.get("end_ms", 0),
                "app_name": item.get("app_name", ""),
                "operation_type": item.get("operation_type", ""),
                "resource": item.get("resource", ""),
                "confidence": item.get("confidence", 0.0),
                "description": item.get("description", ""),
            },
        )
        _merge_file_edge(tx, "DLDFrameObservation", observation_id, item.get("resource", ""), "OBSERVES_FILE")


def _write_correlated_events(tx: TransactionLike, report_id: str, report: dict[str, Any]) -> None:
    for item in report.get("event_correlator", {}).get("correlated_events", []):
        event_id = str(item.get("event_id", ""))
        tx.run(
            """
            MATCH (r:DLDReport {id: $report_id})
            MERGE (e:DLDCorrelatedEvent {id: $id})
            SET e.timestamp = $timestamp,
                e.event_type = $event_type,
                e.app_name = $app_name,
                e.operation_type = $operation_type,
                e.behavior_category = $behavior_category,
                e.confidence = $confidence,
                e.evidence_refs = $evidence_refs
            MERGE (r)-[:HAS_CORRELATED_EVENT]->(e)
            """,
            {
                "report_id": report_id,
                "id": event_id,
                "timestamp": item.get("timestamp", ""),
                "event_type": item.get("event_type", ""),
                "app_name": item.get("app_name", ""),
                "operation_type": item.get("operation_type", ""),
                "behavior_category": item.get("behavior_category", ""),
                "confidence": item.get("confidence", 0.0),
                "evidence_refs": item.get("evidence_refs", []),
            },
        )
        _merge_file_edge(tx, "DLDCorrelatedEvent", event_id, item.get("original_file", ""), "ORIGINAL_FILE")
        _merge_file_edge(tx, "DLDCorrelatedEvent", event_id, item.get("current_file", ""), "CURRENT_FILE")


def _write_upload_candidates(tx: TransactionLike, report_id: str, report: dict[str, Any]) -> None:
    for item in report.get("event_correlator", {}).get("upload_candidates", []):
        candidate_id = str(item.get("candidate_id", ""))
        tx.run(
            """
            MATCH (r:DLDReport {id: $report_id})
            MERGE (u:DLDUploadCandidate {id: $id})
            SET u.timestamp = $timestamp,
                u.app_name = $app_name,
                u.sink_type = $sink_type,
                u.risk_level = $risk_level,
                u.confidence = $confidence
            MERGE (r)-[:HAS_UPLOAD_CANDIDATE]->(u)
            """,
            {
                "report_id": report_id,
                "id": candidate_id,
                "timestamp": item.get("timestamp", ""),
                "app_name": item.get("app_name", ""),
                "sink_type": item.get("sink_type", ""),
                "risk_level": item.get("risk_level", ""),
                "confidence": item.get("confidence", 0.0),
            },
        )
        _merge_file_edge(tx, "DLDUploadCandidate", candidate_id, item.get("original_file", ""), "ORIGINAL_FILE")
        _merge_file_edge(tx, "DLDUploadCandidate", candidate_id, item.get("current_file", ""), "CURRENT_FILE")


def _write_lineage(tx: TransactionLike, report_id: str, report: dict[str, Any]) -> None:
    mappings = report.get("event_correlator", {}).get("file_lineage", {}).get("direct_file_mappings", {})
    for derived, source in mappings.items():
        tx.run(
            """
            MATCH (r:DLDReport {id: $report_id})
            MERGE (derived:DLDFile {path: $derived})
            MERGE (source:DLDFile {path: $source})
            MERGE (derived)-[:DERIVED_FROM {report_id: $report_id}]->(source)
            MERGE (r)-[:HAS_FILE]->(derived)
            MERGE (r)-[:HAS_FILE]->(source)
            """,
            {"report_id": report_id, "derived": derived, "source": source},
        )


def _write_datalog_facts(tx: TransactionLike, report_id: str, report: dict[str, Any]) -> None:
    for item in report.get("event_correlator", {}).get("datalog_facts", []):
        fact_id = _fact_id(item)
        tx.run(
            """
            MATCH (r:DLDReport {id: $report_id})
            MERGE (f:DLDDatalogFact {id: $id})
            SET f.relation = $relation,
                f.args_json = $args_json
            MERGE (r)-[:HAS_DATALOG_FACT]->(f)
            """,
            {
                "report_id": report_id,
                "id": fact_id,
                "relation": item.get("relation", ""),
                "args_json": _json(item.get("args", [])),
            },
        )


def _write_leak_paths(tx: TransactionLike, report_id: str, report: dict[str, Any]) -> None:
    for index, item in enumerate(report.get("leak_reasoner", {}).get("leak_paths", [])):
        path_id = str(item.get("end_op") or f"leak_path_{index}")
        tx.run(
            """
            MATCH (r:DLDReport {id: $report_id})
            MERGE (p:DLDLeakPath {id: $id})
            SET p.start_op = $start_op,
                p.end_op = $end_op,
                p.leaking_proc = $leaking_proc,
                p.leaked_file = $leaked_file,
                p.leak_channel = $leak_channel,
                p.full_path = $full_path
            MERGE (r)-[:HAS_LEAK_PATH]->(p)
            """,
            {
                "report_id": report_id,
                "id": path_id,
                "start_op": item.get("start_op", ""),
                "end_op": item.get("end_op", ""),
                "leaking_proc": item.get("leaking_proc", ""),
                "leaked_file": item.get("leaked_file", ""),
                "leak_channel": item.get("leak_channel", ""),
                "full_path": item.get("full_path", ""),
            },
        )
        _merge_file_edge(tx, "DLDLeakPath", path_id, item.get("leaked_file", ""), "LEAKED_FILE")


def _merge_file_edge(tx: TransactionLike, label: str, node_id: str, file_path: Any, relationship: str) -> None:
    path = str(file_path or "")
    if not path:
        return
    query = f"""
    MATCH (n:{label} {{id: $node_id}})
    MERGE (f:DLDFile {{path: $path}})
    MERGE (n)-[:{relationship}]->(f)
    """
    tx.run(query, {"node_id": node_id, "path": path})


def _app_name(item: dict[str, Any]) -> str:
    process = item.get("process_info") if isinstance(item.get("process_info"), dict) else {}
    return str(item.get("app_name") or process.get("process_name") or "")


def _fact_id(item: dict[str, Any]) -> str:
    text = _json(item)
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]
    return f"fact_{digest}"


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)
