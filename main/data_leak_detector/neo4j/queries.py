"""Cypher queries used by the Neo4j log miner."""

from __future__ import annotations


class Neo4jLogQueries:
    @staticmethod
    def candidate_event_ids(session, case_id: str) -> list[str]:
        return session.execute_read(Neo4jLogQueries._candidate_event_ids_tx, case_id)

    @staticmethod
    def active_apps_for_events(session, case_id: str, event_ids: list[str], radius_ms: int) -> dict[str, tuple[str, ...]]:
        return session.execute_read(Neo4jLogQueries._active_apps_for_events_tx, case_id, event_ids, radius_ms)

    @staticmethod
    def sensitive_activity_intervals(session, case_id: str, whitelisted_apps: tuple[str, ...]) -> list[dict[str, object]]:
        return session.execute_read(
            Neo4jLogQueries._sensitive_activity_intervals_tx,
            case_id,
            list(whitelisted_apps),
        )

    @staticmethod
    def _candidate_event_ids_tx(tx, case_id: str) -> list[str]:
        rows = tx.run(
            """
            MATCH (c:DLDCaseImport {case_id: $case_id})-[:HAS_EVENT]->(e:DLDLogEvent)
            WHERE e.video_time_ms >= 0 AND e.is_candidate = true
            RETURN e.event_id AS event_id
            ORDER BY e.video_time_ms ASC
            """,
            {"case_id": case_id},
        )
        return [str(row["event_id"]) for row in rows]

    @staticmethod
    def _active_apps_for_events_tx(
        tx,
        case_id: str,
        event_ids: list[str],
        radius_ms: int,
    ) -> dict[str, tuple[str, ...]]:
        if not event_ids:
            return {}
        rows = tx.run(
            """
            MATCH (c:DLDCaseImport {case_id: $case_id})-[:HAS_EVENT]->(e:DLDLogEvent)
            WHERE e.event_id IN $event_ids
            OPTIONAL MATCH (c)-[:HAS_EVENT]->(near:DLDLogEvent)
            WHERE near.video_time_ms >= e.video_time_ms - $radius_ms
              AND near.video_time_ms <= e.video_time_ms + $radius_ms
              AND (near.is_risky_app = true OR near.is_sink_action = true OR near.is_transfer_action = true)
            WITH e, collect(DISTINCT coalesce(near.app_name, near.process_name, "")) AS apps
            RETURN e.event_id AS event_id, [app IN apps WHERE app <> ""] AS active_apps
            ORDER BY e.video_time_ms ASC
            """,
            {"case_id": case_id, "event_ids": event_ids, "radius_ms": radius_ms},
        )
        return {str(row["event_id"]): tuple(str(app) for app in row["active_apps"]) for row in rows}

    @staticmethod
    def _sensitive_activity_intervals_tx(tx, case_id: str, whitelisted_apps: list[str]) -> list[dict[str, object]]:
        rows = tx.run(
            """
            MATCH (c:DLDCaseImport {case_id: $case_id})-[:HAS_EVENT]->(session_event:DLDLogEvent)
            WHERE session_event.video_time_ms >= 0
            WITH c, max(session_event.video_time_ms) AS session_end_ms
            MATCH (c)-[:HAS_EVENT]->(e:DLDLogEvent)
            WHERE e.video_time_ms >= 0 AND e.is_sensitive_related = true
            WITH c, session_end_ms,
                 coalesce(nullif(e.file_path_lower, ""), "sensitive_context") AS sensitive_file,
                 min(e.video_time_ms) AS start_ms,
                 collect(DISTINCT e.video_time_ms) AS anchors
            OPTIONAL MATCH (c)-[:HAS_EVENT]->(close_event:DLDLogEvent)
            WHERE close_event.video_time_ms >= start_ms
              AND close_event.event_type IN ["closed", "close", "file_closed", "file_close"]
              AND coalesce(nullif(close_event.file_path_lower, ""), "sensitive_context") = sensitive_file
            WITH c, sensitive_file, start_ms, anchors, session_end_ms,
                 min(close_event.video_time_ms) AS close_ms
            WITH c, sensitive_file, start_ms, anchors,
                 coalesce(close_ms, session_end_ms) AS end_ms
            OPTIONAL MATCH (c)-[:HAS_EVENT]->(active:DLDLogEvent)
            WHERE active.video_time_ms >= start_ms
              AND active.video_time_ms <= end_ms
              AND coalesce(active.app_name, active.process_name, "") <> ""
              AND NOT toLower(coalesce(active.app_name, active.process_name, "")) IN $whitelisted_apps
            RETURN sensitive_file, start_ms, end_ms, anchors,
                   collect(DISTINCT coalesce(active.app_name, active.process_name, "")) AS active_apps
            ORDER BY start_ms ASC
            """,
            {"case_id": case_id, "whitelisted_apps": whitelisted_apps},
        )
        return [dict(row) for row in rows]
