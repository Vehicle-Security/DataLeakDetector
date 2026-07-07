"""Neo4j graph package boundary.

Graph persistence is optional and lives behind this export surface so the core
detection pipeline can run without a database while deployments can still write
the same report into Neo4j.
"""

from __future__ import annotations

from .config import Neo4jConfig
from .store import Neo4jGraphStore, write_report_to_neo4j

__all__ = ["Neo4jConfig", "Neo4jGraphStore", "write_report_to_neo4j"]
