"""Neo4j adapters for optional persistence and log mining.

The detector can run without Neo4j. This package contains only the backend
integration: environment config, report graph writing, log import, and Cypher
queries used by the optional log miner.
"""

from __future__ import annotations

from .config import Neo4jConfig
from .store import Neo4jGraphStore, write_report_to_neo4j

__all__ = [
    "Neo4jConfig",
    "Neo4jGraphStore",
    "write_report_to_neo4j",
]
