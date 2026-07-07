"""Neo4j 图包边界。

图持久化是可选的，并被放在这个导出入口之后，这样核心检测流水线就能在没有数据库的情况下运行，
而部署环境仍然可以把同一份报告写入 Neo4j。
"""

from __future__ import annotations

from .config import Neo4jConfig
from .store import Neo4jGraphStore, write_report_to_neo4j

__all__ = ["Neo4jConfig", "Neo4jGraphStore", "write_report_to_neo4j"]
