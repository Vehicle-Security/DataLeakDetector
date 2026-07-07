"""由环境变量驱动的 Neo4j 配置。

Neo4j 是带副作用的输出，而不是必需的检测依赖。本模块在一个地方读取 .env 和进程变量，
这样 CLI 参数、测试和本地辅助脚本就能共享同一套配置契约。
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Neo4jConfig:
    """可选 Neo4j 图汇聚点的运行时配置。"""

    enabled: bool = False
    uri: str = "bolt://localhost:7687"
    user: str = "neo4j"
    password: str = "data-leak-detector"
    database: str = "neo4j"
    strict: bool = False
    clear_session: bool = False

    @classmethod
    def from_env(cls) -> "Neo4jConfig":
        _load_dotenv()
        return cls(
            enabled=_env_bool("DLD_NEO4J_ENABLED", False),
            uri=os.getenv("DLD_NEO4J_URI", "bolt://localhost:7687"),
            user=os.getenv("DLD_NEO4J_USER", "neo4j"),
            password=os.getenv("DLD_NEO4J_PASSWORD", "data-leak-detector"),
            database=os.getenv("DLD_NEO4J_DATABASE", "neo4j"),
            strict=_env_bool("DLD_NEO4J_STRICT", False),
            clear_session=_env_bool("DLD_NEO4J_CLEAR_SESSION", False),
        )


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _load_dotenv() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    root = Path(__file__).resolve().parents[3]
    load_dotenv(root / ".env", override=False)
