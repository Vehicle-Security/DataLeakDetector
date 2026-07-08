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
    log_miner_enabled: bool = False
    log_miner_strict: bool = False
    reuse_import: bool = True
    log_miner_schema_version: int = 1
    log_miner_batch_size: int = 2_000

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
            log_miner_enabled=_env_bool("DLD_NEO4J_LOG_MINER", False),
            log_miner_strict=_env_bool("DLD_NEO4J_LOG_MINER_STRICT", False),
            reuse_import=_env_bool("DLD_NEO4J_REUSE_IMPORT", True),
            log_miner_schema_version=_env_int("DLD_NEO4J_LOG_MINER_SCHEMA_VERSION", 1),
            log_miner_batch_size=_env_int("DLD_NEO4J_LOG_MINER_BATCH_SIZE", 2_000),
        )

    def with_overrides(
        self,
        *,
        enabled: bool | None = None,
        strict: bool | None = None,
        clear_session: bool | None = None,
        log_miner_enabled: bool | None = None,
        log_miner_strict: bool | None = None,
        reuse_import: bool | None = None,
    ) -> "Neo4jConfig":
        return Neo4jConfig(
            enabled=self.enabled if enabled is None else enabled,
            uri=self.uri,
            user=self.user,
            password=self.password,
            database=self.database,
            strict=self.strict if strict is None else strict,
            clear_session=self.clear_session if clear_session is None else clear_session,
            log_miner_enabled=self.log_miner_enabled if log_miner_enabled is None else log_miner_enabled,
            log_miner_strict=self.log_miner_strict if log_miner_strict is None else log_miner_strict,
            reuse_import=self.reuse_import if reuse_import is None else reuse_import,
            log_miner_schema_version=self.log_miner_schema_version,
            log_miner_batch_size=self.log_miner_batch_size,
        )


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


def _load_dotenv() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    root = Path(__file__).resolve().parents[3]
    load_dotenv(root / ".env", override=False)
