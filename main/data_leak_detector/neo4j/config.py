"""Runtime configuration for the optional Neo4j log miner."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Neo4jConfig:
    """Neo4j is only used as an optional log-mining backend."""

    uri: str = "bolt://localhost:7687"
    user: str = "neo4j"
    password: str = "data-leak-detector"
    database: str = "neo4j"
    log_miner_enabled: bool = False
    log_miner_strict: bool = False
    reuse_import: bool = True
    log_miner_schema_version: int = 1
    log_miner_batch_size: int = 2_000

    @classmethod
    def from_env(cls) -> "Neo4jConfig":
        _load_dotenv()
        return cls(
            uri=os.getenv("DLD_NEO4J_URI", "bolt://localhost:7687"),
            user=os.getenv("DLD_NEO4J_USER", "neo4j"),
            password=os.getenv("DLD_NEO4J_PASSWORD", "data-leak-detector"),
            database=os.getenv("DLD_NEO4J_DATABASE", "neo4j"),
            log_miner_enabled=_env_bool("DLD_NEO4J_LOG_MINER", False),
            log_miner_strict=_env_bool("DLD_NEO4J_LOG_MINER_STRICT", False),
            reuse_import=_env_bool("DLD_NEO4J_REUSE_IMPORT", True),
            log_miner_schema_version=_env_int("DLD_NEO4J_LOG_MINER_SCHEMA_VERSION", 1),
            log_miner_batch_size=_env_int("DLD_NEO4J_LOG_MINER_BATCH_SIZE", 2_000),
        )

    def with_overrides(
        self,
        *,
        log_miner_enabled: bool | None = None,
        log_miner_strict: bool | None = None,
        reuse_import: bool | None = None,
    ) -> "Neo4jConfig":
        return Neo4jConfig(
            uri=self.uri,
            user=self.user,
            password=self.password,
            database=self.database,
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
