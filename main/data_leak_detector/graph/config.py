"""Environment-backed Neo4j configuration.

Neo4j is a side-effecting output, not a required detection dependency. This
module reads .env and process variables in one place so CLI flags, tests, and
local helper scripts share the same configuration contract.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Neo4jConfig:
    """Runtime configuration for the optional Neo4j graph sink."""

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
