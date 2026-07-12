"""Load the initial sensitive-source set for detection."""

from __future__ import annotations

import json
import os
from pathlib import Path

from .io import normalize_path, read_text


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SENSITIVE_FILES_CONFIG_PATH = REPO_ROOT / "spec" / "config" / "sensitive_files..json"


def load_sensitive_files_config(path: str | Path | None = None) -> tuple[str, ...]:
    """Load the only initial sensitive-source set used by detection.

    The configuration contains original sensitive files only. Copies,
    conversions, screenshots, and other descendants are established later by
    lineage reasoning and are never sent to log mining, VLM, or Datalog as
    initial source context.
    """

    configured_path = path or os.getenv("DLD_SENSITIVE_FILES_CONFIG") or DEFAULT_SENSITIVE_FILES_CONFIG_PATH
    target = Path(configured_path)
    if not target.exists():
        raise FileNotFoundError(f"sensitive files config does not exist: {target}")
    try:
        payload = json.loads(read_text(target))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid sensitive files config: {target}") from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("sensitive_files"), list):
        raise ValueError(f"sensitive files config must contain a sensitive_files array: {target}")

    sources: list[str] = []
    seen: set[str] = set()
    for value in payload["sensitive_files"]:
        if not isinstance(value, str):
            continue
        normalized = normalize_path(value)
        key = normalized.lower()
        if normalized and key not in seen:
            seen.add(key)
            sources.append(normalized)
    return tuple(sources)
