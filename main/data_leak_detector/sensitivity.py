"""Load the initial sensitive-source set for detection."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

from .io import normalize_path, read_text


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SENSITIVE_FILES_CONFIG_PATH = REPO_ROOT / "spec" / "config" / "sensitive_files..json"
STAGE_SENSITIVE_FILES_CONFIG_DIR = REPO_ROOT / "spec" / "config"


def resolve_sensitive_files_config(path: str | Path, configured_path: str | Path | None = None) -> Path | None:
    """Select the stage-specific source list for NAS samples unless overridden."""

    if configured_path:
        return Path(configured_path)

    sample_path = Path(path).resolve()
    for candidate in (sample_path, *sample_path.parents):
        match = re.fullmatch(r"stage(\d+)", candidate.name, flags=re.IGNORECASE)
        if not match or candidate.parent.name.lower() != "nas_samples":
            continue
        stage_config = STAGE_SENSITIVE_FILES_CONFIG_DIR / f"sensitive_files_{match.group(1)}.json"
        if not stage_config.exists():
            raise FileNotFoundError(f"sensitive files config does not exist for {candidate.name}: {stage_config}")
        return stage_config
    return None


def load_sensitive_files_config(path: str | Path | None = None) -> tuple[str, ...]:
    """Load the only initial sensitive-source set used by detection.

    Log review may establish additional original sources, which must first be
    persisted in this configuration. Groundtruth never contributes sources.
    Copies, conversions, screenshots, and other descendants are established by
    lineage reasoning and are never persisted as initial sources.
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
