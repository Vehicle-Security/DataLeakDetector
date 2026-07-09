"""Dataset-case discovery for real samples under spec/data."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any
import re

from .io import parse_timestamp_ms, read_text
from .sensitivity import extract_sensitive_sources


@dataclass(frozen=True)
class DataCase:
    case_dir: Path
    log_file: Path
    video_file: Path | None
    groundtruth_file: Path | None
    sensitive_files: tuple[str, ...]
    recording_start_ms: int = 0

    def to_input_metadata(self) -> dict[str, Any]:
        return {
            "case_dir": str(self.case_dir),
            "log_file": str(self.log_file),
            "video_file": str(self.video_file or ""),
            "groundtruth_file": str(self.groundtruth_file or ""),
            "sensitive_files_from_case": list(self.sensitive_files),
            "recording_start_ms": self.recording_start_ms,
        }


def discover_data_case(path: str | Path) -> DataCase:
    """Resolve a NAS-style sample directory into pipeline input files."""

    case_dir = Path(path)
    if case_dir.is_file():
        case_dir = case_dir.parent
    case_dir = case_dir.resolve()
    if not case_dir.exists():
        raise FileNotFoundError(f"case path does not exist: {case_dir}")

    log_file = _choose_log_file(case_dir)
    video_file = _choose_video_file(case_dir)
    groundtruth_file = case_dir / "groundtruth.json"
    if not groundtruth_file.exists():
        groundtruth_file = None

    sensitive = set(extract_sensitive_sources(groundtruth_file))
    recording_start_ms = _recording_start_ms(case_dir, groundtruth_file)

    return DataCase(
        case_dir=case_dir,
        log_file=log_file,
        video_file=video_file,
        groundtruth_file=groundtruth_file,
        sensitive_files=tuple(sorted(item for item in sensitive if item)),
        recording_start_ms=recording_start_ms,
    )


def _choose_log_file(case_dir: Path) -> Path:
    candidates = [
        case_dir / "logs" / "logs.json",
        case_dir / "logs" / "keyevents.json",
        case_dir / "keyevents.json",
        case_dir / "logs.json",
    ]
    for candidate in candidates:
        if candidate.exists() and candidate.stat().st_size > 2:
            return candidate
    found = sorted(case_dir.glob("**/keyevents.json")) or sorted(case_dir.glob("**/logs.json"))
    if found:
        return found[0]
    raise FileNotFoundError(f"no logs.json or keyevents.json found under {case_dir}")


def _choose_video_file(case_dir: Path) -> Path | None:
    video_dir = case_dir / "video"
    candidates = sorted(video_dir.glob("*.mp4")) if video_dir.exists() else []
    if not candidates:
        candidates = sorted(case_dir.glob("**/*.mp4"))
    indexed = _video_from_index(case_dir, candidates)
    if indexed is not None:
        return indexed
    return candidates[0] if candidates else None


def _video_from_index(case_dir: Path, candidates: list[Path]) -> Path | None:
    index_file = case_dir / "INDEX.md"
    if not index_file.exists() or not candidates:
        return None
    text = index_file.read_text(encoding="utf-8", errors="ignore")

    for match in re.findall(r"`([^`]+\.mp4)`", text, flags=re.IGNORECASE):
        path = (case_dir / match).resolve()
        if path.exists():
            return path

    session_match = re.search(r"Session ID\*\*:\s*([0-9_]+)", text)
    if session_match:
        session_id = session_match.group(1)
        for candidate in candidates:
            if session_id in candidate.name:
                return candidate
    return None


def _recording_start_ms(case_dir: Path, groundtruth_file: Path | None) -> int:
    if groundtruth_file is not None:
        try:
            payload = json.loads(read_text(groundtruth_file))
            timestamp = payload.get("recording_start_time") if isinstance(payload, dict) else ""
            parsed = parse_timestamp_ms(timestamp)
            if parsed:
                return parsed
        except (OSError, ValueError, TypeError):
            pass

    index_file = case_dir / "INDEX.md"
    if index_file.exists():
        text = index_file.read_text(encoding="utf-8", errors="ignore")
        match = re.search(r"Recording Time\*\*:\s*([0-9][^\r\n]+)", text)
        if match:
            return parse_timestamp_ms(match.group(1).strip())
    return 0
