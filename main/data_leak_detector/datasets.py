"""Dataset-case discovery for real samples under spec/data."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import re

from .io import parse_timestamp_ms
from .sensitivity import load_sensitive_files_config, resolve_sensitive_files_config

GROUNDTRUTH_FILENAMES = ("groundtruth.json", "groundtrutn.json")


@dataclass(frozen=True)
class DataSession:
    session_id: str
    session_dir: Path
    log_file: Path
    video_file: Path | None
    recording_start_ms: int = 0

    def to_input_metadata(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "session_dir": str(self.session_dir),
            "log_file": str(self.log_file),
            "video_file": str(self.video_file or ""),
            "recording_start_ms": self.recording_start_ms,
        }


@dataclass(frozen=True)
class DataCase:
    case_id: str
    case_dir: Path
    log_file: Path
    video_file: Path | None
    groundtruth_file: Path | None
    sensitive_files: tuple[str, ...]
    recording_start_ms: int = 0
    case_relative_path: str = ""
    case_name: str = ""
    groundtruth_status: str = ""
    nearest_ancestor_groundtruth_file: Path | None = None
    sessions: tuple[DataSession, ...] = ()

    def to_input_metadata(self) -> dict[str, Any]:
        sessions = self.sessions or (
            DataSession(
                session_id=self.case_dir.name,
                session_dir=self.case_dir,
                log_file=self.log_file,
                video_file=self.video_file,
                recording_start_ms=self.recording_start_ms,
            ),
        )
        return {
            "case_id": self.case_id,
            "case_name": self.case_name or self.case_dir.name,
            "case_relative_path": self.case_relative_path or self.case_id,
            "case_dir": str(self.case_dir),
            "log_file": str(self.log_file),
            "video_file": str(self.video_file or ""),
            "groundtruth_file": str(self.groundtruth_file or ""),
            "groundtruth_status": self.groundtruth_status,
            "nearest_ancestor_groundtruth_file": str(self.nearest_ancestor_groundtruth_file or ""),
            "sensitive_files_from_case": list(self.sensitive_files),
            "recording_start_ms": self.recording_start_ms,
            "session_count": len(sessions),
            "sessions": [item.to_input_metadata() for item in sessions],
            "log_files": [str(item.log_file) for item in sessions],
            "video_files": [str(item.video_file or "") for item in sessions],
        }


def discover_data_case_directories(root: str | Path) -> list[Path]:
    """Find logical cases, grouping direct ``session_*`` children under their case."""

    root_path = Path(root)
    if not root_path.is_dir():
        raise FileNotFoundError(f"case root does not exist: {root_path}")
    candidates = [root_path, *root_path.rglob("*")]
    composite_cases = {
        path
        for path in candidates
        if path.is_dir() and _is_composite_case_dir(path)
    }
    atomic_cases = {
        path
        for path in candidates
        if path.is_dir() and (path / "logs").is_dir() and (path / "video").is_dir()
    }
    grouped_sessions = {
        session
        for case_dir in composite_cases
        for session in _direct_session_directories(case_dir)
    }
    logical_cases = composite_cases | (atomic_cases - grouped_sessions)
    return sorted(logical_cases, key=lambda path: str(path).lower())


def discover_data_case(
    path: str | Path,
    *,
    case_root: str | Path | None = None,
    inherit_ancestor_groundtruth: bool = False,
    sensitive_files_config: str | Path | None = None,
) -> DataCase:
    """Resolve a NAS-style sample directory into pipeline input files."""

    case_dir = Path(path)
    if case_dir.is_file():
        case_dir = case_dir.parent
    case_dir = case_dir.resolve()
    if not case_dir.exists():
        raise FileNotFoundError(f"case path does not exist: {case_dir}")
    if case_dir.name.startswith("session_") and _is_composite_case_dir(case_dir.parent):
        case_dir = case_dir.parent

    case_relative_path = data_case_id(case_dir, case_root)
    session_dirs = _direct_session_directories(case_dir) if _is_composite_case_dir(case_dir) else [case_dir]
    sessions = tuple(_discover_data_session(path) for path in session_dirs)
    log_file = sessions[0].log_file
    video_file = sessions[0].video_file
    groundtruth_candidate = _groundtruth_file(case_dir)
    groundtruth_file = groundtruth_candidate if groundtruth_candidate.exists() else None
    nearest_ancestor_groundtruth_file = None if groundtruth_file else _nearest_ancestor_groundtruth(case_dir, case_root)
    inherited_groundtruth = False
    if groundtruth_file is None and inherit_ancestor_groundtruth and nearest_ancestor_groundtruth_file:
        groundtruth_file = nearest_ancestor_groundtruth_file
        inherited_groundtruth = True
    if groundtruth_file:
        groundtruth_status = "inherited_from_ancestor" if inherited_groundtruth else "available"
    elif nearest_ancestor_groundtruth_file:
        groundtruth_status = "missing_current_directory_with_ancestor_groundtruth"
    else:
        groundtruth_status = "missing"

    sensitive = load_sensitive_files_config(resolve_sensitive_files_config(case_dir, sensitive_files_config))
    recording_starts = [item.recording_start_ms for item in sessions if item.recording_start_ms]
    recording_start_ms = min(recording_starts) if recording_starts else _recording_start_ms(case_dir)

    return DataCase(
        case_id=case_relative_path,
        case_dir=case_dir,
        log_file=log_file,
        video_file=video_file,
        groundtruth_file=groundtruth_file,
        sensitive_files=sensitive,
        recording_start_ms=recording_start_ms,
        case_relative_path=case_relative_path,
        case_name=case_dir.name,
        groundtruth_status=groundtruth_status,
        nearest_ancestor_groundtruth_file=nearest_ancestor_groundtruth_file,
        sessions=sessions,
    )


def _discover_data_session(session_dir: Path) -> DataSession:
    return DataSession(
        session_id=session_dir.name,
        session_dir=session_dir,
        log_file=_choose_log_file(session_dir),
        video_file=_choose_video_file(session_dir),
        recording_start_ms=_recording_start_ms(session_dir),
    )


def _direct_session_directories(case_dir: Path) -> list[Path]:
    return sorted(
        (
            path
            for path in case_dir.glob("session_*")
            if path.is_dir() and (path / "logs").is_dir() and (path / "video").is_dir()
        ),
        key=lambda path: path.name.lower(),
    )


def _is_composite_case_dir(case_dir: Path) -> bool:
    return bool(_direct_session_directories(case_dir))


def data_case_id(path: str | Path, case_root: str | Path | None = None) -> str:
    """Return a stable, human-readable case id.

    Batch/release runs use paths relative to the case root so duplicated case
    directory names stay distinct in reports and precompute caches.
    """

    case_dir = Path(path)
    if case_dir.is_file():
        case_dir = case_dir.parent
    case_dir = case_dir.resolve()
    if case_root is None:
        return case_dir.name
    try:
        relative = case_dir.relative_to(Path(case_root).resolve())
    except ValueError:
        return case_dir.name
    text = relative.as_posix().strip("/")
    return text or case_dir.name


def _choose_log_file(case_dir: Path) -> Path:
    candidates = [
        case_dir / "logs" / "keyevents.json",
        case_dir / "keyevents.json",
        case_dir / "logs" / "logs.json",
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


def _recording_start_ms(case_dir: Path) -> int:
    index_file = case_dir / "INDEX.md"
    if index_file.exists():
        text = index_file.read_text(encoding="utf-8", errors="ignore")
        match = re.search(r"Recording Time\*\*:\s*([0-9][^\r\n]+)", text)
        if match:
            return parse_timestamp_ms(match.group(1).strip())
    video_file = _choose_video_file(case_dir)
    if video_file is not None:
        match = re.search(r"recording[_-](\d{8})[_-](\d{6})", video_file.stem, flags=re.IGNORECASE)
        if match:
            date, time = match.groups()
            return parse_timestamp_ms(
                f"{date[:4]}-{date[4:6]}-{date[6:]} {time[:2]}:{time[2:4]}:{time[4:]}"
            )
    return 0


def _nearest_ancestor_groundtruth(case_dir: Path, case_root: str | Path | None) -> Path | None:
    if case_root is None:
        return None
    try:
        stop_at = Path(case_root).resolve()
        current = case_dir.resolve().parent
    except OSError:
        return None
    while True:
        candidate = _groundtruth_file(current)
        if candidate.exists():
            return candidate
        if current == stop_at or current.parent == current:
            return None
        try:
            current.relative_to(stop_at)
        except ValueError:
            return None
        current = current.parent


def _groundtruth_file(case_dir: Path) -> Path:
    for filename in GROUNDTRUTH_FILENAMES:
        candidate = case_dir / filename
        if candidate.exists():
            return candidate
    return case_dir / "groundtruth.json"
