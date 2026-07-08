"""Dataset-case discovery for real samples under spec/data."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .sensitivity import extract_sensitive_sources


@dataclass(frozen=True)
class DataCase:
    case_dir: Path
    log_file: Path
    video_file: Path | None
    groundtruth_file: Path | None
    sensitive_files: tuple[str, ...]

    def to_input_metadata(self) -> dict[str, Any]:
        return {
            "case_dir": str(self.case_dir),
            "log_file": str(self.log_file),
            "video_file": str(self.video_file or ""),
            "groundtruth_file": str(self.groundtruth_file or ""),
            "sensitive_files_from_case": list(self.sensitive_files),
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

    return DataCase(
        case_dir=case_dir,
        log_file=log_file,
        video_file=video_file,
        groundtruth_file=groundtruth_file,
        sensitive_files=tuple(sorted(item for item in sensitive if item)),
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
    return candidates[0] if candidates else None

