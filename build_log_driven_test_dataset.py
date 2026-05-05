#!/usr/bin/env python3
"""
Build a curated log-driven E2E test dataset from stage folders.

The output dataset is made of symlinks to the original cases, so large videos are
not copied. The selection is for deployment-cost benchmarking, not accuracy
scoring.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi"}
DOC_EXTENSIONS = {
    ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx", ".pdf",
    ".zip", ".rar", ".7z", ".csv",
}
SOURCE_EXTENSIONS = {
    ".txt", ".md", ".json", ".py", ".java", ".js", ".ts",
    ".cpp", ".c", ".h", ".go", ".rs", ".sql",
}
SENSITIVE_KEYWORDS = [
    "\u85aa\u8d44", "\u5de5\u8d44", "\u673a\u5bc6", "\u7edd\u5bc6",
    "\u5408\u540c", "\u8d22\u52a1", "\u5ba2\u6237", "\u5bc6\u7801",
    "\u6838\u5fc3", "\u79d8\u5bc6", "\u5185\u90e8", "\u62a5\u8868",
    "\u9884\u7b97", "\u6218\u7565", "\u89c4\u5212",
    "\u4f1a\u8bae\u7eaa\u8981", "\u5458\u5de5", "\u4ea7\u54c1",
    "\u65b9\u6848", "secret", "confidential", "contract",
    "salary", "finance", "internal",
]
RELEVANT_EVENT_TYPES = {
    "created", "modified", "renamed", "moved", "opened",
    "file_selected", "upload_detected", "clipboard_text", "clipboard_image",
    "browser_file_access",
}
NOISE_SEGMENTS = (
    "/appdata/", "/cache/", "/cookies/", "/history/", "/temp/", "/tmp/",
    "/node_modules/", "/.git/", "/windows/", "/program files/",
    "/screenmonitor/", "/winows_monitor/", "/windows_monitor/",
    "/recordings/session_", "/logs/", "/video/",
)
NOISE_BASENAMES = {
    "logs.json", "keyevents.json", "index.md", "global.json", "global.dat",
    "config.ini", "onceflag.ini", "personalsetting.xml", "appsettingapp.dat",
    "amcache.hve",
}
NOISE_SUFFIXES = (
    ".sqlite", ".sqlite3", ".db", ".db-journal", ".db-wal", ".wal",
    ".journal", ".lock", ".dat", ".ini", ".hve", ".log", ".tmp", ".lnk",
    ".crdownload",
)


@dataclass
class CaseInfo:
    case: str
    stage: str
    category: str
    log_events: int
    candidate_events: int
    video_sec: float | None
    video_size_mb: float
    score: float
    selected: bool = False
    exclude_reason: str = ""
    link: str = ""


def norm_path(value: str) -> str:
    text = str(value or "").strip().replace("\\", "/")
    while "//" in text:
        text = text.replace("//", "/")
    return text


def canonical_path(value: str) -> str:
    return re.sub(r"(?i)(?:\.baiduyun)?\.uploading\.cfg$", "", norm_path(value))


def slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")[:180] or "case"


def stage_name(root: Path) -> str:
    match = re.search(r"stage\d+", root.name, flags=re.IGNORECASE)
    return match.group(0).lower() if match else root.name.lower()


def find_video(case_dir: Path) -> Path | None:
    video_dir = case_dir / "video"
    if not video_dir.exists():
        return None
    videos = [
        path for path in sorted(video_dir.iterdir())
        if path.is_file() and path.suffix.casefold() in VIDEO_EXTENSIONS
    ]
    return videos[0] if videos else None


def video_duration_sec(video: Path) -> float | None:
    if shutil.which("ffprobe") is None:
        return None
    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=nw=1:nk=1",
                str(video),
            ],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=10,
        )
        if result.returncode == 0 and result.stdout.strip():
            return float(result.stdout.strip())
    except Exception:
        return None
    return None


def is_noise_path(file_path: str) -> bool:
    normalized = canonical_path(file_path).casefold()
    if not normalized:
        return True
    basename = normalized.rsplit("/", 1)[-1]
    if basename in NOISE_BASENAMES:
        return True
    if basename.endswith(NOISE_SUFFIXES):
        return True
    return any(segment in normalized for segment in NOISE_SEGMENTS)


def is_candidate_event(event: dict[str, Any]) -> bool:
    file_path = canonical_path(str(event.get("file_path") or ""))
    file_name = str(event.get("file_name") or file_path.rsplit("/", 1)[-1])
    if is_noise_path(file_path):
        return False
    basename = canonical_path(file_name).rsplit("/", 1)[-1]
    ext = Path(basename).suffix.casefold()
    has_sensitive_name = any(keyword.casefold() in basename.casefold() for keyword in SENSITIVE_KEYWORDS)
    has_candidate_ext = ext in DOC_EXTENSIONS or (ext in SOURCE_EXTENSIONS and has_sensitive_name)
    event_type = str(event.get("event_type") or "").casefold()
    extra = event.get("extra") if isinstance(event.get("extra"), dict) else {}
    raw_operation = str(extra.get("raw_operation") or "").casefold()
    has_relevant_event = event_type in RELEVANT_EVENT_TYPES or raw_operation in RELEVANT_EVENT_TYPES
    return has_candidate_ext and (has_relevant_event or has_sensitive_name)


def load_case(root: Path, case_dir: Path) -> CaseInfo | None:
    log_file = case_dir / "logs" / "logs.json"
    video = find_video(case_dir)
    if not log_file.exists() or not video:
        return None
    try:
        logs = json.loads(log_file.read_text(encoding="utf-8"))
        if not isinstance(logs, list):
            return None
    except Exception:
        return None

    stage = stage_name(root)
    rel_parts = list(case_dir.relative_to(root).parts)
    if rel_parts and rel_parts[0].lower() == stage:
        rel_parts = rel_parts[1:]
    first = rel_parts[0] if rel_parts else case_dir.name
    category_match = re.match(r"^(\d+-[^-]+)", first)
    category = category_match.group(1) if category_match else first

    candidate_events = sum(1 for event in logs if isinstance(event, dict) and is_candidate_event(event))
    duration = video_duration_sec(video)
    video_size_mb = video.stat().st_size / 1024 / 1024
    duration_term = duration if duration is not None else 180.0
    score = candidate_events * 20.0 + len(logs) / 100.0 + duration_term / 10.0
    return CaseInfo(
        case=str(case_dir),
        stage=stage,
        category=category,
        log_events=len(logs),
        candidate_events=candidate_events,
        video_sec=duration,
        video_size_mb=round(video_size_mb, 3),
        score=round(score, 3),
    )


def discover_cases(roots: list[Path]) -> list[CaseInfo]:
    cases: list[CaseInfo] = []
    seen: set[str] = set()
    for root in roots:
        root = root.expanduser().resolve()
        for log_file in sorted(root.rglob("logs/logs.json")):
            case_dir = log_file.parent.parent
            key = str(case_dir.resolve())
            if key in seen:
                continue
            seen.add(key)
            info = load_case(root, case_dir)
            if info:
                cases.append(info)
    return cases


def apply_filters(cases: list[CaseInfo], args: argparse.Namespace) -> None:
    for case in cases:
        if case.log_events > args.max_log_events:
            case.exclude_reason = f"log_events>{args.max_log_events}"
        elif case.candidate_events < args.min_candidate_events:
            case.exclude_reason = f"candidate_events<{args.min_candidate_events}"
        elif case.candidate_events > args.max_candidate_events:
            case.exclude_reason = f"candidate_events>{args.max_candidate_events}"
        elif case.video_sec is not None and case.video_sec > args.max_video_sec:
            case.exclude_reason = f"video_sec>{args.max_video_sec}"


def select_balanced(cases: list[CaseInfo], args: argparse.Namespace) -> list[CaseInfo]:
    eligible = [case for case in cases if not case.exclude_reason]
    selected: list[CaseInfo] = []
    per_stage_count: dict[str, int] = {}
    category_seen: set[tuple[str, str]] = set()

    # First pass: pick the best case per stage/category.
    for case in sorted(eligible, key=lambda item: (item.stage, item.category, item.score)):
        if args.max_per_stage and per_stage_count.get(case.stage, 0) >= args.max_per_stage:
            continue
        key = (case.stage, case.category)
        if key in category_seen:
            continue
        case.selected = True
        selected.append(case)
        category_seen.add(key)
        per_stage_count[case.stage] = per_stage_count.get(case.stage, 0) + 1

    # Second pass: fill remaining per-stage quota with the cheapest cases.
    for case in sorted(eligible, key=lambda item: item.score):
        if case.selected:
            continue
        if args.max_total and len(selected) >= args.max_total:
            break
        if args.max_per_stage and per_stage_count.get(case.stage, 0) >= args.max_per_stage:
            continue
        case.selected = True
        selected.append(case)
        per_stage_count[case.stage] = per_stage_count.get(case.stage, 0) + 1

    if args.max_total:
        selected = sorted(selected, key=lambda item: (item.stage, item.score))[:args.max_total]
        selected_set = {case.case for case in selected}
        for case in cases:
            case.selected = case.case in selected_set
    return selected


def make_links(selected: list[CaseInfo], output: Path, force: bool = False) -> None:
    output.mkdir(parents=True, exist_ok=True)
    for case in selected:
        source = Path(case.case)
        dest_dir = output / case.stage
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / f"{slug(case.category)}__{slug(source.name)}"
        if dest.exists() or dest.is_symlink():
            if force:
                if dest.is_dir() and not dest.is_symlink():
                    raise RuntimeError(f"Refusing to remove real directory: {dest}")
                dest.unlink()
            else:
                suffix = 2
                base = dest
                while dest.exists() or dest.is_symlink():
                    dest = base.with_name(f"{base.name}_{suffix}")
                    suffix += 1
        os.symlink(source, dest, target_is_directory=True)
        case.link = str(dest)


def write_outputs(cases: list[CaseInfo], selected: list[CaseInfo], output: Path) -> None:
    manifest = {
        "output": str(output),
        "selected_count": len(selected),
        "scanned_count": len(cases),
        "selected": [asdict(case) for case in selected],
        "excluded_summary": {},
    }
    for case in cases:
        if case.exclude_reason:
            manifest["excluded_summary"][case.exclude_reason] = (
                manifest["excluded_summary"].get(case.exclude_reason, 0) + 1
            )
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output / "cases.txt").write_text(
        "\n".join(case.link or case.case for case in selected) + "\n",
        encoding="utf-8",
    )
    rows = ["stage,category,log_events,candidate_events,video_sec,video_size_mb,score,case,link"]
    for case in selected:
        rows.append(
            f"{case.stage},{case.category},{case.log_events},{case.candidate_events},"
            f"{case.video_sec if case.video_sec is not None else ''},{case.video_size_mb},"
            f"{case.score},{case.case},{case.link}"
        )
    (output / "selected_cases.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a curated log-driven E2E test dataset")
    parser.add_argument("--roots", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-per-stage", type=int, default=30)
    parser.add_argument("--max-total", type=int, default=0)
    parser.add_argument("--max-log-events", type=int, default=8000)
    parser.add_argument("--min-candidate-events", type=int, default=1)
    parser.add_argument("--max-candidate-events", type=int, default=8)
    parser.add_argument("--max-video-sec", type=float, default=600)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    output = args.output.expanduser().resolve()
    cases = discover_cases(args.roots)
    apply_filters(cases, args)
    selected = select_balanced(cases, args)
    make_links(selected, output, force=args.force)
    write_outputs(cases, selected, output)

    print(f"scanned={len(cases)} selected={len(selected)} output={output}")
    by_stage: dict[str, int] = {}
    for case in selected:
        by_stage[case.stage] = by_stage.get(case.stage, 0) + 1
    for stage, count in sorted(by_stage.items()):
        print(f"{stage}: {count}")
    print(f"cases_file={output / 'cases.txt'}")
    print(f"manifest={output / 'manifest.json'}")


if __name__ == "__main__":
    main()
