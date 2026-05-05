#!/usr/bin/env python3
"""
Add video-normalized metrics to an existing log-driven E2E benchmark run.

The original benchmark may only contain samples/min. This script reads each
concurrency_*/jobs.csv, resolves the source video for every case, computes video
duration, and rewrites summary.json and summary_all.json with:

- video_sec_total
- video_sec_per_sec
- sampled_frames_total
- sampled_frames_per_sec
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
from pathlib import Path
from typing import Any


VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi"}


def find_video(case_dir: Path) -> Path | None:
    video_dir = case_dir / "video"
    if not video_dir.exists():
        return None
    videos = [
        path for path in sorted(video_dir.iterdir())
        if path.is_file() and path.suffix.casefold() in VIDEO_EXTENSIONS
    ]
    return videos[0] if videos else None


def video_duration_sec(video_path: Path | None) -> float:
    if not video_path:
        return 0.0

    try:
        import cv2  # type: ignore

        cap = cv2.VideoCapture(str(video_path))
        if cap.isOpened():
            frames = float(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            fps = float(cap.get(cv2.CAP_PROP_FPS) or 0)
            cap.release()
            if frames > 0 and fps > 0:
                return frames / fps
    except Exception:
        pass

    try:
        proc = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(video_path),
            ],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=10,
        )
        if proc.returncode == 0:
            value = float(proc.stdout.strip() or 0)
            if value > 0:
                return value
    except Exception:
        pass

    return 0.0


def load_summary(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_summary(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def enrich_level(level_dir: Path, sample_fps: float) -> dict[str, Any]:
    jobs_path = level_dir / "jobs.csv"
    summary_path = level_dir / "summary.json"
    if not jobs_path.exists() or not summary_path.exists():
        raise FileNotFoundError(f"missing jobs.csv or summary.json under {level_dir}")

    rows = list(csv.DictReader(jobs_path.open(encoding="utf-8")))
    duration_cache: dict[str, tuple[float, float]] = {}
    total_video_sec = 0.0
    total_sampled_frames = 0.0

    for row in rows:
        case = row.get("case") or ""
        if case not in duration_cache:
            video = find_video(Path(case))
            sec = video_duration_sec(video)
            size_mb = video.stat().st_size / 1024 / 1024 if video and video.exists() else 0.0
            duration_cache[case] = (sec, size_mb)
        else:
            sec, size_mb = duration_cache[case]
        row["video_sec"] = f"{sec:.3f}"
        row["video_size_mb"] = f"{size_mb:.3f}"
        row["sampled_frames_est"] = f"{sec * sample_fps:.3f}"
        total_video_sec += sec
        total_sampled_frames += sec * sample_fps

    if rows:
        with jobs_path.open("w", newline="", encoding="utf-8") as handle:
            fieldnames = list(rows[0].keys())
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    summary = load_summary(summary_path)
    elapsed = float(summary.get("elapsed_sec") or 0)
    summary.update({
        "video_sec_total": round(total_video_sec, 3),
        "video_sec_per_sec": total_video_sec / elapsed if elapsed else 0,
        "sampled_frames_total": round(total_sampled_frames, 3),
        "sampled_frames_per_sec": total_sampled_frames / elapsed if elapsed else 0,
        "video_metric_source": "postprocessed_from_case_videos",
    })
    write_summary(summary_path, summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Enrich an E2E bench run with video-normalized metrics")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--sample-fps", type=float, default=1.0)
    args = parser.parse_args()

    run_dir = args.run_dir.expanduser().resolve()
    summaries = []
    for level_dir in sorted(run_dir.glob("concurrency_*")):
        if level_dir.is_dir():
            summaries.append(enrich_level(level_dir, args.sample_fps))

    (run_dir / "summary_all.json").write_text(
        json.dumps(summaries, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("concurrency\tcases\tvideo_sec/sec\tsampled_frames/sec\tsamples/min\tmean_wall\tp95_wall")
    for item in summaries:
        print(
            f"{item.get('concurrency')}\t{item.get('cases')}\t"
            f"{float(item.get('video_sec_per_sec') or 0):.3f}\t"
            f"{float(item.get('sampled_frames_per_sec') or 0):.3f}\t"
            f"{float(item.get('samples_per_min') or 0):.3f}\t"
            f"{float(item.get('mean_wall_sec') or 0):.3f}\t"
            f"{float(item.get('p95_wall_sec') or 0):.3f}"
        )


if __name__ == "__main__":
    main()
