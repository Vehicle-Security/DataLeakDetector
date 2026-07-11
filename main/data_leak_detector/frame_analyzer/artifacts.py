"""Artifact export and precompute cache helpers for direct keyframes."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from .frames import AnalysisWindow, KeyFrame, KeyFrameDuplicate, KeyFrameSelection
from .vlm_client import build_vlm_frame_grids, prepare_vlm_frame_images


def export_vision_artifacts(
    *,
    artifact_dir: str | Path | None,
    keyframes: list[Any],
    raw_all_keyframes: list[Any] | None = None,
    duplicate_keyframes: list[KeyFrameDuplicate] | None = None,
) -> dict[str, Any]:
    if artifact_dir is None:
        return {}

    root = Path(artifact_dir)
    raw_all_dir = root / "keyframes_raw_all"
    raw_dir = root / "keyframes_raw"
    for directory in (raw_all_dir, raw_dir):
        if directory.exists():
            shutil.rmtree(directory)
        directory.mkdir(parents=True, exist_ok=True)

    raw_all_files = _copy_frame_images(raw_all_keyframes if raw_all_keyframes is not None else keyframes, raw_all_dir)
    raw_files = _copy_frame_images(keyframes, raw_dir)
    duplicate_file = root / "keyframe_duplicates.json"
    duplicate_file.write_text(
        json.dumps([_keyframe_duplicate_to_dict(item) for item in duplicate_keyframes or []], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    manifest_file = root / "artifact_manifest.json"
    manifest_file.write_text(
        json.dumps({"keyframes_raw_all_files": raw_all_files, "keyframes_raw_files": raw_files}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return {
        "root_dir": str(root),
        "keyframes_raw_all_dir": str(raw_all_dir),
        "keyframes_raw_dir": str(raw_dir),
        "keyframes_raw_files": raw_files,
        "keyframe_duplicates_file": str(duplicate_file),
        "artifact_manifest_file": str(manifest_file),
        "counts": {
            "keyframes_raw_all_files": len(raw_all_files),
            "keyframes_raw_files": len(raw_files),
        },
    }


def write_vision_precompute(
    manifest: dict[str, Any],
    *,
    windows: list[AnalysisWindow],
    selection: KeyFrameSelection,
) -> None:
    root_text = str(manifest.get("root_dir") or "")
    if not root_text:
        return

    root = Path(root_text)
    copied = list(manifest.get("keyframes_raw_files") or [])
    image_by_id = {
        frame.frame_id: copied[index]
        for index, frame in enumerate(selection.keyframes)
        if index < len(copied)
    }
    payload = {
        "schema_version": 1,
        "windows": [_analysis_window_to_dict(item) for item in windows],
        "keyframes": [_keyframe_to_dict(item, image_path=image_by_id.get(item.frame_id, item.image_path)) for item in selection.keyframes],
        "raw_keyframe_count": len(selection.raw_keyframes),
        "duplicate_count": len(selection.duplicates),
        "warnings": list(selection.warnings),
    }
    path = root / "vision_precompute.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    manifest["vision_precompute_file"] = str(path)
    update_artifact_manifest_file(manifest, {"vision_precompute_file": str(path)})


def load_vision_precompute(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError(f"unsupported_vision_precompute: {path}")
    keyframes = [_keyframe_from_dict(item) for item in payload.get("keyframes", []) if isinstance(item, dict)]
    return {
        "windows": [_analysis_window_from_dict(item) for item in payload.get("windows", []) if isinstance(item, dict)],
        "selection": KeyFrameSelection(
            keyframes=keyframes,
            raw_keyframes=keyframes,
            duplicates=[],
            warnings=[str(item) for item in payload.get("warnings", [])],
        ),
    }


def prepare_vlm_request_frames(
    frames: list[Any],
    *,
    max_image_side: int,
    grid_size: int,
    artifact_dir: str | Path | None,
    manifest: dict[str, Any],
) -> list[Any]:
    input_dir = Path(artifact_dir) / "keyframes_vlm_input" if artifact_dir is not None and max_image_side > 0 else None
    if input_dir is not None:
        _reset_dir(input_dir)

    prepared_frames = prepare_vlm_frame_images(frames, max_image_side=max_image_side, output_dir=input_dir)
    if input_dir is not None:
        input_files = [item.frame.image_path for item in prepared_frames if Path(item.frame.image_path).parent == input_dir]
        if input_files:
            manifest["keyframes_vlm_input_dir"] = str(input_dir)
            manifest["keyframes_vlm_input_files"] = input_files
            _set_manifest_count(manifest, "keyframes_vlm_input_files", len(input_files))
            update_artifact_manifest_file(manifest, {"keyframes_vlm_input_files": input_files})

    if grid_size <= 1:
        return prepared_frames

    grid_dir = Path(artifact_dir) / "keyframes_vlm_grid" if artifact_dir is not None else None
    if grid_dir is not None:
        _reset_dir(grid_dir)
    grid_frames = build_vlm_frame_grids(prepared_frames, grid_size=grid_size, output_dir=grid_dir)
    if grid_dir is not None:
        grid_files = [item.frame.image_path for item in grid_frames]
        manifest["keyframes_vlm_grid_dir"] = str(grid_dir)
        manifest["keyframes_vlm_grid_files"] = grid_files
        _set_manifest_count(manifest, "keyframes_vlm_grid_files", len(grid_files))
        update_artifact_manifest_file(manifest, {"keyframes_vlm_grid_files": grid_files})
    return grid_frames


def write_json_artifact(
    artifact_dir: str | Path | None,
    filename: str,
    payload: dict[str, Any],
    manifest: dict[str, Any],
    manifest_key: str,
) -> None:
    if artifact_dir is None:
        return
    path = Path(artifact_dir) / filename
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    manifest[manifest_key] = str(path)


def update_artifact_manifest_file(manifest: dict[str, Any], updates: dict[str, Any]) -> None:
    manifest_file = str(manifest.get("artifact_manifest_file") or "")
    if not manifest_file:
        return
    path = Path(manifest_file)
    if path.exists():
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            payload = {}
    else:
        payload = {}
    if not isinstance(payload, dict):
        payload = {}
    payload.update(updates)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _reset_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _set_manifest_count(manifest: dict[str, Any], key: str, value: int) -> None:
    counts = manifest.setdefault("counts", {})
    if isinstance(counts, dict):
        counts[key] = value


def _copy_frame_images(frames: list[Any], target_dir: Path) -> list[str]:
    copied: list[str] = []
    for index, frame in enumerate(frames):
        source = Path(str(getattr(frame, "image_path", "")))
        if not source.exists():
            continue
        timestamp = int(getattr(frame, "timestamp_ms", 0))
        reason = str(getattr(frame, "reason", "frame")).replace(":", "-").replace("/", "-").replace("\\", "-")
        target = target_dir / f"{index:03d}_{timestamp}ms_{reason}{source.suffix or '.jpg'}"
        shutil.copy2(source, target)
        copied.append(str(target))
    return copied


def _keyframe_to_dict(frame: KeyFrame, *, image_path: str | None = None) -> dict[str, Any]:
    return {
        "frame_id": frame.frame_id,
        "timestamp_ms": frame.timestamp_ms,
        "image_path": image_path or frame.image_path,
        "score": frame.score,
        "reason": frame.reason,
        "window_id": frame.window_id,
    }


def _keyframe_from_dict(item: dict[str, Any]) -> KeyFrame:
    return KeyFrame(
        frame_id=str(item.get("frame_id") or ""),
        timestamp_ms=int(item.get("timestamp_ms") or 0),
        image_path=str(item.get("image_path") or ""),
        score=float(item.get("score") or 0.0),
        reason=str(item.get("reason") or ""),
        window_id=str(item.get("window_id") or ""),
    )


def _analysis_window_to_dict(window: AnalysisWindow) -> dict[str, Any]:
    return {
        "start_ms": window.start_ms,
        "end_ms": window.end_ms,
        "reason": window.reason,
        "priority": window.priority,
        "step_ms": window.step_ms,
        "max_keyframes": window.max_keyframes,
        "diff_threshold": window.diff_threshold,
        "anchor_ms": list(window.anchor_ms),
        "active_apps": list(window.active_apps),
    }


def _analysis_window_from_dict(item: dict[str, Any]) -> AnalysisWindow:
    return AnalysisWindow(
        start_ms=int(item.get("start_ms") or 0),
        end_ms=int(item.get("end_ms") or 0),
        reason=str(item.get("reason") or ""),
        priority=str(item.get("priority") or "medium"),
        step_ms=int(item.get("step_ms") or 1_000),
        max_keyframes=int(item.get("max_keyframes") or 0),
        diff_threshold=float(item.get("diff_threshold") or 0.0),
        anchor_ms=tuple(int(value) for value in item.get("anchor_ms") or []),
        active_apps=tuple(str(value) for value in item.get("active_apps") or []),
    )


def _keyframe_duplicate_to_dict(duplicate: KeyFrameDuplicate) -> dict[str, Any]:
    return {
        "frame_id": duplicate.frame.frame_id,
        "timestamp_ms": duplicate.frame.timestamp_ms,
        "image_path": duplicate.frame.image_path,
        "reason": duplicate.frame.reason,
        "window_id": duplicate.frame.window_id,
        "kept_frame_id": duplicate.kept_frame_id,
        "duplicate_reason": duplicate.reason,
        "delta": duplicate.delta,
        "hash_distance": duplicate.hash_distance,
    }
