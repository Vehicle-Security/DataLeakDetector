"""Command-line entry point for the canonical DataLeakDetector pipeline."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
import sys
import threading
import time
from pathlib import Path

from data_leak_detector import run_data_case, run_pipeline
from data_leak_detector.datasets import data_case_id, discover_data_case_directories
from data_leak_detector.frame_analyzer.vlm_dispatch import vlm_dispatcher_snapshots


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Run DataLeakDetector end-to-end.")
    parser.add_argument("--log", "-l", default="", help="Path to a JSON/JSONL monitor log.")
    parser.add_argument("--case", "-c", default="", help="Path to a spec/data sample case directory.")
    parser.add_argument("--case-root", default="", help="Recursively run every case directory below this root with shared VLM plan quotas.")
    parser.add_argument("--case-workers", type=int, default=1, help="Concurrent cases for --case-root; VLM plan limits remain global.")
    parser.add_argument("--release", action="store_true", help="For --case-root, write one consolidated release report instead of per-case debug artifacts.")
    parser.add_argument("--release-precompute-only", action="store_true", help="Build or reuse Release precompute caches, then exit before VLM.")
    parser.add_argument(
        "--release-precompute-neo4j-log-miner",
        action="store_true",
        help="Use Neo4j log mining only while building the reusable Release vision precompute cache.",
    )
    parser.add_argument("--video", "-v", default="", help="Optional screen recording path for frame analysis.")
    parser.add_argument("--groundtruth", default="", help="Optional groundtruth.json path for verdict evaluation.")
    parser.add_argument("--output-dir", "-o", default="", help="Optional directory for the JSON report.")
    parser.add_argument("--sensitive-file", action="append", default=[], help="Sensitive file path. Can be repeated.")
    parser.add_argument("--observations", default="", help="Optional precomputed frame observation JSON.")
    parser.add_argument("--vision", action="store_true", help="Enable direct-keyframe VLM frame analysis.")
    parser.add_argument("--max-vlm-frames", type=int, default=None, help="Maximum keyframes sent to VLM; 0 disables VLM frames, negative means no cap.")
    parser.add_argument("--vlm-dry-run", action="store_true", help="Write VLM request artifacts without calling the model API.")
    parser.add_argument("--vlm-grid-size", type=int, default=0, help="Pack selected VLM frames into NxN grid images before calling the model.")
    parser.add_argument("--vlm-workers", type=int, default=0, help="VLM workers; in fast dispatch mode this is the per-Key concurrency.")
    parser.add_argument("--vlm-fast-dispatch", action="store_true", help="Use every configured VLM API Key concurrently.")
    parser.add_argument("--vlm-max-image-side", type=int, default=-1, help="Resize VLM input images to this max side; 0 keeps originals.")
    parser.add_argument("--no-non-vlm", action="store_true", help="Disable deterministic log evidence in correlation; useful for VLM-only evaluation.")
    parser.add_argument("--neo4j-log-miner", action="store_true", help="Use Neo4j to mine analysis windows before frame extraction.")
    parser.add_argument("--no-neo4j-log-miner", action="store_true", help="Use in-memory log mining even when the environment enables Neo4j.")
    parser.add_argument("--no-reuse-neo4j-import", action="store_true", help="Reimport logs even when the case fingerprint already exists in Neo4j.")
    parser.add_argument(
        "--inherit-ancestor-groundtruth",
        action="store_true",
        help="For child session cases without groundtruth.json, use the nearest ancestor groundtruth for evaluation and sensitive sources.",
    )
    args = parser.parse_args(argv)
    if args.vlm_dry_run:
        os.environ["DLD_VLM_DRY_RUN"] = "1"
    if args.vlm_grid_size:
        os.environ["DLD_VLM_GRID_SIZE"] = str(max(1, args.vlm_grid_size))
    if args.vlm_workers:
        os.environ["DLD_VLM_WORKERS"] = str(max(1, args.vlm_workers))
    if args.vlm_fast_dispatch:
        os.environ["DLD_VLM_FAST_DISPATCH"] = "1"
    if args.vlm_max_image_side >= 0:
        os.environ["DLD_VLM_MAX_IMAGE_SIDE"] = str(max(0, args.vlm_max_image_side))

    common_args = {
        "output_dir": args.output_dir or None,
        "sensitive_files": args.sensitive_file,
        "observations_file": args.observations or None,
        "neo4j_log_miner": False if args.no_neo4j_log_miner else (True if args.neo4j_log_miner else None),
        "reuse_neo4j_import": False if args.no_reuse_neo4j_import else None,
        "vision_enabled": True if args.vision else None,
        "max_vlm_frames": args.max_vlm_frames,
        "non_vlm_enabled": False if args.no_non_vlm else None,
        "vision_debug_artifacts": True,
        "inherit_ancestor_groundtruth": args.inherit_ancestor_groundtruth,
    }
    if args.release:
        common_args = _release_direct_defaults(common_args, args)
    if args.release and not args.case_root:
        parser.error("--release requires --case-root")
    if args.release_precompute_only and not args.release:
        parser.error("--release-precompute-only requires --release")
    if args.neo4j_log_miner and args.no_neo4j_log_miner:
        parser.error("--neo4j-log-miner and --no-neo4j-log-miner cannot be used together")
    if args.case and args.case_root:
        parser.error("--case and --case-root cannot be used together")
    if args.case_root:
        if args.release_precompute_only:
            root = Path(args.case_root)
            output_root = Path(args.output_dir) if args.output_dir else Path("artifacts") / f"{root.name}_release_{time.strftime('%Y%m%d_%H%M%S')}"
            caches = _build_release_vision_precompute(
                args.case_root,
                common_args=common_args,
                cache_root=output_root / "vision_precompute",
                workers=max(1, args.case_workers),
                neo4j_log_miner=True if args.release_precompute_neo4j_log_miner else None,
            )
            report = {"batch": {"mode": "release_precompute", "case_count": len(caches), "cache_root": str(output_root / "vision_precompute")}}
        elif args.release:
            report = _run_release(
                args.case_root,
                common_args=common_args,
                output_dir=args.output_dir or None,
                workers=max(1, args.case_workers),
                grid_size=max(1, args.vlm_grid_size or int(os.getenv("DLD_VLM_GRID_SIZE", "2"))),
                precompute_neo4j_log_miner=True if args.release_precompute_neo4j_log_miner else None,
            )
        else:
            report = _run_case_root(
                args.case_root,
                common_args=common_args,
                output_dir=args.output_dir or None,
                workers=max(1, args.case_workers),
                release=args.release,
            )
    elif args.case:
        report = run_data_case(args.case, **common_args)
    else:
        if not args.log:
            parser.error("either --log or --case is required")
        report = run_pipeline(log_file=args.log, video_file=args.video, groundtruth_file=args.groundtruth or None, **common_args)

    print(json.dumps(_build_cli_summary(report), ensure_ascii=False, indent=2))
    return 0


def _release_direct_defaults(common_args: dict, args: argparse.Namespace) -> dict:
    release_args = dict(common_args)
    release_args["vision_enabled"] = True
    release_args["max_vlm_frames"] = -1 if args.max_vlm_frames is None else args.max_vlm_frames
    release_args["non_vlm_enabled"] = False
    release_args["vision_debug_artifacts"] = False
    release_args["inherit_ancestor_groundtruth"] = True
    if not args.neo4j_log_miner:
        release_args["neo4j_log_miner"] = False
    return release_args


def _build_cli_summary(report: dict) -> dict:
    """Keep command output readable; full evidence lives in report/detail files."""

    if "batch" in report:
        return {"batch": dict(report["batch"])}
    return {
        "report_id": report.get("report_id", ""),
        "conclusion": report.get("conclusion", ""),
        "report_file": report.get("report_file", ""),
        "detail_files": report.get("detail_files", {}),
        "summary": report.get("summary", {}),
        "log_miner": report.get("log_miner", {}),
        "vision": report.get("frame_analyzer", {}).get("statistics", {}).get("vision", {}),
        "verdict": report.get("verdict", {}),
        "graph": report.get("graph", {}),
    }


def _run_case_root(
    case_root: str,
    *,
    common_args: dict,
    output_dir: str | None,
    workers: int,
    release: bool = False,
    write_release_report: bool = True,
    case_arg_overrides: dict[str, dict] | None = None,
    progress_file: Path | None = None,
    progress_context: dict | None = None,
) -> dict:
    root = Path(case_root)
    case_dirs = discover_data_case_directories(root)
    cases = [(data_case_id(case, root), case) for case in case_dirs]
    if not cases:
        raise ValueError(f"no case directories found under {root}")

    output_root = Path(output_dir) if output_dir else None
    if release and output_root is None:
        output_root = Path("artifacts") / f"{root.name}_release_{time.strftime('%Y%m%d_%H%M%S')}"
    started = time.perf_counter()
    if release:
        assert output_root is not None
        output_root.mkdir(parents=True, exist_ok=True)
        progress_file = progress_file or output_root / "release_progress.json"
    progress_lock = threading.Lock()
    running: set[str] = set()
    finished_uncollected: set[str] = set()
    recent_cases: list[dict] = []
    last_case: dict[str, object] | None = None

    def persist_progress(state: str) -> None:
        if progress_file is None:
            return
        payload = {
            "mode": "release",
            "state": state,
            "case_root": str(root),
            "case_count": len(cases),
            "case_workers": min(workers, len(cases)),
            "completed_cases": len(completed),
            "failed_cases": len(errors),
            "running_cases": sorted(running),
            "queued_cases": max(
                0,
                len(cases) - len(completed) - len(errors) - len(running) - len(finished_uncollected),
            ),
            "last_case": last_case,
            "recent_cases": recent_cases[-20:],
            "elapsed_seconds": round(time.perf_counter() - started, 3),
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "vlm_dispatchers": vlm_dispatcher_snapshots(),
        }
        if progress_context:
            payload.update(progress_context)
        try:
            _write_json_atomic(progress_file, payload)
        except OSError as exc:
            # Progress visibility must never fail a case when an editor temporarily locks the file.
            print(f"release progress write warning: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)

    def mark_started(case_name: str) -> None:
        with progress_lock:
            running.add(case_name)
            persist_progress("running")
            print(
                f"  started case={case_name} "
                f"running={len(running)}/{min(workers, len(cases))}",
                flush=True,
            )

    def mark_finished(case_name: str) -> None:
        with progress_lock:
            running.discard(case_name)
            finished_uncollected.add(case_name)
            persist_progress("running")

    def run_one(case_id: str, case: Path) -> tuple[str, dict, float]:
        case_args = dict(common_args)
        overrides = (case_arg_overrides or {}).get(case_id) or (case_arg_overrides or {}).get(case.name) or {}
        case_args.update(overrides)
        if output_root is not None and not release:
            case_args["output_dir"] = str(output_root / Path(case_id))
        elif release:
            case_args["output_dir"] = None
        case_started = time.perf_counter()
        mark_started(case_id)
        try:
            return case_id, run_data_case(case, case_root=root, **case_args), time.perf_counter() - case_started
        finally:
            mark_finished(case_id)

    completed: list[tuple[str, dict, float]] = []
    errors: list[dict[str, str]] = []
    heartbeat_stop = threading.Event()
    heartbeat_thread: threading.Thread | None = None
    if release:
        with progress_lock:
            persist_progress("starting")
        print(
            f"release cases=0/{len(cases)} workers={min(workers, len(cases))} progress={progress_file}",
            flush=True,
        )
        def heartbeat() -> None:
            while not heartbeat_stop.wait(5):
                with progress_lock:
                    persist_progress("running")

        heartbeat_thread = threading.Thread(target=heartbeat, name="dld_release_progress", daemon=True)
        heartbeat_thread.start()
    try:
        with ThreadPoolExecutor(max_workers=min(workers, len(cases))) as executor:
            futures = {executor.submit(run_one, case_id, case): case_id for case_id, case in cases}
            for future in as_completed(futures):
                case_name = futures[future]
                try:
                    name, result, seconds = future.result()
                    with progress_lock:
                        finished_uncollected.discard(name)
                        completed.append((name, result, seconds))
                        evaluation = _release_case_evaluation(result)
                        last_case = {
                            "case": name,
                            "state": "completed",
                            "seconds": round(seconds, 3),
                            "conclusion": result.get("conclusion", ""),
                            **evaluation,
                        }
                        recent_cases.append(last_case)
                        persist_progress("running")
                        correct = evaluation["detector_correct"]
                        correct_text = str(correct).lower() if correct is not None else "n/a"
                        print(
                            f"  [{len(completed) + len(errors)}/{len(cases)}] completed "
                            f"case={name} seconds={seconds:.1f} "
                            f"detector={evaluation['detector_conclusion']} "
                            f"expected={evaluation['expected_conclusion'] or 'n/a'} correct={correct_text} "
                            f"running={len(running)} "
                            f"queued={max(0, len(cases) - len(completed) - len(errors) - len(running) - len(finished_uncollected))}",
                            flush=True,
                        )
                except Exception as exc:
                    with progress_lock:
                        running.discard(case_name)
                        finished_uncollected.discard(case_name)
                        error = {"case": case_name, "error": f"{type(exc).__name__}: {exc}"}
                        errors.append(error)
                        last_case = {"case": case_name, "state": "failed", "error": error["error"]}
                        recent_cases.append(last_case)
                        persist_progress("running")
                        print(
                            f"  [{len(completed) + len(errors)}/{len(cases)}] failed "
                            f"case={case_name} error={error['error']}",
                            flush=True,
                        )
    finally:
        heartbeat_stop.set()
        if heartbeat_thread is not None:
            heartbeat_thread.join(timeout=1)

    completed.sort(key=lambda item: item[0])
    batch = {
        "mode": "release" if release else "debug",
        "report_file": "",
        "case_root": str(root),
        "case_count": len(cases),
        "case_workers": min(workers, len(cases)),
        "completed_cases": len(completed),
        "failed_cases": len(errors),
        "timing_seconds": round(time.perf_counter() - started, 3),
        "cases": [
            {
                "case": name,
                "seconds": round(seconds, 3),
                "conclusion": result.get("conclusion", ""),
                "report_file": result.get("report_file", ""),
            }
            for name, result, seconds in completed
        ],
        "errors": errors,
    }
    if not release:
        return {"batch": batch}

    assert output_root is not None
    output_root.mkdir(parents=True, exist_ok=True)
    release_report = {
        "batch": batch,
        "summary": _release_summary(completed),
        "cases": [_release_case_report(name, report, seconds) for name, report, seconds in completed],
    }
    if write_release_report:
        report_file = output_root / "release_report.json"
        release_report["batch"]["report_file"] = str(report_file)
        report_file.write_text(json.dumps(release_report, ensure_ascii=False, indent=2), encoding="utf-8")
    if release:
        with progress_lock:
            persist_progress("completed")
    return {"batch": release_report["batch"], "release_report": release_report}


def _run_release(
    case_root: str,
    *,
    common_args: dict,
    output_dir: str | None,
    workers: int,
    grid_size: int,
    precompute_neo4j_log_miner: bool | None = None,
) -> dict:
    root = Path(case_root)
    output_root = Path(output_dir) if output_dir else Path("artifacts") / f"{root.name}_release_{time.strftime('%Y%m%d_%H%M%S')}"
    previous_grid = os.environ.get("DLD_VLM_GRID_SIZE")
    started = time.perf_counter()
    output_root.mkdir(parents=True, exist_ok=True)
    progress_file = output_root / "release_progress.json"
    comparison_file = output_root / "release_comparison.json"
    vision_precompute = _build_release_vision_precompute(
        case_root,
        common_args=common_args,
        cache_root=output_root / "vision_precompute",
        workers=workers,
        neo4j_log_miner=precompute_neo4j_log_miner,
    )
    os.environ["DLD_VLM_GRID_SIZE"] = str(grid_size)
    try:
        print(f"release direct_keyframes grid={grid_size}", flush=True)
        case_overrides = {
            name: {
                "precomputed_baseline_file": path,
                "non_vlm_enabled": False,
                "vision_debug_artifacts": False,
            }
            for name, path in vision_precompute.items()
        }
        result = _run_case_root(
            case_root,
            common_args=common_args,
            output_dir=str(output_root),
            workers=workers,
            release=True,
            write_release_report=True,
            progress_file=progress_file,
            progress_context={"mode": "release", "vlm_grid_size": grid_size},
            case_arg_overrides=case_overrides,
        )
    finally:
        _restore_env("DLD_VLM_GRID_SIZE", previous_grid)

    release_report = result["release_report"]
    release_report["batch"]["mode"] = "release"
    release_report["batch"]["vlm_frame_source"] = "direct_keyframes"
    release_report["batch"]["vlm_grid_size"] = grid_size
    release_report["batch"]["timing_seconds"] = round(time.perf_counter() - started, 3)
    report_file = output_root / "release_report.json"
    release_report["batch"]["report_file"] = str(report_file)
    report_file.write_text(json.dumps(release_report, ensure_ascii=False, indent=2), encoding="utf-8")
    comparison = _build_release_comparison(root=root, workers=workers, release_report=release_report, grid_size=grid_size)
    _write_optional_json(comparison_file, comparison)
    return {
        "batch": {
            "mode": "release",
            "report_file": str(report_file),
            "comparison_file": str(comparison_file),
            "case_root": str(root),
            "case_workers": workers,
            "vlm_frame_source": "direct_keyframes",
            "vlm_grid_size": grid_size,
            "timing_seconds": round(time.perf_counter() - started, 3),
        },
        "release_report": release_report,
    }

def _build_release_vision_precompute(
    case_root: str,
    *,
    common_args: dict,
    cache_root: Path,
    workers: int,
    neo4j_log_miner: bool | None,
) -> dict[str, str]:
    root = Path(case_root)
    case_dirs = discover_data_case_directories(root)
    cases = [(data_case_id(case, root), case) for case in case_dirs]
    cache_root.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    progress_file = cache_root.parent / "release_precompute_progress.json"
    progress_lock = threading.Lock()
    running: set[str] = set()
    errors: list[dict] = []
    recent_cases: list[dict] = []
    completed: dict[str, str] = {}

    def persist_progress(state: str) -> None:
        payload = {
            "mode": "release_precompute",
            "state": state,
            "case_root": str(root),
            "cache_root": str(cache_root),
            "case_count": len(cases),
            "case_workers": min(workers, len(cases)),
            "completed_cases": len(completed),
            "failed_cases": len(errors),
            "running_cases": sorted(running),
            "queued_cases": max(0, len(cases) - len(completed) - len(errors) - len(running)),
            "recent_cases": recent_cases[-20:],
            "elapsed_seconds": round(time.perf_counter() - started, 3),
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        }
        _write_json_atomic(progress_file, payload)

    def run_one(case_id: str, case: Path) -> tuple[str, str]:
        with progress_lock:
            running.add(case_id)
            persist_progress("running")
        print(f"  precompute started case={case_id}", flush=True)
        case_root_dir = cache_root / Path(case_id)
        try:
            existing = sorted(case_root_dir.rglob("pipeline_baseline.json")) if case_root_dir.exists() else []
            if existing and _precompute_baseline_matches_mode(existing[-1], "direct_keyframes_only"):
                return case_id, str(existing[-1])
            args = dict(common_args)
            args.update(
                {
                    "output_dir": str(case_root_dir),
                    "vision_enabled": True,
                    "max_vlm_frames": 0,
                    "non_vlm_enabled": False,
                    "report_case_name": case.name,
                }
            )
            if neo4j_log_miner is not None:
                args["neo4j_log_miner"] = neo4j_log_miner
            report = run_data_case(case, case_root=root, **args)
            vision = dict(report.get("frame_analyzer", {}).get("statistics", {}).get("vision", {}))
            cache_file = str(vision.get("artifacts", {}).get("vision_precompute_file") or "")
            if not cache_file or not Path(cache_file).exists():
                raise RuntimeError(f"vision_precompute_missing: {case_id}")
            baseline_file = Path(cache_file).with_name("pipeline_baseline.json")
            baseline_file.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "precompute_mode": "direct_keyframes_only",
                        "vision_precompute_file": cache_file,
                        "records": report.get("event_correlator", {}).get("raw_log_events", []),
                        "log_observations": [
                            item for item in report.get("frame_analyzer", {}).get("observations", [])
                            if item.get("source") == "log_anchored"
                        ],
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            return case_id, str(baseline_file)
        finally:
            with progress_lock:
                running.discard(case_id)
                persist_progress("running")

    print(f"release precompute cases=0/{len(cases)} workers={min(workers, len(cases))}", flush=True)
    persist_progress("running")
    with ThreadPoolExecutor(max_workers=min(workers, len(cases))) as executor:
        futures = {executor.submit(run_one, case_id, case): case_id for case_id, case in cases}
        for future in as_completed(futures):
            future_case_id = futures[future]
            try:
                name, cache_file = future.result()
            except Exception as exc:
                with progress_lock:
                    errors.append({"case_id": future_case_id, "error": f"{type(exc).__name__}: {exc}"})
                    recent_cases.append({"case_id": future_case_id, "status": "error"})
                    persist_progress("failed")
                raise
            with progress_lock:
                completed[name] = cache_file
                recent_cases.append({"case_id": name, "status": "completed", "baseline_file": cache_file})
                persist_progress("running")
            print(f"  precompute [{len(completed)}/{len(cases)}] case={name}", flush=True)
    persist_progress("completed")
    return completed


def _restore_env(name: str, previous: str | None) -> None:
    if previous is None:
        os.environ.pop(name, None)
    else:
        os.environ[name] = previous


def _precompute_baseline_matches_mode(path: Path, mode: str) -> bool:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return isinstance(payload, dict) and payload.get("schema_version") == 1 and payload.get("precompute_mode") == mode


def _write_json_atomic(path: Path, payload: dict, *, attempts: int = 3) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_file = path.with_suffix(".tmp")
    content = json.dumps(payload, ensure_ascii=False, indent=2)
    for attempt in range(attempts):
        temporary_file.write_text(content, encoding="utf-8")
        try:
            os.replace(temporary_file, path)
            return
        except PermissionError:
            if attempt + 1 == attempts:
                raise
            time.sleep(0.05 * (attempt + 1))


def _write_optional_json(path: Path, payload: dict) -> None:
    try:
        _write_json_atomic(path, payload)
    except OSError as exc:
        print(f"optional report write warning: {path.name}: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)


def _build_release_comparison(
    *,
    root: Path,
    workers: int,
    release_report: dict,
    grid_size: int,
) -> dict:
    batch = dict(release_report.get("batch", {}))
    summary = dict(release_report.get("summary", {}))
    cases = list(release_report.get("cases", []))
    evaluations = [dict(case.get("evaluation", {})) for case in cases]
    correct = sum(evaluation.get("detector_correct") is True for evaluation in evaluations)
    incorrect = sum(evaluation.get("detector_correct") is False for evaluation in evaluations)
    missing_groundtruth = sum(evaluation.get("unscored_reason") == "missing_groundtruth" for evaluation in evaluations)
    unsupported_groundtruth = sum(
        str(evaluation.get("unscored_reason") or "").startswith("unsupported_groundtruth:")
        for evaluation in evaluations
    )
    scored = correct + incorrect
    case_rows: list[dict] = []
    for case in cases:
        evaluation = dict(case.get("evaluation", {}))
        case_rows.append(
            {
                "case": case.get("case", ""),
                "case_id": case.get("case_id", case.get("case", "")),
                "case_name": case.get("case_name", ""),
                "case_relative_path": case.get("case_relative_path", ""),
                "vlm_frame_source": "direct_keyframes",
                "vlm_grid_size": grid_size,
                "seconds": case.get("seconds", 0),
                "detector_conclusion": evaluation.get("detector_conclusion", case.get("conclusion", "")),
                "expected_conclusion": evaluation.get("expected_conclusion", ""),
                "score_status": evaluation.get("score_status", ""),
                "unscored_reason": evaluation.get("unscored_reason", ""),
                "groundtruth_available": evaluation.get("groundtruth_available"),
                "groundtruth_status": evaluation.get("groundtruth_status", ""),
                "nearest_ancestor_groundtruth_file": evaluation.get("nearest_ancestor_groundtruth_file", ""),
                "detector_correct": evaluation.get("detector_correct"),
                "errors": case.get("errors", []),
            }
        )
    return {
        "mode": "release_comparison",
        "case_root": str(root),
        "case_workers": workers,
        "vlm_frame_source": "direct_keyframes",
        "vlm_grid_size": grid_size,
        "report_file": batch.get("report_file", ""),
        "case_count": batch.get("case_count", len(cases)),
        "completed_cases": batch.get("completed_cases", len(cases)),
        "failed_cases": batch.get("failed_cases", 0),
        "case_seconds": summary.get("case_seconds", 0),
        "vlm_seconds": summary.get("vlm_seconds", 0),
        "vlm_frames": summary.get("vlm_frames", 0),
        "correct_cases": correct,
        "incorrect_cases": incorrect,
        "unscored_cases": len(cases) - scored,
        "unscored_missing_groundtruth_cases": missing_groundtruth,
        "unscored_unsupported_groundtruth_cases": unsupported_groundtruth,
        "accuracy": round(correct / scored, 6) if scored else None,
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "case_rows": case_rows,
    }

def _release_case_report(case_name: str, report: dict, seconds: float) -> dict:
    vision = dict(report.get("frame_analyzer", {}).get("statistics", {}).get("vision", {}))
    event_correlator = dict(report.get("event_correlator", {}))
    input_metadata = dict(report.get("input", {}))
    event_correlator.pop("raw_log_events", None)
    return {
        "case": case_name,
        "case_id": input_metadata.get("case_id", case_name),
        "case_name": input_metadata.get("case_name", ""),
        "case_relative_path": input_metadata.get("case_relative_path", case_name),
        "seconds": round(seconds, 3),
        "report_id": report.get("report_id", ""),
        "conclusion": report.get("conclusion", ""),
        "summary": report.get("summary", {}),
        "vision": vision,
        "verdict": report.get("verdict", {}),
        "evaluation": _release_case_evaluation(report),
        "detection_core": report.get("detection_core", {}),
        "leak_reasoner": report.get("leak_reasoner", {}),
        "event_correlator": event_correlator,
        "errors": report.get("frame_analyzer", {}).get("errors", []),
    }


def _release_case_evaluation(report: dict) -> dict[str, object]:
    verdict = dict(report.get("verdict", {}))
    groundtruth = dict(report.get("groundtruth", {}))
    input_metadata = dict(report.get("input", {}))
    detector = str(verdict.get("detector_conclusion") or report.get("conclusion") or "")
    expected = str(groundtruth.get("conclusion") or verdict.get("groundtruth_conclusion") or "")
    available = bool(groundtruth.get("available"))
    is_scorable_expected = _is_scorable_conclusion(expected)
    if not available:
        score_status = "unscored"
        unscored_reason = "missing_groundtruth"
    elif is_scorable_expected:
        score_status = "scored"
        unscored_reason = ""
    else:
        score_status = "unscored"
        unscored_reason = f"unsupported_groundtruth:{expected or 'unknown'}"
    return {
        "detector_conclusion": detector,
        "expected_conclusion": expected,
        "groundtruth_available": available,
        "groundtruth_status": input_metadata.get("groundtruth_status", ""),
        "nearest_ancestor_groundtruth_file": input_metadata.get("nearest_ancestor_groundtruth_file", ""),
        "score_status": score_status,
        "unscored_reason": unscored_reason,
        "detector_correct": detector == expected if available and is_scorable_expected else None,
    }


def _release_summary(completed: list[tuple[str, dict, float]]) -> dict:
    reports = [report for _, report, _ in completed]
    visions = [dict(report.get("frame_analyzer", {}).get("statistics", {}).get("vision", {})) for report in reports]
    evaluations = [_release_case_evaluation(report) for report in reports]
    correct = sum(item.get("detector_correct") is True for item in evaluations)
    incorrect = sum(item.get("detector_correct") is False for item in evaluations)
    scored = correct + incorrect
    return {
        "case_count": len(reports),
        "scored_cases": scored,
        "correct_cases": correct,
        "incorrect_cases": incorrect,
        "unscored_cases": len(reports) - scored,
        "unscored_missing_groundtruth_cases": sum(
            item.get("unscored_reason") == "missing_groundtruth" for item in evaluations
        ),
        "data_leak_risk_detected": sum(report.get("conclusion") == "data_leak_risk_detected" for report in reports),
        "logs": sum(int(report.get("summary", {}).get("logs", 0)) for report in reports),
        "frame_observations": sum(int(report.get("summary", {}).get("frame_observations", 0)) for report in reports),
        "vlm_frames": sum(int(vision.get("vlm_frames", 0)) for vision in visions),
        "vlm_events": sum(int(vision.get("vlm_events", 0)) for vision in visions),
        "vlm_seconds": round(sum(float(vision.get("timing_seconds", {}).get("vlm", 0.0)) for vision in visions), 3),
        "case_seconds": round(sum(seconds for _, _, seconds in completed), 3),
    }


def _is_scorable_conclusion(value: str) -> bool:
    return value in {"data_leak_risk_detected", "suspicious_behavior_detected", "no_confirmed_data_leak"}


if __name__ == "__main__":
    sys.exit(main())


