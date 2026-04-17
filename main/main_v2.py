import argparse
import ctypes
import json
import os
import re
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
MAIN_ROOT = Path(__file__).resolve().parent
FRAME_ROOT = REPO_ROOT / "01-FrameAnalyzer"
CORRELATOR_ROOT = REPO_ROOT / "02-EventCorrelator"
REASONER_ROOT = REPO_ROOT / "03-LeakReasoner"

for path in (MAIN_ROOT, FRAME_ROOT, CORRELATOR_ROOT, REASONER_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


from event_correlator import EventCorrelator  # noqa: E402
from frame_analyzer import FrameAnalyzerRequest, FrameAnalyzerService  # noqa: E402
from leak_reasoner import LeakReasoner  # noqa: E402

from pipeline_support import (  # noqa: E402
    SampleContext,
    build_demo_segments as build_demo_segments_for_sample,
    discover_sample_roots,
    load_pipeline_config,
    load_sample_context,
)


def get_windows_desktop() -> Path:
    if os.name != "nt":
        return Path.home() / "Desktop"

    buffer = ctypes.create_unicode_buffer(260)
    result = ctypes.windll.shell32.SHGetFolderPathW(None, 0x0010, None, 0, buffer)
    if result == 0 and buffer.value:
        return Path(buffer.value)
    return Path.home() / "Desktop"


USER_PATH_PATTERN = re.compile(r"([A-Za-z]:[\\/]+Users[\\/]+)([^\\/]+)")


def _sanitize_export_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _sanitize_export_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_sanitize_export_value(item) for item in value]
    if isinstance(value, str):
        return USER_PATH_PATTERN.sub(r"\1<redacted>", value)
    return value


def _load_log_events(sample_root: Path) -> list[dict]:
    log_path = sample_root / "logs" / "keyevents.json"
    with log_path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def build_demo_segments(record_id: str) -> list[dict]:
    sample_root = get_windows_desktop() / record_id
    if not sample_root.exists():
        return []
    log_events = _load_log_events(sample_root)
    pipeline_config = load_pipeline_config()
    return build_demo_segments_for_sample(sample_root, log_events, pipeline_config)


def build_sensitive_files(record_id: str) -> list[str]:
    sample_root = get_windows_desktop() / record_id
    if not sample_root.exists():
        return []
    log_events = _load_log_events(sample_root)
    pipeline_config = load_pipeline_config()
    return load_sample_context(sample_root, log_events, pipeline_config).sensitive_files


def build_full_mode_request(record_id: str, sample_root: Path) -> FrameAnalyzerRequest | None:
    if not sample_root.exists():
        return None
    log_events = _load_log_events(sample_root)
    pipeline_config = load_pipeline_config()
    context = load_sample_context(sample_root, log_events, pipeline_config, mode="full")
    return FrameAnalyzerRequest(
        video_path=context.video_path,
        recording_start_time=context.recording_start_time,
        search_start_time=context.search_start_time,
        search_end_time=context.search_end_time,
        target_keywords=context.target_keywords,
    )


def _build_sample_context(sample_root: Path, log_events: list[dict], pipeline_config: dict) -> SampleContext:
    return load_sample_context(sample_root, log_events, pipeline_config, mode="full")


def _cleanup_stale_frame_caches(sample_contexts: list[SampleContext]) -> None:
    cache_dir = Path(os.environ.get("FRAME_ANALYZER_CACHE_DIR", "output/frame_cache"))
    if not cache_dir.exists():
        return

    service = FrameAnalyzerService()
    expected_names = set()
    for context in sample_contexts:
        request = FrameAnalyzerRequest(
            video_path=context.video_path,
            recording_start_time=context.recording_start_time,
            search_start_time=context.search_start_time,
            search_end_time=context.search_end_time,
            target_keywords=context.target_keywords,
        )
        expected_names.add(service._cache_path(request).name)

    keep_names = set(expected_names)
    keep_names.add("README.md")

    for cache_path in cache_dir.iterdir():
        if cache_path.name in keep_names:
            continue
        try:
            if cache_path.is_file():
                cache_path.unlink()
        except OSError:
            continue


def _build_frame_analysis_record(
    *,
    mode: str,
    status: str,
    metadata: dict,
    segments: list[dict],
    summary: dict | None = None,
) -> dict:
    return {
        "mode": mode,
        "status": status,
        "metadata": dict(metadata or {}),
        "segments": list(segments or []),
        "summary": dict(summary or {}),
    }


def run_single_sample(
    sample_root: Path,
    mode: str = "full",
    pipeline_config: dict | None = None,
    *,
    fresh_run: bool = False,
) -> dict:
    record_id = sample_root.name
    log_events = _load_log_events(sample_root)
    pipeline_config = pipeline_config or load_pipeline_config()
    context = _build_sample_context(sample_root, log_events, pipeline_config)

    if mode == "demo":
        frame_segments = build_demo_segments_for_sample(sample_root, log_events, pipeline_config)
        frame_analysis_status = "demo_segments"
        analysis_metadata = {
            "analysis_backend": "demo_segments",
            "analysis_backend_version": "demo_segments_v1",
            "cache_hit": False,
            "cache_schema_version": "demo",
            "fresh_run_requested": False,
            "prompt_signature": "demo_segments_v1",
        }
        frame_summary = {
            "apps": sorted({segment.get("app_name", "") for segment in frame_segments if segment.get("app_name", "")}),
            "operations": sorted(
                {segment.get("operation_type", "") for segment in frame_segments if segment.get("operation_type", "")}
            ),
            "resources": sorted(
                {
                    item
                    for segment in frame_segments
                    for item in [segment.get("primary_resource", ""), *(segment.get("related_resources", []) or [])]
                    if item
                }
            ),
        }
    else:
        frame_segments = []
        frame_analysis_status = "not_started"
        analysis_metadata = {}
        frame_summary = {}
        print(f"[full] starting FrameAnalyzer on sample {record_id} ...")
        request = FrameAnalyzerRequest(
            video_path=context.video_path,
            recording_start_time=context.recording_start_time,
            search_start_time=context.search_start_time,
            search_end_time=context.search_end_time,
            target_keywords=context.target_keywords,
            force_refresh=fresh_run,
        )
        frame_result = FrameAnalyzerService().analyze(request)
        frame_analysis_status = frame_result.get("status", "unknown")
        analysis_metadata = dict(frame_result.get("analysis_metadata", {}) or {})
        frame_summary = dict(frame_result.get("summary", {}) or {})
        if frame_result.get("status") == "success":
            frame_segments = frame_result.get("segments", [])
            print(
                f"[full] FrameAnalyzer finished for {record_id}: "
                f"segments={len(frame_segments)}, cache_hit={analysis_metadata.get('cache_hit', False)}"
            )
        else:
            frame_segments = []
            print(
                f"[full] FrameAnalyzer failed for {record_id}: "
                f"status={frame_analysis_status}, cache_hit={analysis_metadata.get('cache_hit', False)}"
            )

    frame_analysis = _build_frame_analysis_record(
        mode=mode,
        status=frame_analysis_status,
        metadata=analysis_metadata,
        segments=frame_segments,
        summary=frame_summary,
    )

    frame_analysis_failed = mode == "full" and frame_analysis_status != "success"

    if frame_analysis_failed:
        correlation_bundle = {
            "session_id": record_id,
            "analysis_status": "blocked_by_frame_analyzer",
            "correlated_events": [],
            "operation_records": [],
            "upload_candidates": [],
            "file_lineage": {
                "direct_file_mappings": {},
                "full_file_mapping_chains": {},
            },
            "statistics": {
                "log_events_input": len(log_events),
                "frame_segments_input": 0,
                "correlated_events_output": 0,
                "upload_candidates_output": 0,
                "lineage_direct_mappings": 0,
                "lineage_full_chains": 0,
            },
            "errors": [
                {
                    "stage": "frame_analyzer",
                    "code": "frame_analysis_failed",
                    "message": (
                        "full mode requires successful FrameAnalyzer output; "
                        "log-only fallback is blocked for acceptance integrity"
                    ),
                    "status": frame_analysis_status,
                }
            ],
        }
        reasoner_output = {
            "session_id": record_id,
            "analysis_status": "blocked_by_frame_analyzer",
            "risk_cases": [],
            "evidence_bundles": [],
            "metrics": {
                "facts_input": 0,
                "upload_candidates_input": 0,
                "risk_cases_output": 0,
                "leak_paths_output": 0,
            },
            "errors": [
                {
                    "stage": "frame_analyzer",
                    "code": "frame_analysis_failed",
                    "message": "LeakReasoner was not run because FrameAnalyzer did not succeed in full mode",
                    "status": frame_analysis_status,
                }
            ],
        }
    else:
        correlator_input = {
            "session_id": record_id,
            "record_id": record_id,
            "recording_start_time": context.recording_start_time,
            "log_events": log_events,
            "frame_segments": frame_segments,
            "sensitive_files": list(context.sensitive_files),
            "session_metadata": {
                "sample_root": str(sample_root),
                "frame_mode": mode,
                "frame_analysis_status": frame_analysis_status,
                "frame_analysis_metadata": analysis_metadata,
                "frame_analysis_summary": frame_summary,
                "pipeline_context": context.as_dict(),
            },
        }

        correlation_bundle = EventCorrelator().run(correlator_input)
        reasoner_output = LeakReasoner().run(
            {
                "session_id": record_id,
                "correlation_bundle": correlation_bundle,
            }
        )

    return {
        "record_id": record_id,
        "frame_mode": mode,
        "fresh_run": fresh_run,
        "sample_context": context.as_dict(),
        "frame_analysis": frame_analysis,
        "correlation_bundle": correlation_bundle,
        "reasoner_output": reasoner_output,
        "detected": bool(reasoner_output.get("risk_cases", [])),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the new 01 -> 02 -> 03 pipeline.")
    parser.add_argument(
        "--samples",
        nargs="*",
        default=["10-2", "5-2"],
        help="Sample names under desktop. Default: 10-2 5-2",
    )
    parser.add_argument(
        "--samples-root",
        type=str,
        default=None,
        help="Optional directory that contains sample folders. Default: Windows Desktop",
    )
    parser.add_argument(
        "--mode",
        choices=["demo", "full"],
        default="full",
        help="demo = derive demo segments from sample metadata; full = run real FrameAnalyzer backend",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Optional pipeline config path",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(REPO_ROOT / "output" / "e2e_v2_summary.json"),
        help="Output summary JSON path",
    )
    parser.add_argument(
        "--fresh-run",
        action="store_true",
        help="Force FrameAnalyzer to ignore cached snapshots and rerun visual analysis",
    )
    args = parser.parse_args()

    samples_root = Path(args.samples_root) if args.samples_root else get_windows_desktop()
    pipeline_config = load_pipeline_config(args.config)
    sample_roots = discover_sample_roots(samples_root, args.samples)

    sample_contexts = []
    for sample_root in sample_roots:
        log_events = _load_log_events(sample_root)
        sample_contexts.append(_build_sample_context(sample_root, log_events, pipeline_config))
    _cleanup_stale_frame_caches(sample_contexts)

    results = []
    for sample_root in sample_roots:
        print(f"[pipeline] sample={sample_root.name}, mode={args.mode}")
        results.append(
            run_single_sample(
                sample_root,
                mode=args.mode,
                pipeline_config=pipeline_config,
                fresh_run=args.fresh_run,
            )
        )

    detected_count = sum(1 for item in results if item["detected"])
    summary = {
        "samples_total": len(results),
        "samples_detected": detected_count,
        "success_rate": f"{detected_count}/{len(results)}" if results else "0/0",
        "results": results,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(_sanitize_export_value(summary), fh, ensure_ascii=False, indent=2)

    print(f"saved: {output_path}")
    print(f"success_rate: {summary['success_rate']}")
    for item in results:
        print(
            f"{item['record_id']}: mode={item['frame_mode']}, detected={item['detected']}, "
            f"cases={len(item['reasoner_output'].get('risk_cases', []))}, "
            f"uploads={len(item['correlation_bundle'].get('upload_candidates', []))}"
        )


if __name__ == "__main__":
    main()
