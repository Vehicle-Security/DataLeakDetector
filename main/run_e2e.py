"""Command-line entry point for the canonical DataLeakDetector pipeline."""

from __future__ import annotations

import argparse
import json
import sys

from data_leak_detector import run_data_case, run_pipeline


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Run DataLeakDetector end-to-end.")
    parser.add_argument("--log", "-l", default="", help="Path to a JSON/JSONL monitor log.")
    parser.add_argument("--case", "-c", default="", help="Path to a spec/data sample case directory.")
    parser.add_argument("--video", "-v", default="", help="Optional screen recording path for frame analysis.")
    parser.add_argument("--groundtruth", default="", help="Optional groundtruth.json path for verdict evaluation.")
    parser.add_argument("--output-dir", "-o", default="", help="Optional directory for the JSON report.")
    parser.add_argument("--sensitive-file", action="append", default=[], help="Sensitive file path. Can be repeated.")
    parser.add_argument("--observations", default="", help="Optional precomputed frame observation JSON.")
    parser.add_argument("--vision", action="store_true", help="Enable OCR/VLM-assisted frame analysis.")
    parser.add_argument("--vision-mode", choices=["hybrid", "ocr", "vlm"], default="", help="Frame analysis mode.")
    parser.add_argument("--max-vlm-frames", type=int, default=0, help="Maximum keyframes sent to VLM.")
    parser.add_argument("--no-non-vlm", action="store_true", help="Disable deterministic OCR/log evidence in correlation; useful for VLM-only evaluation.")
    parser.add_argument("--neo4j", action="store_true", help="Write the report graph to Neo4j for this run.")
    parser.add_argument("--neo4j-strict", action="store_true", help="Fail if Neo4j writing fails.")
    parser.add_argument("--neo4j-log-miner", action="store_true", help="Use Neo4j to mine analysis windows before frame extraction.")
    parser.add_argument("--no-reuse-neo4j-import", action="store_true", help="Reimport logs even when the case fingerprint already exists in Neo4j.")
    args = parser.parse_args(argv)

    common_args = {
        "output_dir": args.output_dir or None,
        "sensitive_files": args.sensitive_file,
        "observations_file": args.observations or None,
        "neo4j_enabled": True if args.neo4j else None,
        "neo4j_strict": True if args.neo4j_strict else None,
        "neo4j_log_miner": True if args.neo4j_log_miner else None,
        "reuse_neo4j_import": False if args.no_reuse_neo4j_import else None,
        "vision_enabled": True if args.vision else None,
        "vision_mode": args.vision_mode or None,
        "max_vlm_frames": args.max_vlm_frames or None,
        "non_vlm_enabled": False if args.no_non_vlm else None,
    }
    if args.case:
        report = run_data_case(args.case, **common_args)
    else:
        if not args.log:
            parser.error("either --log or --case is required")
        report = run_pipeline(log_file=args.log, video_file=args.video, groundtruth_file=args.groundtruth or None, **common_args)

    print(json.dumps(_build_cli_summary(report), ensure_ascii=False, indent=2))
    return 0


def _build_cli_summary(report: dict) -> dict:
    """Keep command output readable; full evidence lives in report/detail files."""

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


if __name__ == "__main__":
    sys.exit(main())
