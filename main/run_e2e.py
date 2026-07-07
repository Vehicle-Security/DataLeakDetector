"""Command-line entry point for the canonical DataLeakDetector pipeline.

This file exists so operators can run the project without importing Python
objects directly. It deliberately stays thin: argument parsing and JSON output
belong here, while detection logic remains in data_leak_detector.pipeline.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from data_leak_detector import run_pipeline


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run DataLeakDetector end-to-end.")
    parser.add_argument("--log", "-l", required=True, help="Path to a JSON/JSONL monitor log.")
    parser.add_argument("--video", "-v", default="", help="Optional video path for report metadata.")
    parser.add_argument("--output-dir", "-o", default="", help="Optional directory for the JSON report.")
    parser.add_argument("--sensitive-file", action="append", default=[], help="Sensitive file path. Can be repeated.")
    parser.add_argument("--observations", default="", help="Optional precomputed frame observation JSON.")
    parser.add_argument("--neo4j", action="store_true", help="Write the report graph to Neo4j for this run.")
    parser.add_argument("--neo4j-strict", action="store_true", help="Fail if Neo4j writing fails.")
    args = parser.parse_args(argv)

    report = run_pipeline(
        log_file=args.log,
        video_file=args.video,
        output_dir=args.output_dir or None,
        sensitive_files=args.sensitive_file,
        observations_file=args.observations or None,
        neo4j_enabled=True if args.neo4j else None,
        neo4j_strict=True if args.neo4j_strict else None,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
