"""Quick health check against the real spec/data sample layout."""

from __future__ import annotations

import argparse
import json

from data_leak_detector import run_data_case


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a quick pipeline smoke test.")
    parser.add_argument("--case", default=r"spec\data\nas_samples\stage1\0-normal-ai-chatgpt-1")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--neo4j", action="store_true")
    parser.add_argument("--vision", action="store_true")
    args = parser.parse_args()

    report = run_data_case(
        args.case,
        output_dir=args.output_dir or None,
        neo4j_enabled=bool(args.neo4j),
        vision_enabled=bool(args.vision),
    )
    print(json.dumps(report["input"], ensure_ascii=False, indent=2))
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    print(json.dumps(report["detection_core"], ensure_ascii=False, indent=2))
    print(json.dumps(report["verdict"], ensure_ascii=False, indent=2))
    print(json.dumps(report["graph"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
