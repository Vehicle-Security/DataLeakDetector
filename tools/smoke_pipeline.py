"""Quick local health check for the canonical pipeline.

This helper is intentionally smaller than the E2E CLI: it runs the sample leak
fixture and prints only the summary and graph status. It exists so a developer
can verify the environment quickly after dependency or Neo4j changes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from data_leak_detector import run_pipeline


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a quick pipeline smoke test.")
    parser.add_argument("--log", default="spec/fixtures/sample_leak.json")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--neo4j", action="store_true")
    args = parser.parse_args()

    report = run_pipeline(
        log_file=Path(args.log),
        output_dir=args.output_dir or None,
        neo4j_enabled=True if args.neo4j else None,
    )
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    print(json.dumps(report["graph"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
