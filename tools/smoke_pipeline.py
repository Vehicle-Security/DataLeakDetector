"""规范流水线的本地快速健康检查。

这个辅助脚本刻意比端到端 CLI 更小：它运行样例泄露 fixture，只打印摘要和图状态。这样开发者就能在
依赖或 Neo4j 变更后快速验证环境。
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
