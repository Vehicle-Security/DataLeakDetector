"""DataLeakDetector 规范流水线的命令行入口。

这个文件的存在是为了让运维人员无需直接导入 Python 对象即可运行项目。
它刻意保持轻量：参数解析和 JSON 输出放在这里，而检测逻辑保留在 data_leak_detector.pipeline 中。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from data_leak_detector import run_pipeline


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="运行 DataLeakDetector 端到端流程。")
    parser.add_argument("--log", "-l", required=True, help="JSON/JSONL 监控日志路径。")
    parser.add_argument("--video", "-v", default="", help="用于报告元数据的可选视频路径。")
    parser.add_argument("--output-dir", "-o", default="", help="用于 JSON 报告的可选输出目录。")
    parser.add_argument("--sensitive-file", action="append", default=[], help="敏感文件路径，可重复传入。")
    parser.add_argument("--observations", default="", help="可选的预计算帧观察 JSON。")
    parser.add_argument("--neo4j", action="store_true", help="将本次运行的报告图写入 Neo4j。")
    parser.add_argument("--neo4j-strict", action="store_true", help="Neo4j 写入失败时直接报错。")
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
