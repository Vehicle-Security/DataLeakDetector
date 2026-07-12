"""Continuously adjudicate completed Release disagreements while the run is active."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

from adjudicate_release import _case_evidence, _effective_conclusion, _request_decision, _summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Watch release_progress.json and adjudicate completed disagreements.")
    parser.add_argument("--release-progress", required=True)
    parser.add_argument("--case-debug-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--base-url", default=os.getenv("DLD_JUDGE_BASE_URL", "https://api.deepseek.com"))
    parser.add_argument("--model", default=os.getenv("DLD_JUDGE_MODEL", "deepseek-v4-pro"))
    parser.add_argument("--api-key-env", default="DLD_JUDGE_API_KEY")
    parser.add_argument("--poll-seconds", type=float, default=5.0)
    args = parser.parse_args()

    api_key = os.getenv(args.api_key_env, "").strip()
    if not api_key:
        parser.error(f"missing API key environment variable: {args.api_key_env}")

    progress_path = Path(args.release_progress)
    debug_root = Path(args.case_debug_root)
    output_path = Path(args.output)
    output = _load_output(output_path, progress_path, args.model, args.base_url)
    handled = {str(item.get("case") or "") for item in output["cases"]}

    while True:
        progress = _read_json(progress_path)
        for raw_case in progress.get("recent_cases", []):
            if not _is_disputed_completion(raw_case) or raw_case["case"] in handled:
                continue
            evidence = _case_evidence(_progress_case_to_report_case(raw_case), debug_root)
            if not evidence["evidence_refs"].get("parsed"):
                continue
            response = _request_decision(
                base_url=args.base_url,
                api_key=api_key,
                model=args.model,
                evidence=evidence,
            )
            result = {
                "case": evidence["case"],
                "expected_conclusion": evidence["expected_conclusion"],
                "detector_conclusion": evidence["detector_conclusion"],
                "decision": response["decision"],
                "reason": response["reason"],
                "evidence_refs": evidence["evidence_refs"],
                "effective_conclusion": _effective_conclusion(evidence, response["decision"]),
            }
            result["effective_correct"] = result["effective_conclusion"] == evidence["detector_conclusion"] if result["effective_conclusion"] else None
            output["cases"].append(result)
            output["summary"] = _summary(output["cases"])
            _write_json(output_path, output)
            handled.add(result["case"])
            print(f"adjudicated case={result['case']} decision={result['decision']}", flush=True)
        if progress.get("state") == "completed":
            return 0
        time.sleep(max(1.0, args.poll_seconds))


def _is_disputed_completion(case: dict[str, Any]) -> bool:
    return case.get("state") == "completed" and case.get("score_status") == "scored" and case.get("detector_correct") is False


def _progress_case_to_report_case(case: dict[str, Any]) -> dict[str, Any]:
    return {
        "case": case["case"],
        "case_relative_path": case["case"],
        "conclusion": case.get("conclusion", ""),
        "evaluation": case,
    }


def _load_output(path: Path, progress_path: Path, model: str, base_url: str) -> dict[str, Any]:
    if path.exists():
        return _read_json(path)
    return {
        "schema_version": 1,
        "source_release_progress": str(progress_path),
        "model": model,
        "base_url": base_url.rstrip("/"),
        "summary": _summary([]),
        "cases": [],
    }


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(path)


if __name__ == "__main__":
    raise SystemExit(main())
