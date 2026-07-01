"""Inspect wrong NAS benchmark cases without changing benchmark decisions."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


def _load_report(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _case_key(case: Dict[str, Any]) -> str:
    return str(case.get("case", "") or "")


def _stage(case_id: str) -> str:
    first = case_id.replace("\\", "/").split("/", 1)[0]
    return first if first.lower().startswith("stage") else "unknown"


def _bucket(case: Dict[str, Any], metric: str) -> str:
    return str(case.get(f"{metric}_bucket", "") or "")


def _vlm_status(case: Dict[str, Any]) -> str:
    verdict = case.get("live_vlm_verdict")
    if isinstance(verdict, dict):
        status = str(verdict.get("status", "") or "")
        reason = str(verdict.get("reason", "") or "")
        return status or reason or "verdict"
    if case.get("vlm_live_queued"):
        return "queued"
    if case.get("vlm_decision") == "run":
        return "triage_only"
    return "none"


def _vlm_action(case: Dict[str, Any]) -> str:
    verdict = case.get("live_vlm_verdict")
    if not isinstance(verdict, dict):
        return ""
    return str(verdict.get("completed_action", "") or verdict.get("risk_level", "") or "")


def _short_reasons(case: Dict[str, Any]) -> str:
    reasons = case.get("vlm_reasons", []) or []
    if not isinstance(reasons, list):
        return str(reasons)
    return ",".join(str(item) for item in reasons[:4])


def _wrong_cases(report: Dict[str, Any], metric: str) -> List[Dict[str, Any]]:
    wanted = {"fp", "fn"}
    return [case for case in report.get("cases", []) if _bucket(case, metric) in wanted]


def _rows(cases: Iterable[Dict[str, Any]], metric: str) -> List[Dict[str, Any]]:
    rows = []
    for case in cases:
        case_id = _case_key(case)
        rows.append(
            {
                "case": case_id,
                "stage": _stage(case_id),
                "bucket": _bucket(case, metric),
                "expected": int(bool(case.get("expected_positive"))),
                "predicted": int(bool(case.get(f"{metric}_positive"))),
                "det": int(bool(case.get("deterministic_positive"))),
                "triage": int(bool(case.get("triage_positive"))),
                "uploads": int(case.get("upload_events", 0) or 0),
                "ops": int(case.get("operation_records", 0) or 0),
                "vlm": _vlm_status(case),
                "action": _vlm_action(case),
                "reasons": _short_reasons(case),
                "log_file": str(case.get("log_file", "") or ""),
            }
        )
    return rows


def _print_table(rows: List[Dict[str, Any]], limit: int) -> None:
    headers = ["bucket", "case", "det", "triage", "uploads", "ops", "vlm", "action", "reasons"]
    visible = rows[:limit] if limit > 0 else rows
    print("| " + " | ".join(headers) + " |")
    print("| " + " | ".join("---" for _ in headers) + " |")
    for row in visible:
        print("| " + " | ".join(str(row.get(header, "")) for header in headers) + " |")
    if limit > 0 and len(rows) > limit:
        print(f"\n... {len(rows) - limit} more")


def _summarize(rows: List[Dict[str, Any]]) -> None:
    print(f"wrong_cases={len(rows)}")
    print("by_bucket=" + json.dumps(Counter(row["bucket"] for row in rows), ensure_ascii=False))
    print("by_stage=" + json.dumps(Counter(row["stage"] for row in rows), ensure_ascii=False))
    print("by_vlm=" + json.dumps(Counter(row["vlm"] for row in rows), ensure_ascii=False))
    reason_counter: Counter[str] = Counter()
    for row in rows:
        for reason in str(row["reasons"]).split(","):
            if reason:
                reason_counter[reason] += 1
    print("top_reasons=" + json.dumps(dict(reason_counter.most_common(10)), ensure_ascii=False))


def _compare_reports(before: Dict[str, Any], after: Dict[str, Any], metric: str) -> Dict[str, List[Dict[str, Any]]]:
    before_cases = {_case_key(case): case for case in before.get("cases", [])}
    after_cases = {_case_key(case): case for case in after.get("cases", [])}
    common = sorted(set(before_cases) & set(after_cases))
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for case_id in common:
        old = before_cases[case_id]
        new = after_cases[case_id]
        old_bucket = _bucket(old, metric)
        new_bucket = _bucket(new, metric)
        row = {
            "case": case_id,
            "from": old_bucket,
            "to": new_bucket,
            "old_vlm": _vlm_status(old),
            "new_vlm": _vlm_status(new),
            "old_action": _vlm_action(old),
            "new_action": _vlm_action(new),
            "old_reasons": _short_reasons(old),
            "new_reasons": _short_reasons(new),
        }
        if old_bucket in {"fp", "fn"} and new_bucket in {"tp", "tn"}:
            groups["fixed"].append(row)
        elif old_bucket in {"tp", "tn"} and new_bucket in {"fp", "fn"}:
            groups["regressed"].append(row)
        elif old_bucket in {"fp", "fn"} and new_bucket in {"fp", "fn"}:
            groups["still_wrong"].append(row)
    return groups


def _print_compare(groups: Dict[str, List[Dict[str, Any]]], limit: int) -> None:
    for name in ("fixed", "regressed", "still_wrong"):
        rows = groups.get(name, [])
        print(f"\n{name}={len(rows)}")
        headers = ["case", "from", "to", "old_vlm", "new_vlm", "old_action", "new_action"]
        print("| " + " | ".join(headers) + " |")
        print("| " + " | ".join("---" for _ in headers) + " |")
        visible = rows[:limit] if limit > 0 else rows
        for row in visible:
            print("| " + " | ".join(str(row.get(header, "")) for header in headers) + " |")
        if limit > 0 and len(rows) > limit:
            print(f"... {len(rows) - limit} more")


def build_error_payload(
    report: Dict[str, Any],
    metric: str = "final",
    compare_report: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    rows = _rows(_wrong_cases(report, metric), metric)
    payload: Dict[str, Any] = {"metric": metric, "wrong_cases": rows}
    if compare_report is not None:
        payload["compare"] = _compare_reports(report, compare_report, metric)
    return payload


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="List and compare wrong benchmark cases.")
    parser.add_argument("report", type=Path, help="Benchmark JSON report.")
    parser.add_argument("--compare", type=Path, help="Compare report against this newer/other report.")
    parser.add_argument("--metric", choices=("final", "triage", "deterministic", "confirmed"), default="final")
    parser.add_argument("--limit", type=int, default=40, help="Rows to print per table; 0 means all.")
    parser.add_argument("--json-output", type=Path, help="Write detailed rows to JSON.")
    args = parser.parse_args(argv)

    report = _load_report(args.report)
    rows = _rows(_wrong_cases(report, args.metric), args.metric)
    print(f"report={args.report}")
    print(f"metric={args.metric}")
    _summarize(rows)
    _print_table(rows, args.limit)

    payload = {"metric": args.metric, "wrong_cases": rows}
    if args.compare:
        other = _load_report(args.compare)
        groups = _compare_reports(report, other, args.metric)
        _print_compare(groups, args.limit)
        payload["compare"] = groups

    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
