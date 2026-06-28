"""Benchmark DataLeakDetector fixture accuracy and VLM fallback pressure.

This runner is intentionally offline: it uses existing fixture expectations and
the project post-processing/log-first code, without calling a live VLM.
"""

import argparse
import importlib.util
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


REPO_ROOT = Path(__file__).resolve().parents[1]
REALISTIC_CASES_PATH = REPO_ROOT / "fixtures" / "realistic_log_cases.json"
QWEN_VLM_CASES_PATH = REPO_ROOT / "fixtures" / "qwen_vlm_response_cases.json"
MISSED_CASES_PATH = REPO_ROOT / "fixtures" / "currently_unrecognized_violation_cases.json"

DEFAULT_BLACKLIST_APPS = [
    "ChatGPT",
    "Feishu",
    "Lark",
    "Gmail",
    "163\u90ae\u7bb1",
    "mail.163.com",
    "msedge.exe",
    "curl.exe",
    "Dropbox.exe",
    "explorer.exe",
]
DEFAULT_WHITELIST_APPS = [
    "Excel",
    "Word",
    "WeCom",
    "\u4f01\u4e1a\u5fae\u4fe1",
    "WeCom.exe",
]


@dataclass
class BinaryMetrics:
    tp: int = 0
    fp: int = 0
    tn: int = 0
    fn: int = 0

    def add(self, expected_positive: bool, predicted_positive: bool) -> None:
        if expected_positive and predicted_positive:
            self.tp += 1
        elif not expected_positive and predicted_positive:
            self.fp += 1
        elif not expected_positive and not predicted_positive:
            self.tn += 1
        else:
            self.fn += 1

    @property
    def total(self) -> int:
        return self.tp + self.fp + self.tn + self.fn

    @property
    def precision(self) -> float:
        denom = self.tp + self.fp
        return self.tp / denom if denom else 1.0

    @property
    def recall(self) -> float:
        denom = self.tp + self.fn
        return self.tp / denom if denom else 1.0

    @property
    def accuracy(self) -> float:
        return (self.tp + self.tn) / self.total if self.total else 1.0

    @property
    def f1(self) -> float:
        denom = self.precision + self.recall
        return 2 * self.precision * self.recall / denom if denom else 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total": self.total,
            "tp": self.tp,
            "fp": self.fp,
            "tn": self.tn,
            "fn": self.fn,
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "accuracy": round(self.accuracy, 4),
            "f1": round(self.f1, 4),
        }


@dataclass
class BenchmarkResult:
    vlm_postprocess: BinaryMetrics = field(default_factory=BinaryMetrics)
    log_triage: BinaryMetrics = field(default_factory=BinaryMetrics)
    deterministic_resolution: BinaryMetrics = field(default_factory=BinaryMetrics)
    cases: List[Dict[str, Any]] = field(default_factory=list)
    estimated_vlm_calls: int = 0
    estimated_vlm_frames: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "summary": {
                "vlm_postprocess": self.vlm_postprocess.to_dict(),
                "log_triage": self.log_triage.to_dict(),
                "deterministic_resolution": self.deterministic_resolution.to_dict(),
            },
            "vlm_pressure": {
                "estimated_vlm_calls": self.estimated_vlm_calls,
                "estimated_vlm_frames": self.estimated_vlm_frames,
                "frame_budget_per_call": _vlm_frame_budget(),
                "note": "Offline estimate: one fallback triage call is counted as one VLM call at the configured frame budget.",
            },
            "cases": self.cases,
        }


def _load_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _vlm_frame_budget() -> int:
    try:
        return max(1, int(os.getenv("DLD_VLM_MAX_FRAMES", "6")))
    except ValueError:
        return 6


def _expected_vlm_kept(case: Dict[str, Any]) -> int:
    if "desired_expected_kept" in case:
        return int(case["desired_expected_kept"])
    return int(case.get("expected_kept", 0))


def _evaluate_vlm_case(agent: Any, case: Dict[str, Any], suite: str) -> Dict[str, Any]:
    parsed = agent._parse_vlm_response_content(case["response"])
    raw_events = agent._coerce_event_list(parsed)
    final_events, meta = agent._filter_vlm_events(raw_events, case["keywords"])
    expected_kept = _expected_vlm_kept(case)
    expected_positive = expected_kept > 0
    predicted_positive = len(final_events) > 0
    return {
        "suite": suite,
        "id": case["id"],
        "expected_positive": expected_positive,
        "predicted_positive": predicted_positive,
        "expected_kept": expected_kept,
        "actual_kept": len(final_events),
        "raw_events": len(raw_events),
        "operation": final_events[0].get("operation_type") if final_events else "",
        "pass": expected_positive == predicted_positive and len(final_events) == expected_kept,
        "meta": meta,
    }


def _expected_log_triage_positive(case: Dict[str, Any]) -> bool:
    expected = case.get("expected", {})
    if expected:
        return int(expected.get("upload_events", 0)) > 0 or expected.get("vlm_decision") == "run"
    desired = case.get("desired_expected", {})
    if desired:
        return desired.get("upload_events_or_vlm_decision") in {"alert", "alert_or_run", "run"}
    return False


def _expected_deterministic_upload(case: Dict[str, Any]) -> bool:
    expected = case.get("expected", {})
    if expected:
        return int(expected.get("upload_events", 0)) > 0
    desired = case.get("desired_expected", {})
    return desired.get("upload_events_or_vlm_decision") in {"alert", "alert_or_run"}


def _evaluate_log_case(
    log_first_module: Any,
    run_e2e_module: Any,
    case: Dict[str, Any],
    suite: str,
) -> Dict[str, Any]:
    detector = log_first_module.LogFirstDetector(
        sensitive_files=case["sensitive_files"],
        blacklist_apps=DEFAULT_BLACKLIST_APPS,
        whitelist_apps=DEFAULT_WHITELIST_APPS,
    )
    result = detector.analyze(case["logs"])
    should_run_vlm, fallback_meta = run_e2e_module._should_use_vlm_fallback(case["logs"], result)
    upload_count = len(result.get("upload_events", []))
    alert_count = len(result.get("alert_events", []))
    predicted_triage_positive = upload_count > 0 or should_run_vlm

    return {
        "suite": suite,
        "id": case["id"],
        "expected_positive": _expected_log_triage_positive(case),
        "predicted_positive": predicted_triage_positive,
        "expected_deterministic_upload": _expected_deterministic_upload(case),
        "predicted_deterministic_upload": upload_count > 0,
        "upload_events": upload_count,
        "alert_events": alert_count,
        "vlm_decision": fallback_meta.get("decision"),
        "vlm_reasons": fallback_meta.get("reasons", []),
        "operation_types": [event.operation_type for event in result.get("upload_events", [])],
        "pass": _expected_log_triage_positive(case) == predicted_triage_positive,
    }


def _iter_vlm_fixture_cases(include_missed: bool) -> Iterable[tuple[str, Dict[str, Any]]]:
    for case in _read_json(QWEN_VLM_CASES_PATH):
        yield "qwen_vlm_response_cases", case
    if include_missed:
        missed = _read_json(MISSED_CASES_PATH)
        for case in missed["vlm_postprocess_misses"]:
            yield "currently_unrecognized_vlm_cases", case


def _iter_log_fixture_cases(include_missed: bool) -> Iterable[tuple[str, Dict[str, Any]]]:
    for case in _read_json(REALISTIC_CASES_PATH):
        yield "realistic_log_cases", case
    if include_missed:
        missed = _read_json(MISSED_CASES_PATH)
        for case in missed["log_first_and_fallback_misses"]:
            yield "currently_unrecognized_log_cases", case


def run_benchmark(include_missed: bool = True) -> Dict[str, Any]:
    frame_dir = REPO_ROOT / "1-FrameAnalyzer"
    risk_dir = REPO_ROOT / "3-RiskHunter"
    sys.path.insert(0, str(frame_dir))
    sys.path.insert(0, str(risk_dir))
    try:
        agent_module = _load_module("benchmark_frame_agent", frame_dir / "agent.py")
        log_first_module = _load_module("benchmark_log_first_detector", risk_dir / "log_first_detector.py")
        run_e2e_module = _load_module("benchmark_run_e2e", REPO_ROOT / "run_e2e.py")
    finally:
        for path in (str(risk_dir), str(frame_dir)):
            if path in sys.path:
                sys.path.remove(path)

    agent = agent_module.VideoFileOperationAgent.__new__(agent_module.VideoFileOperationAgent)
    result = BenchmarkResult()

    for suite, case in _iter_vlm_fixture_cases(include_missed):
        item = _evaluate_vlm_case(agent, case, suite)
        result.vlm_postprocess.add(item["expected_positive"], item["predicted_positive"])
        result.cases.append(item)

    for suite, case in _iter_log_fixture_cases(include_missed):
        item = _evaluate_log_case(log_first_module, run_e2e_module, case, suite)
        result.log_triage.add(item["expected_positive"], item["predicted_positive"])
        result.deterministic_resolution.add(
            item["expected_deterministic_upload"],
            item["predicted_deterministic_upload"],
        )
        if item["upload_events"] == 0 and item["vlm_decision"] == "run":
            result.estimated_vlm_calls += 1
            result.estimated_vlm_frames += _vlm_frame_budget()
        result.cases.append(item)

    return result.to_dict()


def _print_human_report(report: Dict[str, Any]) -> None:
    print("\nDataLeakDetector Offline Benchmark")
    print("=" * 40)
    for name, metrics in report["summary"].items():
        print(
            f"{name:24} "
            f"precision={metrics['precision']:.4f} "
            f"recall={metrics['recall']:.4f} "
            f"f1={metrics['f1']:.4f} "
            f"accuracy={metrics['accuracy']:.4f} "
            f"(tp={metrics['tp']}, fp={metrics['fp']}, tn={metrics['tn']}, fn={metrics['fn']})"
        )
    pressure = report["vlm_pressure"]
    print("\nVLM pressure")
    print(
        f"estimated_vlm_calls={pressure['estimated_vlm_calls']}, "
        f"estimated_vlm_frames={pressure['estimated_vlm_frames']}, "
        f"frame_budget_per_call={pressure['frame_budget_per_call']}"
    )

    failures = [case for case in report["cases"] if not case.get("pass")]
    print(f"\ncase_failures={len(failures)}")
    for case in failures:
        print(
            f"- {case['suite']}::{case['id']} "
            f"expected={case.get('expected_positive')} predicted={case.get('predicted_positive')}"
        )


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Run offline detection benchmark over fixture suites.")
    parser.add_argument(
        "--exclude-missed",
        action="store_true",
        help="Only run original fixture suites, excluding the red-team missed-case fixture.",
    )
    parser.add_argument("--json-output", type=Path, help="Write full benchmark JSON to this path.")
    parser.add_argument("--json", action="store_true", help="Print JSON report instead of human summary.")
    args = parser.parse_args(argv)

    report = run_benchmark(include_missed=not args.exclude_missed)
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        _print_human_report(report)

    failures = [case for case in report["cases"] if not case.get("pass")]
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
