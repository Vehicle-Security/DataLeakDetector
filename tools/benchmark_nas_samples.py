"""Evaluate log-first triage over downloaded NAS samples.

The runner is intentionally data-shape driven rather than sample-name driven:
it discovers cases from ``groundtruth.json`` plus logs, prefers key event logs,
and reports whether the current detector can either resolve an event
deterministically or route it to VLM review.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = REPO_ROOT / "data" / "nas_samples"
LOG_FILE_PRIORITY = ("keyevents.json", "logs.json")

RISK_LABEL_PREFIXES = (
    "\u76f4\u63a5\u5916\u53d1",
    "\u76f4\u63a5\u4e0a\u4f20",
    "\u6f5c\u5728\u9690\u85cf\u884c\u4e3a",
    "\u9690\u85cf\u884c\u4e3a",
    "\u8fdd\u89c4",
    "\u654f\u611f\u64cd\u4f5c",
    "\u526a\u8d34\u677f\u64cd\u4f5c",
    "\u5e94\u7528\u5207\u6362",
)
BENIGN_LABEL_PREFIXES = ("\u6b63\u5e38\u64cd\u4f5c",)

DEFAULT_BLACKLIST_APPS = [
    "ChatGPT",
    "Claude",
    "Gemini",
    "DeepSeek",
    "Kimi",
    "Poe",
    "Doubao",
    "Tongyi",
    "Yiyan",
    "Feishu",
    "Lark",
    "DingTalk",
    "WeChat",
    "QQ",
    "TIM",
    "Discord",
    "Slack",
    "Gmail",
    "Outlook",
    "Proton",
    "163",
    "mail.",
    "Google Drive",
    "OneDrive",
    "Dropbox",
    "Baidu",
    "Quark",
    "Weiyun",
    "CSDN",
    "GitHub",
    "GitLab",
    "Gitee",
    "Bitbucket",
    "Medium",
    "Notion",
    "Yuque",
    "Zoom",
    "Teams",
    "Edge",
    "Chrome",
    "msedge.exe",
    "chrome.exe",
]
DEFAULT_WHITELIST_APPS = [
    "Word",
    "WINWORD",
    "Excel",
    "EXCEL",
    "WPS",
    "Explorer",
    "Finder",
    "kdesk64",
]

EXTRA_LOG_TOKENS = (
    "chatgpt",
    "claude",
    "gemini",
    "deepseek",
    "kimi",
    "poe",
    "doubao",
    "tongyi",
    "teams",
    "wechat",
    "qq",
    "dingtalk",
    "feishu",
    "lark",
    "gmail",
    "outlook",
    "mail",
    "drive",
    "upload",
    "attach",
    "send",
    "share",
    "meeting",
    "\u4e0a\u4f20",
    "\u9644\u4ef6",
    "\u53d1\u9001",
    "\u5206\u4eab",
    "\u4f1a\u8bae",
    "\u90ae\u7bb1",
)


@dataclass
class Metrics:
    tp: int = 0
    fp: int = 0
    tn: int = 0
    fn: int = 0

    def add(self, expected: bool, predicted: bool) -> str:
        if expected and predicted:
            self.tp += 1
            return "tp"
        if expected and not predicted:
            self.fn += 1
            return "fn"
        if not expected and predicted:
            self.fp += 1
            return "fp"
        self.tn += 1
        return "tn"

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
            "f1": round(self.f1, 4),
        }


@dataclass
class BenchmarkSummary:
    triage: Metrics = field(default_factory=Metrics)
    deterministic: Metrics = field(default_factory=Metrics)
    cases: List[Dict[str, Any]] = field(default_factory=list)
    skipped: List[Dict[str, str]] = field(default_factory=list)
    deterministic_hits: int = 0
    vlm_reviews: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "summary": {
                "triage": self.triage.to_dict(),
                "deterministic": self.deterministic.to_dict(),
                "deterministic_hits": self.deterministic_hits,
                "vlm_reviews": self.vlm_reviews,
                "skipped_cases": len(self.skipped),
            },
            "cases": self.cases,
            "skipped": self.skipped,
        }


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _read_text(path: Path) -> str:
    for encoding in ("utf-8-sig", "utf-8", "gb18030"):
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
    return path.read_text(encoding="utf-8", errors="replace")


def _read_json_lenient(path: Path) -> Any:
    text = _read_text(path).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        decoder = json.JSONDecoder()
        items = []
        pos = 0
        while pos < len(text):
            while pos < len(text) and text[pos].isspace():
                pos += 1
            if pos >= len(text):
                break
            item, end = decoder.raw_decode(text[pos:])
            items.append(item)
            pos += end
        if items:
            return items
        raise


def _choose_log_file(case_dir: Path) -> Optional[Path]:
    logs_dir = case_dir / "logs"
    if not logs_dir.exists():
        return None
    for name in LOG_FILE_PRIORITY:
        candidate = logs_dir / name
        if candidate.exists():
            return candidate
    files = sorted(logs_dir.glob("*.json"), key=lambda item: item.stat().st_size)
    return files[0] if files else None


def _log_key(log: Dict[str, Any]) -> tuple:
    return (
        str(log.get("timestamp", "")),
        str(log.get("event_type", "")),
        str(log.get("file_path", "")),
        str(log.get("window_info", {}).get("window_title", "")),
    )


def _is_extra_log_relevant(log: Dict[str, Any], log_first: Any) -> bool:
    text = " ".join(
        str(part or "")
        for part in (
            log.get("event_type", ""),
            log.get("file_path", ""),
            log.get("file_name", ""),
            log.get("content_preview", ""),
            log.get("app_name", ""),
            log.get("process_info", {}).get("process_name", ""),
            log.get("window_info", {}).get("window_title", ""),
        )
    )
    lowered = text.lower()
    return log_first.is_sensitive_name(text) or any(token in lowered for token in EXTRA_LOG_TOKENS)


def _load_case_logs(case_dir: Path, log_first: Any) -> tuple[List[Dict[str, Any]], str]:
    primary = _choose_log_file(case_dir)
    if not primary:
        return [], ""
    logs = _read_json_lenient(primary)
    if not isinstance(logs, list):
        raise ValueError("primary log file is not a JSON array")

    source_name = primary.name
    full_log = case_dir / "logs" / "logs.json"
    if primary.name == "keyevents.json" and full_log.exists():
        try:
            full_logs = _read_json_lenient(full_log)
        except Exception:
            full_logs = []
        if isinstance(full_logs, list):
            seen = {_log_key(item) for item in logs if isinstance(item, dict)}
            added = 0
            max_extra = int(os.getenv("DLD_NAS_MAX_EXTRA_LOG_EVENTS", "300"))
            for item in full_logs:
                if not isinstance(item, dict) or not _is_extra_log_relevant(item, log_first):
                    continue
                key = _log_key(item)
                if key in seen:
                    continue
                seen.add(key)
                logs.append(item)
                added += 1
                if added >= max_extra:
                    break
            if added:
                source_name = f"{primary.name}+logs.json:{added}"

    return logs, source_name


def _choose_groundtruth(case_dir: Path) -> Optional[Path]:
    for name in ("groundtruth.json", "groudtruth.json", "groungtruth.json"):
        candidate = case_dir / name
        if candidate.exists():
            return candidate
    return None


def _operation_items(groundtruth: Any) -> List[Dict[str, Any]]:
    if isinstance(groundtruth, dict):
        ops = groundtruth.get("operations", [])
        return [item for item in ops if isinstance(item, dict)]
    if isinstance(groundtruth, list):
        merged: List[Dict[str, Any]] = []
        for item in groundtruth:
            if isinstance(item, dict):
                merged.extend(_operation_items(item))
        return merged
    return []


def _is_risk_label(label: str) -> bool:
    text = str(label or "").strip()
    if not text:
        return False
    if any(text.startswith(prefix) for prefix in BENIGN_LABEL_PREFIXES):
        return False
    return any(text.startswith(prefix) for prefix in RISK_LABEL_PREFIXES)


def _expected_positive(groundtruth: Any) -> bool:
    return any(_is_risk_label(item.get("operation", "")) for item in _operation_items(groundtruth))


def _sensitive_files_from_groundtruth(groundtruth: Any) -> List[str]:
    seen = set()
    files = []
    for item in _operation_items(groundtruth):
        path = str(item.get("sensitive_file_path", "") or "").strip()
        if path and path.lower() not in seen:
            seen.add(path.lower())
            files.append(path)
    return files


def _sensitive_files_from_logs(logs: List[Dict[str, Any]], log_first: Any) -> List[str]:
    seen = set()
    files = []
    for log in logs:
        hint = log_first.file_hint_from_log(log)
        path = log_first.normalize_path(log.get("file_path", "") or hint)
        if hasattr(log_first, "is_system_noise_path") and log_first.is_system_noise_path(path):
            continue
        name = log.get("file_name") or log_first.basename(path)
        content_preview = str(log.get("content_preview", "") or "")
        if path and log_first.is_sensitive_name(f"{name} {content_preview}"):
            key = log_first.file_key(path)
            if key not in seen:
                seen.add(key)
                files.append(path)
    return files


def _case_dirs(root: Path, stages: Optional[List[str]]) -> Iterable[Path]:
    stage_dirs = [root / stage for stage in stages] if stages else [path for path in root.iterdir() if path.is_dir()]
    for stage_dir in stage_dirs:
        if not stage_dir.exists():
            continue
        for gt_path in stage_dir.rglob("groundtruth.json"):
            if gt_path.parent.name.lower() in {"logs", "video"}:
                continue
            yield gt_path.parent
        for misspelled in ("groudtruth.json", "groungtruth.json"):
            for gt_path in stage_dir.rglob(misspelled):
                if gt_path.parent.name.lower() in {"logs", "video"}:
                    continue
                yield gt_path.parent


def run_benchmark(root: Path, stages: Optional[List[str]] = None) -> Dict[str, Any]:
    risk_dir = REPO_ROOT / "3-RiskHunter"
    sys.path.insert(0, str(risk_dir))
    try:
        log_first = _load_module("nas_log_first_detector", risk_dir / "log_first_detector.py")
        run_e2e = _load_module("nas_run_e2e", REPO_ROOT / "run_e2e.py")
    finally:
        if str(risk_dir) in sys.path:
            sys.path.remove(str(risk_dir))

    result = BenchmarkSummary()
    seen_cases = set()
    for case_dir in sorted(_case_dirs(root, stages)):
        if case_dir in seen_cases:
            continue
        seen_cases.add(case_dir)

        gt_path = _choose_groundtruth(case_dir)
        case_id = str(case_dir.relative_to(root))
        if not gt_path:
            result.skipped.append({"case": case_id, "reason": "missing_groundtruth_or_logs"})
            continue
        try:
            groundtruth = _read_json_lenient(gt_path)
            logs, log_source_name = _load_case_logs(case_dir, log_first)
            if not logs:
                raise ValueError("missing or empty log file")
        except Exception as exc:
            result.skipped.append({"case": case_id, "reason": f"parse_error: {exc}"})
            continue

        sensitive_files = _sensitive_files_from_groundtruth(groundtruth)
        for path in _sensitive_files_from_logs(logs, log_first):
            if path.lower() not in {item.lower() for item in sensitive_files}:
                sensitive_files.append(path)

        detector = log_first.LogFirstDetector(
            sensitive_files=sensitive_files,
            blacklist_apps=DEFAULT_BLACKLIST_APPS,
            whitelist_apps=DEFAULT_WHITELIST_APPS,
        )
        detection = detector.analyze(logs)
        should_run_vlm, fallback_meta = run_e2e._should_use_vlm_fallback(logs, detection)

        expected = _expected_positive(groundtruth)
        deterministic_positive = len(detection.get("upload_events", [])) > 0
        triage_positive = deterministic_positive or should_run_vlm
        triage_bucket = result.triage.add(expected, triage_positive)
        deterministic_bucket = result.deterministic.add(expected, deterministic_positive)
        if deterministic_positive:
            result.deterministic_hits += 1
        if should_run_vlm:
            result.vlm_reviews += 1

        result.cases.append(
            {
                "case": case_id,
                "log_file": log_source_name,
                "expected_positive": expected,
                "triage_positive": triage_positive,
                "deterministic_positive": deterministic_positive,
                "triage_bucket": triage_bucket,
                "deterministic_bucket": deterministic_bucket,
                "upload_events": len(detection.get("upload_events", [])),
                "operation_records": len(detection.get("operation_records", [])),
                "sensitive_files": len(sensitive_files),
                "vlm_decision": fallback_meta.get("decision"),
                "vlm_reasons": fallback_meta.get("reasons", []),
            }
        )

    return result.to_dict()


def _print_report(report: Dict[str, Any]) -> None:
    print("\nNAS Sample Benchmark")
    print("=" * 40)
    for name in ("triage", "deterministic"):
        metrics = report["summary"][name]
        print(
            f"{name:14} precision={metrics['precision']:.4f} "
            f"recall={metrics['recall']:.4f} f1={metrics['f1']:.4f} "
            f"(tp={metrics['tp']}, fp={metrics['fp']}, tn={metrics['tn']}, fn={metrics['fn']})"
        )
    print(
        f"deterministic_hits={report['summary']['deterministic_hits']} "
        f"vlm_reviews={report['summary']['vlm_reviews']} "
        f"skipped={report['summary']['skipped_cases']}"
    )
    failures = [case for case in report["cases"] if case["triage_bucket"] in {"fp", "fn"}]
    print(f"triage_failures={len(failures)}")
    for case in failures[:30]:
        print(
            f"- {case['case']} {case['triage_bucket']} "
            f"det={case['upload_events']} vlm={case['vlm_decision']} "
            f"reasons={','.join(case['vlm_reasons'])}"
        )


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Benchmark downloaded NAS samples without live VLM calls.")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--stage", action="append", help="Limit to a stage directory, e.g. --stage stage1")
    parser.add_argument("--json", action="store_true", help="Print JSON instead of a human report.")
    parser.add_argument("--json-output", type=Path, help="Write full JSON report to this path.")
    args = parser.parse_args(argv)

    report = run_benchmark(args.data_root, args.stage)
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        _print_report(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
