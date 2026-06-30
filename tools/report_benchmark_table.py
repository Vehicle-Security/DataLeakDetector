"""Render NAS benchmark JSON as a paper-style category table."""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any


CATEGORY_LABELS = {
    "email": "Email",
    "ai": "AI Chat",
    "messaging": "IM",
    "drive": "Cloud Drive",
    "meeting": "Meeting",
    "workplace": "Collaboration",
    "community": "Technical Forum",
    "git": "Code Hosting",
    "transfer": "Transfer",
    "transport": "Transfer",
    "filestruct": "File Structure",
    "content": "Content",
    "contentchange": "Content Transform",
    "screen": "Screen",
    "vmware": "Virtual Machine",
    "bluetooth": "Bluetooth",
    "stegoimage": "Steganography",
    "annotation": "Annotation",
}


def _case_parts(case_id: str) -> list[str]:
    name = case_id.replace("\\", "/").split("/")[-1].lower()
    return [part for part in re.split(r"[-_\s]+", name) if part]


def infer_category(case_id: str) -> str:
    parts = _case_parts(case_id)
    if len(parts) >= 3 and parts[1] == "normal":
        token = parts[2]
    elif len(parts) >= 2:
        token = parts[1]
    elif parts:
        token = parts[0]
    else:
        token = "unknown"
    return CATEGORY_LABELS.get(token, token.title())


def infer_app(case_id: str) -> str:
    parts = _case_parts(case_id)
    if len(parts) >= 4 and parts[1] == "normal":
        return _clean_app(parts[3:])
    if len(parts) >= 3:
        return _clean_app(parts[2:])
    return "unknown"


def _clean_app(parts: list[str]) -> str:
    tokens = [part for part in parts if not part.isdigit()]
    if not tokens:
        return "unknown"
    return " ".join(tokens)


def is_holdout_case(case_id: str) -> bool:
    parts = _case_parts(case_id)
    return len(parts) >= 2 and parts[1] == "normal"


def _metrics(rows: list[dict[str, Any]]) -> dict[str, float | int]:
    tp = sum(1 for row in rows if row.get("final_bucket") == "tp")
    fp = sum(1 for row in rows if row.get("final_bucket") == "fp")
    tn = sum(1 for row in rows if row.get("final_bucket") == "tn")
    fn = sum(1 for row in rows if row.get("final_bucket") == "fn")
    precision = tp / (tp + fp) if tp + fp else 1.0
    recall = tp / (tp + fn) if tp + fn else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "cases": len(rows),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision": precision * 100,
        "recall": recall * 100,
        "f1": f1 * 100,
    }


def build_table(report: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    apps: dict[str, set[str]] = defaultdict(set)
    held: dict[str, set[str]] = defaultdict(set)

    for row in report.get("cases", []):
        case_id = str(row.get("case", ""))
        category = infer_category(case_id)
        app = infer_app(case_id)
        grouped[category].append(row)
        apps[category].add(app)
        if is_holdout_case(case_id):
            held[category].add(app)

    table_rows = []
    for category in sorted(grouped):
        metrics = _metrics(grouped[category])
        table_rows.append(
            {
                "Category": category,
                "#Case": metrics["cases"],
                "#Apps": len(apps[category]),
                "#Held": len(held[category]),
                "Prec(%)": round(float(metrics["precision"]), 1),
                "Recall(%)": round(float(metrics["recall"]), 1),
                "F1(%)": round(float(metrics["f1"]), 1),
                "TP": metrics["tp"],
                "FP": metrics["fp"],
                "TN": metrics["tn"],
                "FN": metrics["fn"],
            }
        )

    overall_metrics = _metrics(report.get("cases", []))
    all_apps = {infer_app(str(row.get("case", ""))) for row in report.get("cases", [])}
    all_held = {
        infer_app(str(row.get("case", "")))
        for row in report.get("cases", [])
        if is_holdout_case(str(row.get("case", "")))
    }
    overall = {
        "Category": "Overall",
        "#Case": overall_metrics["cases"],
        "#Apps": len(all_apps),
        "#Held": len(all_held),
        "Prec(%)": round(float(overall_metrics["precision"]), 1),
        "Recall(%)": round(float(overall_metrics["recall"]), 1),
        "F1(%)": round(float(overall_metrics["f1"]), 1),
        "TP": overall_metrics["tp"],
        "FP": overall_metrics["fp"],
        "TN": overall_metrics["tn"],
        "FN": overall_metrics["fn"],
    }
    return table_rows, overall


def render_markdown(table_rows: list[dict[str, Any]], overall: dict[str, Any]) -> str:
    headers = ["Category", "#Case", "#Apps", "#Held", "Prec(%)", "Recall(%)", "F1(%)"]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in table_rows + [overall]:
        lines.append("| " + " | ".join(str(row[header]) for header in headers) + " |")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Render benchmark JSON as a category table.")
    parser.add_argument("report", type=Path)
    parser.add_argument("--markdown-output", type=Path)
    parser.add_argument("--json-output", type=Path)
    args = parser.parse_args()

    report = json.loads(args.report.read_text(encoding="utf-8"))
    table_rows, overall = build_table(report)
    markdown = render_markdown(table_rows, overall)

    if args.markdown_output:
        args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_output.write_text(markdown, encoding="utf-8")
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(
            json.dumps({"rows": table_rows, "overall": overall}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    print(markdown, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
