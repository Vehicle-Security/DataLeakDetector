"""Use a text LLM to audit disputed Release cases without changing source labels."""

from __future__ import annotations

import argparse
import json
import os
import re
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


DECISIONS = {"accept_detector", "accept_groundtruth", "unscorable_data_mismatch", "insufficient_evidence"}


def main() -> int:
    parser = argparse.ArgumentParser(description="Adjudicate scored Release disagreements using an OpenAI-compatible text LLM.")
    parser.add_argument("--release-report", required=True, help="Completed release_report.json.")
    parser.add_argument("--case-debug-root", required=True, help="Release case_debug directory containing VLM debug artifacts.")
    parser.add_argument("--output", required=True, help="Output adjudication JSON path.")
    parser.add_argument("--base-url", default=os.getenv("DLD_JUDGE_BASE_URL", "https://api.deepseek.com"))
    parser.add_argument("--model", default=os.getenv("DLD_JUDGE_MODEL", "deepseek-v4-pro"))
    parser.add_argument("--api-key-env", default="DLD_JUDGE_API_KEY", help="Environment variable containing the API key.")
    parser.add_argument("--max-cases", type=int, default=0, help="Optional cap; 0 adjudicates every scored disagreement.")
    args = parser.parse_args()

    api_key = os.getenv(args.api_key_env, "").strip()
    if not api_key:
        parser.error(f"missing API key environment variable: {args.api_key_env}")

    report_path = Path(args.release_report)
    report = _read_json(report_path)
    cases = list(report.get("cases", []))
    disputed = [case for case in cases if _is_scored_disagreement(case)]
    if args.max_cases > 0:
        disputed = disputed[: args.max_cases]

    results = []
    for index, case in enumerate(disputed, start=1):
        evidence = _case_evidence(case, Path(args.case_debug_root))
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
        results.append(result)
        print(f"[{index}/{len(disputed)}] {result['case']}: {result['decision']}", flush=True)

    output = {
        "schema_version": 1,
        "source_release_report": str(report_path),
        "model": args.model,
        "base_url": args.base_url.rstrip("/"),
        "disputed_cases": len(disputed),
        "summary": _summary(results),
        "cases": results,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


def _is_scored_disagreement(case: dict[str, Any]) -> bool:
    evaluation = dict(case.get("evaluation", {}))
    return evaluation.get("score_status") == "scored" and evaluation.get("detector_correct") is False


def _case_evidence(case: dict[str, Any], debug_root: Path) -> dict[str, Any]:
    evaluation = dict(case.get("evaluation", {}))
    case_id = str(case.get("case_relative_path") or case.get("case") or case.get("case_id") or "")
    artifacts = _find_debug_artifacts(debug_root / Path(case_id))
    groundtruth = _read_json(artifacts["groundtruth"]) if artifacts.get("groundtruth") else {}
    parsed = _read_json(artifacts["parsed"]) if artifacts.get("parsed") else {}
    correlator = _read_json(artifacts["correlator"]) if artifacts.get("correlator") else {}
    return {
        "case": case_id,
        "expected_conclusion": str(evaluation.get("expected_conclusion") or ""),
        "detector_conclusion": str(evaluation.get("detector_conclusion") or case.get("conclusion") or ""),
        "groundtruth_operations": list(groundtruth.get("operations", [])),
        "vlm_events": list(parsed.get("events", [])),
        "correlated_events": list(correlator.get("correlated_events", [])),
        "upload_candidates": list(correlator.get("upload_candidates", [])),
        "evidence_refs": artifacts,
    }


def _find_debug_artifacts(case_root: Path) -> dict[str, str]:
    names = {
        "groundtruth": "groundtruth.json",
        "parsed": "vlm_parse_result.json",
        "correlator": "event_correlator_details.json",
    }
    result: dict[str, str] = {}
    for key, name in names.items():
        candidates = sorted(case_root.rglob(name)) if case_root.exists() else []
        if candidates:
            result[key] = str(candidates[-1])
    return result


def _request_decision(*, base_url: str, api_key: str, model: str, evidence: dict[str, Any]) -> dict[str, str]:
    prompt = (
        "You are an evidence adjudicator for a data-leak benchmark. Return JSON only: "
        '{"decision":"accept_detector|accept_groundtruth|unscorable_data_mismatch|insufficient_evidence","reason":"..."}.\n'
        "Rules:\n"
        "- accept_detector only when structured VLM evidence directly supports the detector conclusion and the groundtruth conflicts with that evidence.\n"
        "- accept_groundtruth when the detector lacks evidence or the evidence supports the groundtruth.\n"
        "- unscorable_data_mismatch when groundtruth and visual evidence concern clearly different files/sessions without lineage.\n"
        "- insufficient_evidence when neither side can be justified.\n"
        "- A filename differing only by whitespace immediately before its extension is the same file. "
        "Different document names are not the same file without explicit lineage.\n"
        "Evidence:\n"
        + json.dumps(evidence, ensure_ascii=False)
    )
    payload = {
        "model": model,
        "temperature": 0,
        "messages": [
            {"role": "system", "content": "Be conservative. Do not invent missing visual or lineage evidence."},
            {"role": "user", "content": prompt},
        ],
    }
    request = urllib.request.Request(
        base_url.rstrip("/") + "/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            body = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:500]
        raise RuntimeError(f"judge_http_{exc.code}: {detail}") from exc
    content = str(body["choices"][0]["message"].get("content") or "")
    parsed = _extract_json(content)
    decision = str(parsed.get("decision") or "")
    if decision not in DECISIONS:
        raise RuntimeError(f"judge_invalid_decision: {decision!r}")
    return {"decision": decision, "reason": str(parsed.get("reason") or "")}


def _extract_json(text: str) -> dict[str, Any]:
    match = re.search(r"```(?:json)?\s*(.*?)```", text.strip(), flags=re.IGNORECASE | re.DOTALL)
    return json.loads(match.group(1) if match else text)


def _effective_conclusion(evidence: dict[str, Any], decision: str) -> str:
    if decision == "accept_detector":
        return str(evidence["detector_conclusion"])
    if decision == "accept_groundtruth":
        return str(evidence["expected_conclusion"])
    return ""


def _summary(results: list[dict[str, Any]]) -> dict[str, int | float | None]:
    summary: dict[str, int | float | None] = {
        decision: sum(item["decision"] == decision for item in results) for decision in sorted(DECISIONS)
    }
    scorable = [item for item in results if item["effective_correct"] is not None]
    correct = sum(item["effective_correct"] is True for item in scorable)
    summary["effective_scored_cases"] = len(scorable)
    summary["effective_correct_cases"] = correct
    summary["effective_accuracy"] = round(correct / len(scorable), 6) if scorable else None
    return summary


def _read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


if __name__ == "__main__":
    raise SystemExit(main())
