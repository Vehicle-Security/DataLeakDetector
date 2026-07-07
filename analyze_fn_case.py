#!/usr/bin/env python3
"""Analyze a False Negative case to understand why Datalog didn't infer leak path."""

import json
import sys

report_file = "d:/Projects/Job/DataLeakDetector/spec/output/nas_vlm_old_api_test_20260707_024655/report.json"

with open(report_file, "r", encoding="utf-8") as f:
    data = json.load(f)

cases = data.get("cases", [])

# Find a FN case
fn_case = None
for case in cases:
    if case.get("expected") == 1 and case.get("final_positive") == 0:
        fn_case = case
        break

if not fn_case:
    print("No FN case found")
    sys.exit(1)

print(f"Case ID: {fn_case['case_id']}")
print(f"Expected: {fn_case['expected']}")
print(f"Final Positive: {fn_case['final_positive']}")
print(f"Expected Level: {fn_case.get('expected_level')}")
print()

# Check VLM verdict
vlm_verdict = fn_case.get("live_vlm_verdict", {})
print(f"VLM Status: {vlm_verdict.get('status')}")
print(f"VLM Risk Level: {vlm_verdict.get('risk_level')}")
print(f"VLM Is Violation: {vlm_verdict.get('is_violation')}")
print()

# Check datalog decision
evaluation = fn_case.get("live_evaluation", {})
datalog = evaluation.get("datalog_decision", {})
print(f"Datalog Risk Positive: {evaluation.get('risk_positive')}")
print(f"Datalog Confirmed Leak: {evaluation.get('confirmed_leak')}")
print(f"Datalog Leak Paths: {len(datalog.get('leak_paths', []))}")
print(f"Datalog Risk Support: {len(datalog.get('risk_support', []))}")
print(f"Datalog Reason: {datalog.get('reason')}")
print()

# Check audit actions
audit_actions = evaluation.get("audit_actions", [])
print(f"Total Audit Actions: {len(audit_actions)}")

# Show VLM actions
vlm_actions = [a for a in audit_actions if a.get("evidence_source") == "remote_vlm"]
print(f"VLM Actions: {len(vlm_actions)}")
for i, action in enumerate(vlm_actions[:3], 1):
    print(f"  [{i}] Type: {action.get('action_type')}, Risk: {action.get('risk_level')}, Source: {action.get('source_file', 'N/A')[:50]}")
print()

# Show datalog facts
facts = datalog.get("facts", [])
print(f"Datalog Facts: {len(facts)}")
relation_counts = datalog.get("relation_counts", {})
for rel, count in sorted(relation_counts.items()):
    print(f"  {rel}: {count}")
