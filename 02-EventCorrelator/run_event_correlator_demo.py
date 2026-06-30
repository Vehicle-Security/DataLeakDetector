import argparse
import ctypes
import json
import os
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_ROOT = Path(__file__).resolve().parent

if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))


from event_correlator import EventCorrelator  # noqa: E402


def build_demo_segments() -> list[dict]:
    return [
        {
            "segment_id": "seg_split",
            "time_range": "2026-03-27 12:30:33 - 2026-03-27 12:30:48",
            "app_name": "Command Prompt",
            "operation_type": "split_export",
            "primary_resource": "employee_salary_q4.xlsx",
            "related_resources": [
                "employee_salary_q4_part1.xlsx",
                "employee_salary_q4_part2.xlsx",
            ],
            "action_description": "User split the original salary file into two derived Excel files.",
            "visible_evidence": [
                "employee_salary_q4.xlsx",
                "employee_salary_q4_part1.xlsx",
                "employee_salary_q4_part2.xlsx",
            ],
            "supporting_timestamps": ["2026-03-27 12:30:42"],
            "confidence": 0.92,
        },
        {
            "segment_id": "seg_mail_upload",
            "time_range": "2026-03-27 12:31:46 - 2026-03-27 12:32:17",
            "app_name": "QQMail",
            "operation_type": "mail_attachment_upload",
            "primary_resource": "employee_salary_q4_part1.xlsx",
            "related_resources": ["employee_salary_q4_part2.xlsx"],
            "action_description": "User attached and sent two split salary files in QQMail.",
            "visible_evidence": [
                "QQMail",
                "employee_salary_q4_part1.xlsx",
                "employee_salary_q4_part2.xlsx",
                "send_button",
            ],
            "supporting_timestamps": ["2026-03-27 12:31:46", "2026-03-27 12:31:48"],
            "confidence": 0.96,
        },
    ]


def get_windows_desktop() -> Path:
    if os.name != "nt":
        return Path.home() / "Desktop"

    buffer = ctypes.create_unicode_buffer(260)
    result = ctypes.windll.shell32.SHGetFolderPathW(None, 0x0010, None, 0, buffer)
    if result == 0 and buffer.value:
        return Path(buffer.value)

    return Path.home() / "Desktop"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run EventCorrelator standalone sample analysis.")
    parser.add_argument("--sample-root", type=str, default="", help="Absolute path to sample root")
    parser.add_argument(
        "--output",
        type=str,
        default=str(REPO_ROOT / "spec" / "output" / "event_correlator_sample_10-2.json"),
        help="Output JSON path",
    )
    args = parser.parse_args()

    sample_root = Path(args.sample_root) if args.sample_root else get_windows_desktop() / "10-2"
    log_path = sample_root / "logs" / "keyevents.json"
    output_path = Path(args.output)

    with log_path.open("r", encoding="utf-8") as fh:
        log_events = json.load(fh)

    payload = {
        "session_id": "10-2",
        "record_id": "10-2",
        "recording_start_time": "2026-03-27 12:29:29",
        "log_events": log_events,
        "frame_segments": build_demo_segments(),
        "sensitive_files": [
            "C:/acceptance_samples/employee_salary_q4.xlsx",
        ],
        "session_metadata": {
            "sample_name": "10-2",
            "source": "standalone_sample",
        },
    }

    bundle = EventCorrelator().run(payload)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(bundle, fh, ensure_ascii=False, indent=2)

    print(f"saved: {output_path}")
    print(f"status: {bundle['analysis_status']}")
    print(f"correlated_events: {len(bundle['correlated_events'])}")
    print(f"upload_candidates: {len(bundle['upload_candidates'])}")
    print(f"direct_mappings: {len(bundle['file_lineage']['direct_file_mappings'])}")


if __name__ == "__main__":
    main()
