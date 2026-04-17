from __future__ import annotations

from typing import Any

from .python_datalog_engine import PythonDatalogEngine


def _normalize_process_name(app_name: str) -> str:
    app = str(app_name or "").strip()
    if not app:
        return "unknown"
    lowered = app.lower()
    if lowered.endswith(".exe"):
        return lowered
    if "qq邮箱" in lowered or "qqmail" in lowered or lowered == "qq":
        return "msedge.exe"
    return lowered


def _timestamp_to_int(timestamp_text: str) -> int:
    digits = "".join(ch for ch in str(timestamp_text or "") if ch.isdigit())
    if not digits:
        return 0
    return int(digits[:13]) if len(digits) >= 13 else int(digits)


def _infer_leak_channel_from_sink_type(sink_type: str) -> str:
    normalized = str(sink_type or "").strip().lower()
    if normalized in {
        "mail_attachment",
        "screen_share",
        "chat_upload",
        "cloud_sync",
        "web_post",
    }:
        return normalized
    return "unknown"


def _infer_leak_channel_from_correlated_event(event_type: str, operation_type: str, app_name: str) -> str:
    combined = " ".join([event_type, operation_type, app_name]).lower()
    if any(marker in combined for marker in ("mail", "email", "attachment", "qq邮箱", "邮件", "附件")):
        return "mail_attachment"
    if any(marker in combined for marker in ("screen", "share", "meeting", "共享", "屏幕", "会议")):
        return "screen_share"
    if any(marker in combined for marker in ("chat", "qq", "wechat", "聊天", "微信")):
        return "chat_upload"
    if any(marker in combined for marker in ("cloud", "drive", "sync", "云盘")):
        return "cloud_sync"
    if "upload" in combined:
        return "web_post"
    return "unknown"


class LeakDatalogBridge:
    def run(self, facts: list[dict[str, Any]]) -> list[dict[str, Any]]:
        engine = PythonDatalogEngine()

        open_counter = 0
        transfer_counter = 0
        cross_counter = 0
        leak_counter = 0

        seen_open = set()
        seen_transfer = set()
        seen_cross = set()
        seen_leak = set()

        for fact in facts:
            timestamp = _timestamp_to_int(fact.get("timestamp", ""))
            fact_type = str(fact.get("fact_type", "") or "").strip()

            if fact_type == "correlated_event":
                original_file = str(fact.get("original_file", "") or "").strip()
                current_file = str(fact.get("current_file", "") or "").strip()
                app_name = str(fact.get("app_name", "") or "").strip()
                event_type = str(fact.get("event_type", "") or "").strip().lower()
                operation_type = str(fact.get("operation_type", "") or "").strip().lower()
                process_name = _normalize_process_name(app_name)
                leak_channel = _infer_leak_channel_from_correlated_event(event_type, operation_type, app_name)

                if original_file:
                    open_key = (process_name, original_file)
                    if open_key not in seen_open:
                        open_counter += 1
                        engine.add_fact("OpenFile", f"open_{open_counter}", process_name, original_file, timestamp)
                        seen_open.add(open_key)

                if original_file and current_file and original_file != current_file:
                    transfer_key = (process_name, original_file, current_file)
                    if transfer_key not in seen_transfer:
                        transfer_counter += 1
                        engine.add_fact(
                            "TransferFile",
                            f"transfer_{transfer_counter}",
                            process_name,
                            original_file,
                            current_file,
                            timestamp,
                        )
                        seen_transfer.add(transfer_key)

                if leak_channel != "unknown" and current_file:
                    leak_key = (process_name, current_file, leak_channel)
                    if leak_key not in seen_leak:
                        leak_counter += 1
                        engine.add_fact(
                            "LeakFile",
                            f"leak_{leak_counter}",
                            process_name,
                            current_file,
                            leak_channel,
                            timestamp,
                        )
                        seen_leak.add(leak_key)

            elif fact_type == "upload_candidate":
                original_file = str(fact.get("original_file", "") or "").strip()
                app_name = str(fact.get("app_name", "") or "").strip()
                process_name = _normalize_process_name(app_name)
                current_files = list(fact.get("current_files", []) or [])
                leak_channel = _infer_leak_channel_from_sink_type(str(fact.get("sink_type", "") or ""))

                if original_file:
                    open_key = (process_name, original_file)
                    if open_key not in seen_open:
                        open_counter += 1
                        engine.add_fact("OpenFile", f"open_{open_counter}", process_name, original_file, timestamp)
                        seen_open.add(open_key)

                for current_file in current_files:
                    current_path = str(current_file or "").strip()
                    if not current_path:
                        continue

                    if original_file and current_path != original_file:
                        transfer_key = (process_name, original_file, current_path)
                        if transfer_key not in seen_transfer:
                            transfer_counter += 1
                            engine.add_fact(
                                "TransferFile",
                                f"transfer_{transfer_counter}",
                                process_name,
                                original_file,
                                current_path,
                                timestamp,
                            )
                            seen_transfer.add(transfer_key)

                        cross_key = ("source.exe", process_name, current_path)
                        if cross_key not in seen_cross:
                            cross_counter += 1
                            engine.add_fact(
                                "CrossProcessTransfer",
                                f"cross_{cross_counter}",
                                "source.exe",
                                process_name,
                                current_path,
                                timestamp,
                            )
                            seen_cross.add(cross_key)

                    if leak_channel == "unknown":
                        continue

                    leak_key = (process_name, current_path, leak_channel)
                    if leak_key not in seen_leak:
                        leak_counter += 1
                        engine.add_fact(
                            "LeakFile",
                            f"leak_{leak_counter}",
                            process_name,
                            current_path,
                            leak_channel,
                            timestamp,
                        )
                        seen_leak.add(leak_key)

        leak_paths = engine.run_inference()
        return [path.to_dict() for path in leak_paths]
