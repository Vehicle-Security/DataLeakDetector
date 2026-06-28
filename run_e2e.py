"""
DataLeakDetector Full E2E Pipeline
完整流程: 日志 + 视频 → 模块3(RiskHunter) [内部调用 模块2(FileTracker) → 模块1(FrameAnalyzer)] → 模块4(ThreatDetector) → 证据报告

Pipeline 调用链:
  模块3 (RiskHunter): LangGraph 上传检测 Agent
      ↓ 内部调用
  模块2 (FileTracker): BehaviorAnalysisGraph 隐藏行为分析
      ↓ 内部调用
  模块1 (FrameAnalyzer): 视频帧分析
      ↓ 返回
  模块3: 判断黑白名单，生成报警
      ↓
  模块4 (ThreatDetector): Datalog推理，完整证据链
"""

import os
import sys
import json
import argparse
import logging
import re
from datetime import datetime
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

# Windows 控制台 UTF-8 输出
if sys.platform == 'win32' and not getattr(sys, "_dld_utf8_stdio_wrapped", False):
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
    sys._dld_utf8_stdio_wrapped = True

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("E2E-Pipeline")

# 添加模块路径
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(PROJECT_DIR, '1-FrameAnalyzer'))
sys.path.insert(0, os.path.join(PROJECT_DIR, '2-FileTracker'))
sys.path.insert(0, os.path.join(PROJECT_DIR, '3-RiskHunter'))
sys.path.insert(0, os.path.join(PROJECT_DIR, '4-ThreatDetector'))

try:
    from dotenv import load_dotenv
    # 先加载根目录 .env（模块1/2通过find_dotenv()也会找到它）
    load_dotenv(os.path.join(PROJECT_DIR, '.env'))
    # 再加载模块4的 .env，允许覆盖
    load_dotenv(os.path.join(PROJECT_DIR, '4-ThreatDetector', '.env'), override=True)
except ImportError:
    pass


# ==================== 模块导入 ====================

def import_modules():
    """延迟导入模块，避免启动时的依赖问题"""
    global create_upload_detector_graph, create_initial_state
    global UploadDetectionConfig
    global LogFirstDetector
    global DatalogEngine, PromptTemplates

    # 模块3 (RiskHunter) - LangGraph 上传检测 Agent
    # 模块3 内部会调用模块2 (FileTracker)，模块2 内部会调用模块1 (FrameAnalyzer)
    try:
        from upload_detector_graph import create_upload_detector_graph
        from upload_detector_state import create_initial_state
        from upload_detection_config import UploadDetectionConfig
        from log_first_detector import LogFirstDetector
        print("✅ 模块3 (RiskHunter + FileTracker + FrameAnalyzer) 加载成功")
    except ImportError as e:
        logger.warning(f"模块3导入失败: {e}")
        import traceback
        traceback.print_exc()
        create_upload_detector_graph = None
        create_initial_state = None
        UploadDetectionConfig = None
        LogFirstDetector = None

    # 模块4 (ThreatDetector) - Datalog 推理
    try:
        from datalog.datalog_engine import DatalogEngine
        from threat_prompts import PromptTemplates
        print("✅ 模块4 (ThreatDetector) 加载成功")
    except ImportError as e:
        logger.warning(f"模块4导入失败: {e}")
        DatalogEngine = None
        PromptTemplates = None


# ==================== 辅助函数 ====================

def load_logs(log_file: str) -> List[Dict]:
    """加载日志文件，支持 JSON Lines 和 JSON Array 格式"""
    with open(log_file, 'r', encoding='utf-8') as f:
        content = f.read().strip()

    # 尝试 JSON Array 格式
    if content.startswith('['):
        return json.loads(content)

    # JSON Lines 格式（每行一个 JSON 对象）
    logs = []
    for line in content.split('\n'):
        line = line.strip()
        if line:
            try:
                logs.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return logs


def get_recording_start_time(logs: List[Dict]) -> str:
    """从第一条日志获取录屏开始时间"""
    if logs:
        first_ts = logs[0].get('timestamp', '')
        return first_ts.replace('T', ' ').split('.')[0]
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def create_index_file(rec_start: str, output_dir: str) -> str:
    """创建临时 INDEX.md 文件"""
    index_content = f"**Recording Time**: {rec_start}\n"
    index_path = os.path.join(output_dir, "INDEX.md")
    os.makedirs(output_dir, exist_ok=True)
    with open(index_path, 'w') as f:
        f.write(index_content)
    return index_path


SENSITIVE_HINT_TOKENS = (
    "\u85aa\u8d44",
    "\u5de5\u8d44",
    "\u673a\u5bc6",
    "\u7edd\u5bc6",
    "\u5408\u540c",
    "\u8d22\u52a1",
    "\u5ba2\u6237",
    "\u5bc6\u7801",
    "\u6838\u5fc3",
    "\u79d8\u5bc6",
    "\u5185\u90e8",
    "\u62a5\u8868",
    "\u9884\u7b97",
    "\u6218\u7565",
    "\u89c4\u5212",
    "\u4f1a\u8bae\u7eaa\u8981",
    "\u5458\u5de5",
)

AI_CONTEXT_TOKENS = (
    "chatgpt",
    "chat.openai.com",
    "claude",
    "gemini",
    "deepseek",
    "kimi",
    "doubao",
    "tongyi",
    "yiyan",
    "yuanbao",
    "chatbox",
    "cherry studio",
    "cursor",
    "copilot",
    "llm",
    " ai ",
    "\u4eba\u5de5\u667a\u80fd",
    "\u5927\u6a21\u578b",
    "\u5bf9\u8bdd",
    "\u63d0\u793a\u8bcd",
)

AMBIGUOUS_EXFIL_TOKENS = (
    "clipboard",
    "paste",
    "copy",
    "send",
    "share",
    "upload",
    "attach",
    "attachment",
    "screenshot",
    "screen capture",
    "record",
    "recording",
    "\u7c98\u8d34",
    "\u590d\u5236",
    "\u53d1\u9001",
    "\u5206\u4eab",
    "\u4e0a\u4f20",
    "\u9644\u4ef6",
    "\u622a\u56fe",
    "\u5f55\u5c4f",
)


def _flatten_log_text(value: Any) -> str:
    if isinstance(value, dict):
        return " ".join(_flatten_log_text(item) for item in value.values())
    if isinstance(value, list):
        return " ".join(_flatten_log_text(item) for item in value)
    return str(value or "")


def _contains_any(text: str, tokens) -> bool:
    normalized = f" {text.lower()} "
    return any(token in normalized for token in tokens)


def _is_sensitive_hint_log(log: Dict[str, Any]) -> bool:
    upload_detection = log.get("upload_detection")
    if isinstance(upload_detection, dict) and upload_detection.get("sensitivity"):
        return True
    text = _flatten_log_text(log)
    return _contains_any(text, SENSITIVE_HINT_TOKENS)


def _should_use_vlm_fallback(
    logs: List[Dict[str, Any]],
    log_first_result: Dict[str, Any],
) -> tuple[bool, Dict[str, Any]]:
    """
    Decide whether spending VLM tokens is justified after log-first analysis.

    Deterministic log evidence always wins. VLM is reserved for cases where a
    sensitive file is present but logs only show ambiguous AI/chat/clipboard
    context that needs visual semantics.
    """
    meta = log_first_result.get("log_first", {}) if isinstance(log_first_result, dict) else {}
    sensitive_count = int(meta.get("sensitive_events", 0) or 0)
    decision = {
        "enabled": True,
        "used": False,
        "decision": "skip",
        "reasons": [],
        "candidate_events": [],
    }

    if sensitive_count <= 0:
        decision["reasons"].append("no_sensitive_log_context")
        return False, decision

    sensitive_times = [
        _parse_timestamp_ms(log.get("timestamp", ""))
        for log in logs
        if _is_sensitive_hint_log(log)
    ]
    sensitive_times = [item for item in sensitive_times if item > 0]
    window_seconds = int(os.getenv("DLD_VLM_FALLBACK_WINDOW_SEC", "300"))
    window_ms = max(window_seconds, 1) * 1000

    for log in logs:
        text = _flatten_log_text(log)
        has_ai_context = _contains_any(text, AI_CONTEXT_TOKENS)
        has_ambiguous_exfil = _contains_any(text, AMBIGUOUS_EXFIL_TOKENS)
        if not has_ai_context and not has_ambiguous_exfil:
            continue

        timestamp_ms = _parse_timestamp_ms(log.get("timestamp", ""))
        near_sensitive = (
            not sensitive_times
            or timestamp_ms <= 0
            or any(abs(timestamp_ms - sensitive_ms) <= window_ms for sensitive_ms in sensitive_times)
        )
        if not near_sensitive:
            continue

        reason = "ai_context_near_sensitive_log" if has_ai_context else "ambiguous_exfil_context_near_sensitive_log"
        decision["candidate_events"].append(
            {
                "timestamp": log.get("timestamp", ""),
                "event_type": log.get("event_type", ""),
                "app_name": log.get("app_name") or log.get("process_info", {}).get("process_name", ""),
                "reason": reason,
            }
        )

    if decision["candidate_events"]:
        decision["used"] = True
        decision["decision"] = "run"
        decision["reasons"] = sorted({item["reason"] for item in decision["candidate_events"]})
        return True, decision

    decision["reasons"].append("no_ai_or_ambiguous_exfil_context")
    return False, decision


# ==================== Datalog 数据结构 ====================

@dataclass
class DatalogFact:
    """Datalog 事实"""
    relation: str
    operation_id: str
    process: str
    file: str
    dst_file: str = None
    timestamp: str = None
    description: str = None
    from_process: str = None
    to_process: str = None
    shared_data: str = None

    def to_souffle_args(self):
        ts = self._parse_timestamp()
        if self.relation == "OpenFile":
            return (self.operation_id, self.process, self.file, ts)
        elif self.relation == "TransferFile":
            return (self.operation_id, self.process, self.file, self.dst_file, ts)
        elif self.relation == "CrossProcessTransfer":
            return (self.operation_id, self.from_process or self.process,
                    self.to_process or "Unknown", self.shared_data or self.file, ts)
        elif self.relation == "LeakFile":
            return (self.operation_id, self.process, self.file, "network", ts)
        return None

    def _parse_timestamp(self):
        if self.timestamp:
            try:
                dt = datetime.fromisoformat(self.timestamp.replace('Z', '+00:00'))
                return int(dt.timestamp() * 1000)
            except:
                pass
        return 0


# ==================== 模块3: RiskHunter (完整 LangGraph Pipeline) ====================

def run_module3_pipeline(log_file: str, video_path: str,
                         index_path: str) -> Dict[str, Any]:
    """
    运行模块3完整 LangGraph Pipeline

    模块3 (RiskHunter) 内部调用链:
      initialize_node → 创建 WorklistManager，扫描日志构建 worklist
      process_event_node → 调用模块2 (BehaviorAnalysisGraph)
                              → 模块2 调用模块1 (FrameAnalyzer) 分析视频帧
      analyze_upload_node → 判断黑白名单，生成报警
      循环处理所有 worklist 事件 → finalize_node

    Returns:
        包含模块3完整结果的字典
    """
    print("\n" + "=" * 80)
    print("🔍 模块3: RiskHunter - 完整 LangGraph Pipeline")
    print("   (内部自动调用 模块2 FileTracker → 模块1 FrameAnalyzer)")
    print("=" * 80)

    vlm_fallback_meta = None
    if create_upload_detector_graph is None:
        print("   ⚠️ 模块3未加载，跳过")
        return {
            "alert_events": [],
            "info_events": [],
            "upload_events": [],
            "operation_records": [],
            "statistics": {},
            "file_mappings": {},
            "vlm_fallback": vlm_fallback_meta,
        }

    # 使用模块3配置中的硬编码列表
    config = UploadDetectionConfig()
    sensitive_files = list(config.sensitive_files)
    blacklist_apps = config.blacklist_apps
    whitelist_apps = config.whitelist_apps

    # 从日志中提取额外的敏感文件（配置列表可能不包含测试数据中的文件）
    sensitive_keywords = ['薪资', '工资', '机密', '绝密', '合同', '财务',
                          '客户', '密码', '核心', '秘密', '内部', '报表',
                          '预算', '战略', '规划', '会议纪要', '员工']
    logs_data = load_logs(log_file)
    existing_normalized = set(f.replace('\\\\', '/').replace('\\', '/').lower() for f in sensitive_files)
    
    for log in logs_data:
        file_path = log.get('file_path', '')
        file_name = log.get('file_name', '')
        if file_path and any(kw in file_name for kw in sensitive_keywords):
            normalized = file_path.replace('\\', '/').lower()
            if normalized not in existing_normalized:
                sensitive_files.append(file_path)
                existing_normalized.add(normalized)
                print(f"   📌 从日志发现敏感文件: {file_name}")

    print(f"\n   📋 配置信息:")
    print(f"      - 敏感文件: {len(sensitive_files)} 个 (配置{len(config.sensitive_files)} + 日志提取{len(sensitive_files) - len(config.sensitive_files)})")
    print(f"      - 黑名单应用: {len(blacklist_apps)} 个")
    print(f"      - 白名单应用: {len(whitelist_apps)} 个")

    # 创建初始状态
    use_log_first = os.getenv("DLD_LOG_FIRST", "1").strip().lower() not in {"0", "false", "no", "off"}
    if use_log_first and LogFirstDetector is not None:
        print(f"\n   [log-first] deterministic log analysis enabled")
        log_first_result = LogFirstDetector(
            sensitive_files=sensitive_files,
            blacklist_apps=blacklist_apps,
            whitelist_apps=whitelist_apps,
        ).analyze(logs_data)

        upload_count = len(log_first_result.get("upload_events", []))
        log_first_meta = log_first_result.get("log_first", {})
        print(f"      - sensitive log events: {log_first_meta.get('sensitive_events', 0)}")
        print(f"      - file mappings: {log_first_meta.get('direct_mappings', 0)}")
        print(f"      - confident upload events: {upload_count}")

        if upload_count > 0:
            print("      [OK] log evidence is sufficient; skipping VLM analysis")
            return log_first_result

        enable_vlm_fallback = os.getenv("DLD_ENABLE_VLM_FALLBACK", "1").strip().lower() not in {"0", "false", "no", "off"}
        if not enable_vlm_fallback:
            print("      [INFO] no confident upload chain and VLM fallback is disabled")
            log_first_result["vlm_fallback"] = {
                "enabled": False,
                "used": False,
                "decision": "skip",
                "reasons": ["disabled_by_env"],
                "candidate_events": [],
            }
            return log_first_result

        should_run_vlm, vlm_fallback_meta = _should_use_vlm_fallback(logs_data, log_first_result)
        log_first_result["vlm_fallback"] = vlm_fallback_meta
        if not should_run_vlm:
            print(f"      [INFO] VLM fallback skipped: {', '.join(vlm_fallback_meta.get('reasons', []))}")
            return log_first_result

        print(f"      [INFO] VLM fallback needed: {', '.join(vlm_fallback_meta.get('reasons', []))}")

    initial_state = create_initial_state(
        record_id="e2e",
        base_path=os.path.dirname(log_file),
        log_file=log_file,
        video_path=video_path,
        index_path=index_path,
        sensitive_files=sensitive_files,
        blacklist_apps=blacklist_apps,
        whitelist_apps=whitelist_apps,
        # search_duration 使用模块3默认值 30
    )

    # 创建并运行 LangGraph
    print(f"\n   🚀 启动上传检测 Agent...")
    app = create_upload_detector_graph()
    graph_config = {"recursion_limit": 200}  # 动态派生事件较多时避免提前触发上限

    final_state = None
    for state in app.stream(initial_state, config=graph_config):
        final_state = state

    if final_state:
        # 提取实际的 state（从字典中获取最后一个节点的输出）
        if isinstance(final_state, dict):
            final_state = list(final_state.values())[-1]

    if not final_state:
        print("   ⚠️ 模块3未返回结果")
        return {
            "alert_events": [],
            "info_events": [],
            "upload_events": [],
            "operation_records": [],
            "statistics": {},
            "file_mappings": {},
            "vlm_fallback": vlm_fallback_meta,
        }

    # 提取结果
    upload_events = _dedupe_upload_events(final_state.get("upload_events", []))
    alert_events = [event for event in upload_events if getattr(event, "should_alert", False)]
    info_events = [event for event in upload_events if not getattr(event, "should_alert", False)]
    operation_records = final_state.get("operation_records", [])
    statistics = dict(final_state.get("statistics", {}))
    statistics["upload_events_detected"] = len(upload_events)
    statistics["blacklist_alerts"] = len(alert_events)
    statistics["whitelist_uploads"] = sum(1 for event in upload_events if getattr(event, "app_category", "") == "whitelist")
    statistics["unknown_uploads"] = sum(1 for event in upload_events if getattr(event, "app_category", "") == "unknown")

    # 提取文件映射关系
    file_mappings = {}
    manager = final_state.get("_worklist_manager")
    if manager and hasattr(manager, "export_file_mappings"):
        file_mappings = manager.export_file_mappings()

    print(f"\n   ✅ 模块3 Pipeline 完成")
    print(f"   📊 统计:")
    print(f"      - 已处理事件: {statistics.get('total_events_processed', 0)}")
    print(f"      - 检测到上传事件: {statistics.get('upload_events_detected', 0)}")
    print(f"      - 敏感操作记录(去重后): {len(operation_records)}")
    print(f"      - 黑名单报警: {statistics.get('blacklist_alerts', 0)}")
    print(f"      - 白名单上传: {statistics.get('whitelist_uploads', 0)}")
    print(f"      - 其他应用上传: {statistics.get('unknown_uploads', 0)}")

    if alert_events:
        print(f"\n   ⚠️ 报警事件 ({len(alert_events)} 个):")
        for i, event in enumerate(alert_events, 1):
            print(f"      [{i}] {event.alert_level.upper()}")
            print(f"         文件: {event.file_name}")
            print(f"         应用: {event.app_name} ({event.app_category})")
            print(f"         操作: {event.operation_type}")
            print(f"         外发内容: {event.upload_content}")
            print(f"         映射链: {event.upload_content_mapping_link}")
            print(f"         原因: {event.alert_reason}")

    if info_events:
        print(f"\n   ℹ️ 信息事件 ({len(info_events)} 个):")
        for i, event in enumerate(info_events, 1):
            print(f"      [{i}] {event.file_name} → {event.app_name}")

    return {
        "alert_events": alert_events,
        "info_events": info_events,
        "upload_events": upload_events,
        "operation_records": operation_records,
        "statistics": statistics,
        "file_mappings": file_mappings,
        "final_state": final_state,
        "vlm_fallback": vlm_fallback_meta,
    }


# ==================== 模块4: ThreatDetector ====================

def _parse_timestamp_ms(timestamp: str) -> int:
    if not timestamp:
        return 0
    try:
        dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00').replace(' ', 'T'))
        return int(dt.timestamp() * 1000)
    except Exception:
        return 0


def _normalize_process_name(process_name: str) -> str:
    normalized = str(process_name or "").strip().replace("\\", "/")
    if not normalized:
        return "unknown"
    return normalized.rsplit("/", 1)[-1].lower()


def _normalize_file_path(file_path: str) -> str:
    normalized = str(file_path or "").strip().replace("\\", "/")
    while "//" in normalized:
        normalized = normalized.replace("//", "/")
    return normalized


def _file_identity_key(file_path: str) -> str:
    return _normalize_file_path(file_path).lower()


def _basename_key(file_path: str) -> str:
    return _file_identity_key(file_path).rsplit("/", 1)[-1]


def _paths_match(left: str, right: str) -> bool:
    left_norm = _normalize_file_path(left)
    right_norm = _normalize_file_path(right)
    if not left_norm or not right_norm:
        return False
    if left_norm.lower() == right_norm.lower():
        return True
    if "/" in left_norm and "/" in right_norm:
        return False
    return _basename_key(left_norm) == _basename_key(right_norm)


def _is_sensitive_filename(file_name: str) -> bool:
    sensitive_keywords = [
        "薪资",
        "工资",
        "机密",
        "绝密",
        "合同",
        "财务",
        "客户",
        "密码",
        "核心",
        "秘密",
        "内部",
        "报表",
        "预算",
        "战略",
        "规划",
        "会议纪要",
        "员工",
    ]
    return any(keyword in (file_name or "") for keyword in sensitive_keywords)


def _dedupe_upload_events(events: List[Any]) -> List[Any]:
    deduped = []
    seen = set()
    for event in events or []:
        timestamp = event.timestamp if hasattr(event, "timestamp") else event.get("timestamp", "")
        if hasattr(event, "file_path"):
            file_path = getattr(event, "upload_content", "") or getattr(event, "file_path", "")
            app_name = getattr(event, "app_name", "")
            operation_type = getattr(event, "operation_type", "")
        else:
            file_path = event.get("upload_content") or event.get("file_path", "")
            app_name = event.get("app_name", "")
            operation_type = event.get("operation_type", "")
        bucket = _parse_timestamp_ms(timestamp) // 10000 if timestamp else 0
        key = (
            _file_identity_key(file_path),
            _normalize_process_name(app_name),
            str(operation_type or "").lower(),
            bucket,
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(event)
    return deduped


def _inject_connected_facts_from_module3(engine, module3_result: Dict[str, Any],
                                         logs: List[Dict]) -> List:
    supplementary_facts = []
    op_counter = [1000]
    logs_by_time = sorted(logs, key=lambda item: _parse_timestamp_ms(item.get("timestamp", "")))
    file_mappings = module3_result.get("file_mappings", {})
    direct_mappings = file_mappings.get("direct_file_mappings", {})

    existing_open = set()
    existing_transfer = set()
    existing_cross = set()
    existing_leak = set()

    for fact in engine.facts.get("OpenFile", []):
        _, process_name, file_path, timestamp_ms = fact.args
        existing_open.add((_normalize_process_name(process_name), _file_identity_key(file_path), timestamp_ms))
    for fact in engine.facts.get("TransferFile", []):
        _, process_name, src_path, dst_path, timestamp_ms = fact.args
        existing_transfer.add(
            (
                _normalize_process_name(process_name),
                _file_identity_key(src_path),
                _file_identity_key(dst_path),
                timestamp_ms,
            )
        )
    for fact in engine.facts.get("CrossProcessTransfer", []):
        _, from_process, to_process, shared_data, timestamp_ms = fact.args
        existing_cross.add(
            (
                _normalize_process_name(from_process),
                _normalize_process_name(to_process),
                _file_identity_key(shared_data),
                timestamp_ms,
            )
        )
    for fact in engine.facts.get("LeakFile", []):
        _, process_name, file_path, channel, timestamp_ms = fact.args
        existing_leak.add(
            (
                _normalize_process_name(process_name),
                _file_identity_key(file_path),
                str(channel or "network").lower(),
                timestamp_ms,
            )
        )

    def next_op_id(prefix="sup"):
        op_counter[0] += 1
        return f"{prefix}_{op_counter[0]}"

    def add_open_fact(process_name: str, file_path: str, timestamp_ms: int, timestamp_str: str, description: str):
        normalized_process = _normalize_process_name(process_name)
        normalized_file = _normalize_file_path(file_path)
        if not normalized_file:
            return False
        key = (normalized_process, _file_identity_key(normalized_file), timestamp_ms)
        if key in existing_open:
            return False

        op_id = next_op_id("open")
        engine.add_fact("OpenFile", op_id, normalized_process, normalized_file, timestamp_ms)
        existing_open.add(key)
        supplementary_facts.append(
            DatalogFact(
                relation="OpenFile",
                operation_id=op_id,
                process=normalized_process,
                file=normalized_file,
                timestamp=timestamp_str,
                description=description,
            )
        )
        return True

    def add_transfer_fact(process_name: str, src_path: str, dst_path: str, timestamp_ms: int, timestamp_str: str, description: str):
        normalized_process = _normalize_process_name(process_name)
        normalized_src = _normalize_file_path(src_path)
        normalized_dst = _normalize_file_path(dst_path)
        if not normalized_src or not normalized_dst:
            return False
        key = (
            normalized_process,
            _file_identity_key(normalized_src),
            _file_identity_key(normalized_dst),
            timestamp_ms,
        )
        if key in existing_transfer:
            return False

        op_id = next_op_id("transfer")
        engine.add_fact("TransferFile", op_id, normalized_process, normalized_src, normalized_dst, timestamp_ms)
        existing_transfer.add(key)
        supplementary_facts.append(
            DatalogFact(
                relation="TransferFile",
                operation_id=op_id,
                process=normalized_process,
                file=normalized_src,
                dst_file=normalized_dst,
                timestamp=timestamp_str,
                description=description,
            )
        )
        return True

    def add_cross_fact(from_process: str, to_process: str, shared_data: str, timestamp_ms: int, timestamp_str: str, description: str):
        normalized_from = _normalize_process_name(from_process)
        normalized_to = _normalize_process_name(to_process)
        normalized_data = _normalize_file_path(shared_data)
        if not normalized_data:
            return False
        key = (normalized_from, normalized_to, _file_identity_key(normalized_data), timestamp_ms)
        if key in existing_cross:
            return False

        op_id = next_op_id("cross")
        engine.add_fact("CrossProcessTransfer", op_id, normalized_from, normalized_to, normalized_data, timestamp_ms)
        existing_cross.add(key)
        supplementary_facts.append(
            DatalogFact(
                relation="CrossProcessTransfer",
                operation_id=op_id,
                process=normalized_from,
                from_process=normalized_from,
                to_process=normalized_to,
                shared_data=normalized_data,
                file=normalized_data,
                timestamp=timestamp_str,
                description=description,
            )
        )
        return True

    def add_leak_fact(process_name: str, file_path: str, channel: str, timestamp_ms: int, timestamp_str: str, description: str):
        normalized_process = _normalize_process_name(process_name)
        normalized_file = _normalize_file_path(file_path)
        normalized_channel = str(channel or "network").lower()
        if not normalized_file:
            return False
        key = (normalized_process, _file_identity_key(normalized_file), normalized_channel, timestamp_ms)
        if key in existing_leak:
            return False

        op_id = next_op_id("leak")
        engine.add_fact("LeakFile", op_id, normalized_process, normalized_file, normalized_channel, timestamp_ms)
        existing_leak.add(key)
        supplementary_facts.append(
            DatalogFact(
                relation="LeakFile",
                operation_id=op_id,
                process=normalized_process,
                file=normalized_file,
                timestamp=timestamp_str,
                description=description,
            )
        )
        return True

    def find_latest_log(file_path: str, event_types=None, timestamp_ms: Optional[int] = None):
        best_match = None
        best_ts = -1
        for log in logs_by_time:
            log_ts = _parse_timestamp_ms(log.get("timestamp", ""))
            if event_types and log.get("event_type", "") not in event_types:
                continue
            if timestamp_ms is not None and log_ts and log_ts > timestamp_ms:
                continue
            if _paths_match(log.get("file_path", ""), file_path) and log_ts >= best_ts:
                best_match = log
                best_ts = log_ts
        return best_match

    def find_owner_process(target_path: str, timestamp_ms: int, original_path: str = "") -> str:
        best_process = ""
        best_ts = -1
        for fact in engine.facts.get("TransferFile", []):
            _, process_name, _, dst_path, fact_ts = fact.args
            if fact_ts <= timestamp_ms and _paths_match(dst_path, target_path) and fact_ts >= best_ts:
                best_process = process_name
                best_ts = fact_ts
        if best_process:
            return _normalize_process_name(best_process)

        for fact in engine.facts.get("OpenFile", []):
            _, process_name, file_path, fact_ts = fact.args
            if fact_ts > timestamp_ms:
                continue
            if _paths_match(file_path, target_path) or (original_path and _paths_match(file_path, original_path)):
                if fact_ts >= best_ts:
                    best_process = process_name
                    best_ts = fact_ts
        return _normalize_process_name(best_process) if best_process else ""

    def infer_channel(window_title: str) -> str:
        lowered = (window_title or "").casefold()
        if "mail" in lowered or "qq" in lowered or "邮箱" in lowered:
            return "email"
        return "network"

    print("\n   📎 从模块3结果注入补充 Datalog 事实...")

    open_sources = []
    for log in logs_by_time:
        event_type = log.get("event_type", "")
        file_path = _normalize_file_path(log.get("file_path", ""))
        file_name = log.get("file_name", "")
        if event_type not in ["file_open", "opened"] or not file_path or not file_name:
            continue
        if not _is_sensitive_filename(file_name):
            continue

        timestamp_str = log.get("timestamp", "")
        timestamp_ms = _parse_timestamp_ms(timestamp_str)
        process_name = log.get("process_info", {}).get("process_name", "unknown")
        add_open_fact(
            process_name=process_name,
            file_path=file_path,
            timestamp_ms=timestamp_ms,
            timestamp_str=timestamp_str,
            description=f"补充: {process_name} 打开敏感文件 {file_name}",
        )
        print(f"      + OpenFile({_normalize_process_name(process_name)}, {os.path.basename(file_path)})")
        open_sources.append(
            {
                "base_name": os.path.splitext(file_name)[0],
                "file_path": file_path,
                "process_name": _normalize_process_name(process_name),
                "timestamp_ms": timestamp_ms,
                "timestamp_str": timestamp_str,
            }
        )

    for derived_file, original_file in direct_mappings.items():
        normalized_original = _normalize_file_path(original_file)
        normalized_derived = _normalize_file_path(derived_file)
        if not normalized_original or not normalized_derived:
            continue

        derived_log = find_latest_log(normalized_derived, event_types=["created", "modified", "file_upload"])
        timestamp_str = derived_log.get("timestamp", "") if derived_log else ""
        timestamp_ms = _parse_timestamp_ms(timestamp_str)
        owner_process = find_owner_process(normalized_derived, timestamp_ms or 0, normalized_original)
        if not owner_process:
            owner_log = find_latest_log(normalized_original, event_types=["file_open", "opened"], timestamp_ms=timestamp_ms or None)
            owner_process = _normalize_process_name(
                (owner_log or {}).get("process_info", {}).get("process_name", "unknown")
            )

        if add_open_fact(owner_process, normalized_original, timestamp_ms, timestamp_str, f"补充: 打开原始文件 {os.path.basename(normalized_original)}"):
            print(f"      + OpenFile({owner_process}, {os.path.basename(normalized_original)}) [映射]")
        if add_transfer_fact(owner_process, normalized_original, normalized_derived, timestamp_ms, timestamp_str, f"补充: 文件变换 {os.path.basename(normalized_original)} → {os.path.basename(normalized_derived)}"):
            print(f"      + TransferFile({owner_process}, {os.path.basename(normalized_original)} → {os.path.basename(normalized_derived)})")

    if not direct_mappings:
        print("      📌 模块3未发现文件映射，从日志推断文件派生关系...")
        inferred_derived_paths = set()
        for log in logs_by_time:
            event_type = log.get("event_type", "")
            file_path = _normalize_file_path(log.get("file_path", ""))
            file_name = log.get("file_name", "")
            if event_type not in ["created", "file_upload"] or not file_path or not file_name:
                continue
            derived_key = _file_identity_key(file_path)
            if derived_key in inferred_derived_paths:
                continue

            derived_base = os.path.splitext(file_name)[0]
            for source in open_sources:
                if derived_base == source["base_name"] or not derived_base.startswith(source["base_name"]):
                    continue

                timestamp_str = log.get("timestamp", "")
                timestamp_ms = _parse_timestamp_ms(timestamp_str)
                owner_process = find_owner_process(source["file_path"], timestamp_ms, source["file_path"]) or source["process_name"]
                add_open_fact(
                    owner_process,
                    source["file_path"],
                    source["timestamp_ms"],
                    source["timestamp_str"],
                    f"补充: {owner_process} 打开 {os.path.basename(source['file_path'])}",
                )
                if add_transfer_fact(
                    owner_process,
                    source["file_path"],
                    file_path,
                    timestamp_ms,
                    timestamp_str,
                    f"补充: {os.path.basename(source['file_path'])} → {os.path.basename(file_path)} (日志推断)",
                ):
                    print(f"      + TransferFile({owner_process}, {os.path.basename(source['file_path'])} → {os.path.basename(file_path)}) [日志推断]")
                inferred_derived_paths.add(derived_key)
                break

    for events_key in ["alert_events", "info_events"]:
        for event in module3_result.get(events_key, []):
            timestamp_str = event.timestamp if hasattr(event, "timestamp") else event.get("timestamp", "")
            timestamp_ms = _parse_timestamp_ms(timestamp_str)
            upload_content = event.upload_content if hasattr(event, "upload_content") else event.get("upload_content", "")
            file_path = event.file_path if hasattr(event, "file_path") else event.get("file_path", "")
            leak_file = _normalize_file_path(upload_content or file_path)
            if not leak_file:
                continue

            event_app = event.app_name if hasattr(event, "app_name") else event.get("app_name", "unknown")
            matched_log = find_latest_log(leak_file, event_types=["file_upload", "upload_detected", "created", "modified"], timestamp_ms=timestamp_ms)
            uploader_process = _normalize_process_name(
                (matched_log or {}).get("process_info", {}).get("process_name", event_app)
            )
            window_title = (matched_log or {}).get("window_info", {}).get("window_title", "")
            channel = infer_channel(window_title)

            if add_leak_fact(
                uploader_process,
                leak_file,
                channel,
                timestamp_ms,
                timestamp_str,
                f"补充: {uploader_process} 外发 {os.path.basename(leak_file)} ({event_app})",
            ):
                print(f"      + LeakFile({uploader_process}, {os.path.basename(leak_file)}) [{events_key}]")

            owner_process = find_owner_process(leak_file, timestamp_ms, _normalize_file_path(file_path))
            if owner_process and owner_process != uploader_process:
                if add_cross_fact(
                    owner_process,
                    uploader_process,
                    leak_file,
                    timestamp_ms,
                    timestamp_str,
                    f"补充: {owner_process} → {uploader_process} ({os.path.basename(leak_file)})",
                ):
                    print(f"      + CrossProcessTransfer({owner_process} → {uploader_process}) [{events_key}]")
            elif not owner_process:
                if add_open_fact(
                    uploader_process,
                    leak_file,
                    timestamp_ms,
                    timestamp_str,
                    f"补充: 将上传的敏感文件作为污染源 {os.path.basename(leak_file)}",
                ):
                    print(f"      + OpenFile({uploader_process}, {os.path.basename(leak_file)}) [upload source]")

    for log in logs_by_time:
        if log.get("event_type", "") != "file_upload":
            continue

        file_path = _normalize_file_path(log.get("file_path", ""))
        if not file_path:
            continue

        file_name = log.get("file_name", os.path.basename(file_path))
        timestamp_str = log.get("timestamp", "")
        timestamp_ms = _parse_timestamp_ms(timestamp_str)
        uploader_process = _normalize_process_name(log.get("process_info", {}).get("process_name", "unknown"))
        window_title = log.get("window_info", {}).get("window_title", "")
        channel = infer_channel(window_title)

        if add_leak_fact(
            uploader_process,
            file_path,
            channel,
            timestamp_ms,
            timestamp_str,
            f"补充: {uploader_process} 上传 {file_name} ({window_title})",
        ):
            print(f"      + LeakFile({uploader_process}, {file_name}) [日志file_upload, {window_title}]")

        owner_process = find_owner_process(file_path, timestamp_ms)
        if owner_process and owner_process != uploader_process:
            if add_cross_fact(
                owner_process,
                uploader_process,
                file_path,
                timestamp_ms,
                timestamp_str,
                f"补充: {owner_process} → {uploader_process} ({file_name})",
            ):
                print(f"      + CrossProcessTransfer({owner_process} → {uploader_process}) [日志推断]")

    print(f"      共注入 {len(supplementary_facts)} 条补充事实")
    return supplementary_facts


def _inject_facts_from_module3(engine, module3_result: Dict[str, Any],
                                logs: List[Dict]) -> List:
    """
    从模块3的实际检测结果注入补充 Datalog 事实
    
    解决的问题: LLM生成的Datalog事实经常因为进程名/文件名不一致而无法
    形成连通的污点链。我们用模块3已经确认的结果来补充关键事实。
    
    注入策略:
    1. OpenFile: 从日志中提取敏感文件的打开事件
    2. TransferFile: 从文件映射关系中提取文件变换/分片
    3. LeakFile: 从alert_events中提取确认的外发行为
    """
    supplementary_facts = []
    op_counter = [1000]  # 用列表包裹以便在内部函数中修改
    
    def next_op_id(prefix="sup"):
        op_counter[0] += 1
        return f"{prefix}_{op_counter[0]}"
    
    def parse_ts(ts_str):
        if not ts_str:
            return 0
        try:
            dt = datetime.fromisoformat(ts_str.replace('Z', '+00:00').replace(' ', 'T'))
            return int(dt.timestamp() * 1000)
        except:
            return 0
    
    print("\n   📎 从模块3结果注入补充 Datalog 事实...")
    
    # 1. 从日志中提取敏感文件的 OpenFile 事实
    #    这确保有初始污染源头
    sensitive_files_opened = set()
    for log in logs:
        event_type = log.get('event_type', '')
        file_path = log.get('file_path', '')
        file_name = log.get('file_name', '')
        process_name = log.get('process_info', {}).get('process_name', '')
        timestamp = log.get('timestamp', '')
        
        if event_type in ['file_open', 'opened'] and file_name:
            # 检查是否是敏感文件 (包含工资、机密等关键词)
            sensitive_keywords = ['薪资', '工资', '机密', '绝密', '合同', '财务',
                                  '客户', '密码', '核心', '秘密', '内部']
            is_sensitive = any(kw in file_name for kw in sensitive_keywords)
            if is_sensitive and file_path not in sensitive_files_opened:
                sensitive_files_opened.add(file_path)
                ts = parse_ts(timestamp)
                op_id = next_op_id("open")
                engine.add_fact("OpenFile", op_id, process_name, file_path, ts)
                supplementary_facts.append(
                    DatalogFact(relation="OpenFile", operation_id=op_id,
                                process=process_name, file=file_path,
                                timestamp=timestamp,
                                description=f"补充: {process_name} 打开敏感文件 {file_name}")
                )
                print(f"      + OpenFile({process_name}, {file_name})")
    
    # 2. 从文件映射关系注入 TransferFile 事实
    #    这确保隐藏行为(重命名/分片/格式转换)被追踪
    file_mappings = module3_result.get("file_mappings", {})
    direct_mappings = file_mappings.get("direct_file_mappings", {})
    
    for derived_file, original_file in direct_mappings.items():
        # 从日志中找到对应的进程和时间
        process_name = "unknown"
        timestamp = ""
        for log in logs:
            log_file = log.get('file_path', '').replace('\\', '/')
            if log_file == derived_file or log_file == original_file:
                process_name = log.get('process_info', {}).get('process_name', 'unknown')
                timestamp = log.get('timestamp', '')
                break
        
        ts = parse_ts(timestamp)
        
        # 确保 original 也有 OpenFile
        if original_file not in sensitive_files_opened:
            op_id = next_op_id("open")
            engine.add_fact("OpenFile", op_id, process_name, original_file, ts)
            sensitive_files_opened.add(original_file)
            supplementary_facts.append(
                DatalogFact(relation="OpenFile", operation_id=op_id,
                            process=process_name, file=original_file,
                            timestamp=timestamp,
                            description=f"补充: 打开原始文件 {os.path.basename(original_file)}")
            )
        
        # TransferFile: original → derived (同进程)
        op_id = next_op_id("transfer")
        engine.add_fact("TransferFile", op_id, process_name, original_file, derived_file, ts)
        supplementary_facts.append(
            DatalogFact(relation="TransferFile", operation_id=op_id,
                        process=process_name, file=original_file,
                        dst_file=derived_file, timestamp=timestamp,
                        description=f"补充: 文件变换 {os.path.basename(original_file)} → {os.path.basename(derived_file)}")
        )
        print(f"      + TransferFile({process_name}, {os.path.basename(original_file)} → {os.path.basename(derived_file)})")
        
        # 确保 derived 也有 OpenFile (用于后续传播)
        if derived_file not in sensitive_files_opened:
            op_id2 = next_op_id("open")
            engine.add_fact("OpenFile", op_id2, process_name, derived_file, ts)
            sensitive_files_opened.add(derived_file)
    
    # 3. 从 alert_events 注入 LeakFile 事实
    alert_events = module3_result.get("alert_events", [])
    for event in alert_events:
        app_name = event.app_name if hasattr(event, 'app_name') else event.get('app_name', '')
        file_path = event.file_path if hasattr(event, 'file_path') else event.get('file_path', '')
        file_name = event.file_name if hasattr(event, 'file_name') else event.get('file_name', '')
        timestamp_str = event.timestamp if hasattr(event, 'timestamp') else event.get('timestamp', '')
        upload_content = event.upload_content if hasattr(event, 'upload_content') else event.get('upload_content', '')
        
        ts = parse_ts(timestamp_str)
        leak_file = file_path or upload_content
        
        if leak_file:
            # 找到外发的进程名（从日志中匹配）
            leak_process = app_name
            # 尝试匹配更精确的进程名
            for log in logs:
                log_file = log.get('file_path', '')
                if (os.path.basename(log_file) == file_name and 
                    log.get('event_type', '') in ['file_upload', 'upload_detected', 'created', 'modified']):
                    window_title = log.get('window_info', {}).get('window_title', '')
                    if app_name.lower() in window_title.lower() or '邮箱' in window_title:
                        leak_process = log.get('process_info', {}).get('process_name', app_name)
                        break
            
            op_id = next_op_id("leak")
            engine.add_fact("LeakFile", op_id, leak_process, leak_file, "network", ts)
            supplementary_facts.append(
                DatalogFact(relation="LeakFile", operation_id=op_id,
                            process=leak_process, file=leak_file,
                            timestamp=timestamp_str,
                            description=f"补充: {leak_process} 外发 {os.path.basename(leak_file)} ({app_name})")
            )
            print(f"      + LeakFile({leak_process}, {os.path.basename(leak_file)})")
            
            # 确保泄露文件也有被某进程打开的事实 (用于连接污点链)
            if leak_file not in sensitive_files_opened:
                # 找到实际处理这个文件的进程
                for log in logs:
                    if log.get('file_path', '') == leak_file:
                        open_proc = log.get('process_info', {}).get('process_name', leak_process)
                        log_ts = parse_ts(log.get('timestamp', ''))
                        op_id2 = next_op_id("open")
                        engine.add_fact("OpenFile", op_id2, open_proc, leak_file, log_ts)
                        sensitive_files_opened.add(leak_file)
                        
                        # 如果打开进程和泄露进程不同，加 CrossProcessTransfer
                        if open_proc != leak_process:
                            op_id3 = next_op_id("cross")
                            engine.add_fact("CrossProcessTransfer", op_id3,
                                          open_proc, leak_process, leak_file, ts)
                            supplementary_facts.append(
                                DatalogFact(relation="CrossProcessTransfer",
                                            operation_id=op_id3,
                                            process=open_proc,
                                            from_process=open_proc,
                                            to_process=leak_process,
                                            shared_data=leak_file,
                                            file=leak_file,
                                            timestamp=timestamp_str,
                                            description=f"补充: {open_proc} → {leak_process} ({os.path.basename(leak_file)})")
                            )
                            print(f"      + CrossProcessTransfer({open_proc} → {leak_process})")
                        break
    
    # 4. 也处理 info_events (非黑名单但仍是上传行为)
    info_events = module3_result.get("info_events", [])
    for event in info_events:
        app_name = event.app_name if hasattr(event, 'app_name') else event.get('app_name', '')
        file_path = event.file_path if hasattr(event, 'file_path') else event.get('file_path', '')
        timestamp_str = event.timestamp if hasattr(event, 'timestamp') else event.get('timestamp', '')
        
        ts = parse_ts(timestamp_str)
        if file_path:
            op_id = next_op_id("leak")
            # 找进程名
            leak_process = app_name
            for log in logs:
                if log.get('file_path', '') == file_path:
                    leak_process = log.get('process_info', {}).get('process_name', app_name)
                    break
            engine.add_fact("LeakFile", op_id, leak_process, file_path, "network", ts)
            supplementary_facts.append(
                DatalogFact(relation="LeakFile", operation_id=op_id,
                            process=leak_process, file=file_path,
                            timestamp=timestamp_str,
                            description=f"补充: {leak_process} 上传 {os.path.basename(file_path)} (info)")
            )
    
    # 5. 从日志中检测文件分片/派生模式
    #    如果模块3没发现文件映射（比如敏感文件不在配置中），尝试从日志推断
    if not direct_mappings:
        print("      📌 模块3未发现文件映射，从日志推断文件派生关系...")
        
        # 构建文件名 → 日志条目的映射
        file_events = {}  # file_path → 最早的日志
        for log in logs:
            fp = log.get('file_path', '')
            fn = log.get('file_name', '')
            if fp and fn:
                if fp not in file_events:
                    file_events[fp] = log
        
        # 查找可能的原始文件和派生文件
        # 策略：如果文件A的文件名是文件B文件名的前缀（去掉后缀后），且B在A之后出现
        # 例如：员工薪资明细表Q4.xlsx → 员工薪资明细表Q4_part1.xlsx
        open_file_names = {}  # basename_no_ext → full_path
        for log in logs:
            if log.get('event_type', '') in ['file_open', 'opened']:
                fn = log.get('file_name', '')
                fp = log.get('file_path', '')
                if fn and fp:
                    base_no_ext = os.path.splitext(fn)[0]
                    open_file_names[base_no_ext] = fp
        
        derived_files = {}  # full_path → (original_path, process, timestamp)
        for log in logs:
            if log.get('event_type', '') in ['created', 'file_upload']:
                fn = log.get('file_name', '')
                fp = log.get('file_path', '')
                if fn and fp:
                    fn_no_ext = os.path.splitext(fn)[0]
                    # 检查是否是某个已打开文件的派生
                    for orig_base, orig_path in open_file_names.items():
                        if (fn_no_ext != orig_base and 
                            fn_no_ext.startswith(orig_base) and
                            fp != orig_path):
                            proc = log.get('process_info', {}).get('process_name', 'unknown')
                            ts_str = log.get('timestamp', '')
                            derived_files[fp] = (orig_path, proc, ts_str)
                            break
        
        for derived_path, (orig_path, proc, ts_str) in derived_files.items():
            ts = parse_ts(ts_str)
            orig_name = os.path.basename(orig_path)
            derived_name = os.path.basename(derived_path)
            
            # 确保原始文件有 OpenFile
            if orig_path not in sensitive_files_opened:
                # 找原始文件的打开进程
                orig_proc = proc
                for log in logs:
                    if (log.get('file_path', '') == orig_path and 
                        log.get('event_type', '') in ['file_open', 'opened']):
                        orig_proc = log.get('process_info', {}).get('process_name', proc)
                        break
                op_id = next_op_id("open")
                engine.add_fact("OpenFile", op_id, orig_proc, orig_path, ts)
                sensitive_files_opened.add(orig_path)
                supplementary_facts.append(
                    DatalogFact(relation="OpenFile", operation_id=op_id,
                                process=orig_proc, file=orig_path, timestamp=ts_str,
                                description=f"补充: {orig_proc} 打开 {orig_name}")
                )
            
            # TransferFile: 原始 → 派生 (同进程)
            # 使用原始文件的打开进程（而非创建派生文件的进程）
            transfer_proc = proc
            for t_tup in list(engine.facts.get("OpenFile", [])):
                if len(t_tup.args) >= 3 and t_tup.args[2] == orig_path:
                    transfer_proc = t_tup.args[1]
                    break
            
            op_id = next_op_id("transfer")
            engine.add_fact("TransferFile", op_id, transfer_proc, orig_path, derived_path, ts)
            supplementary_facts.append(
                DatalogFact(relation="TransferFile", operation_id=op_id,
                            process=transfer_proc, file=orig_path, dst_file=derived_path,
                            timestamp=ts_str,
                            description=f"补充: {orig_name} → {derived_name} (日志推断)")
            )
            print(f"      + TransferFile({transfer_proc}, {orig_name} → {derived_name}) [日志推断]")
            
            # 确保派生文件也有 OpenFile
            if derived_path not in sensitive_files_opened:
                op_id2 = next_op_id("open")
                engine.add_fact("OpenFile", op_id2, transfer_proc, derived_path, ts)
                sensitive_files_opened.add(derived_path)
    
    # 6. 从日志中的 file_upload 事件注入 LeakFile
    for log in logs:
        if log.get('event_type', '') == 'file_upload':
            fp = log.get('file_path', '')
            fn = log.get('file_name', '')
            proc = log.get('process_info', {}).get('process_name', 'unknown')
            ts_str = log.get('timestamp', '')
            ts = parse_ts(ts_str)
            window = log.get('window_info', {}).get('window_title', '')
            
            if fp:
                # 检查是否已经有这个LeakFile
                already_has = False
                for existing in engine.facts.get("LeakFile", []):
                    if len(existing.args) >= 3 and existing.args[2] == fp:
                        already_has = True
                        break
                
                if not already_has:
                    op_id = next_op_id("leak")
                    channel = "network"
                    if '邮箱' in window:
                        channel = "email"
                    engine.add_fact("LeakFile", op_id, proc, fp, channel, ts)
                    supplementary_facts.append(
                        DatalogFact(relation="LeakFile", operation_id=op_id,
                                    process=proc, file=fp, timestamp=ts_str,
                                    description=f"补充: {proc} 上传 {fn} ({window})")
                    )
                    print(f"      + LeakFile({proc}, {fn}) [日志file_upload, {window}]")
                    # 确保有对应的污染事实
                    owning_procs = set()
                    for of in engine.facts.get("OpenFile", []):
                        if len(of.args) >= 3 and of.args[2] == fp:
                            owning_procs.add(of.args[1])
                    for tf in engine.facts.get("TransferFile", []):
                        if len(tf.args) >= 4 and tf.args[3] == fp:
                            owning_procs.add(tf.args[1])
                    for cpt in engine.facts.get("CrossProcessTransfer", []):
                        if len(cpt.args) >= 4 and cpt.args[3] == fp:
                            owning_procs.add(cpt.args[2])
                    
                    if proc not in owning_procs:
                        if owning_procs:
                            from_proc = list(owning_procs)[0]
                            op_id_cross = next_op_id("cross")
                            engine.add_fact("CrossProcessTransfer", op_id_cross, from_proc, proc, fp, ts)
                            supplementary_facts.append(
                                DatalogFact(relation="CrossProcessTransfer", operation_id=op_id_cross,
                                            process=from_proc, from_process=from_proc, to_process=proc,
                                            shared_data=fp, file=fp, timestamp=ts_str,
                                            description=f"补充: {from_proc} → {proc} ({fn})")
                            )
                            print(f"      + CrossProcessTransfer({from_proc} → {proc}) [日志推断]")
                        else:
                            if fp not in sensitive_files_opened:
                                op_id2 = next_op_id("open")
                                engine.add_fact("OpenFile", op_id2, proc, fp, ts)
                                sensitive_files_opened.add(fp)
    
    print(f"      共注入 {len(supplementary_facts)} 条补充事实")
    return supplementary_facts


def run_threat_detector(logs: List[Dict],
                        module3_result: Dict[str, Any]) -> Dict[str, Any]:
    """
    运行模块4: ThreatDetector

    功能:
    1. 从模块3结果中构建视频帧分析数据
    2. 使用 LLM 生成 Datalog 事实
    3. 从模块3实际结果注入补充事实（确保污点链连通）
    4. 运行 Souffle/Python 推理
    5. 检测泄露路径
    """
    print("\n" + "=" * 80)
    print("⚖️ 模块4: ThreatDetector - Datalog 推理")
    print("=" * 80)

    if DatalogEngine is None:
        print("   ⚠️ 模块4未加载，跳过")
        return {"leak_paths": [], "datalog_facts": []}

    # 从模块3结果构建视频帧分析数据（用于 module4 LLM 分析）
    print(f"\n   [deterministic] building Datalog facts from logs/module3 results...")
    deterministic_engine = DatalogEngine()
    deterministic_facts = _inject_connected_facts_from_module3(deterministic_engine, module3_result, logs)
    deterministic_leak_paths = deterministic_engine.query_leak()
    deterministic_engine.cleanup()

    enable_llm_facts = os.getenv("DLD_ENABLE_LLM_FACTS", "0").strip().lower() in {"1", "true", "yes", "on"}
    if deterministic_leak_paths or not enable_llm_facts:
        if deterministic_leak_paths:
            print("   [OK] deterministic Datalog found leak paths; skipping LLM fact generation")
        else:
            print("   [INFO] deterministic Datalog found no leak paths; LLM fact fallback is disabled")
        return {
            "leak_paths": deterministic_leak_paths,
            "datalog_facts": deterministic_facts
        }

    if PromptTemplates is None:
        print("   [WARN] LLM fact fallback requested but PromptTemplates is unavailable")
        return {"leak_paths": [], "datalog_facts": deterministic_facts}

    video_frames = _build_video_frames_from_module3(module3_result)
    print(f"   📊 构建视频帧数据: {len(video_frames)} 个帧事件")

    # 如果模块3没有产出帧分析，从日志模拟
    if not video_frames:
        print("   ⚠️ 模块3无帧分析结果，使用日志模拟视频帧...")
        for log in logs:
            if log.get('upload_detection', {}).get('is_upload'):
                video_frames.append({
                    "timestamp": log.get('timestamp', ''),
                    "app_name": log.get('process_info', {}).get('process_name', ''),
                    "operation_type": "上传",
                    "behavior_category": "数据外发",
                    "description": f"文件 {log.get('file_name', '')} 被上传"
                })

    # 使用 LLM 生成 Datalog 事实
    print(f"\n   🤖 调用 LLM 分析...")
    print(f"      - 日志条目: {len(logs)} 条")
    print(f"      - 视频帧: {len(video_frames)} 个")
    
    llm_facts = []
    try:
        llm_facts = generate_datalog_facts(logs, video_frames)
        print(f"   ✅ LLM 生成 {len(llm_facts)} 条 Datalog 事实")
    except Exception as e:
        print(f"   ⚠️ LLM 分析失败: {e}")
        print(f"   将仅使用模块3结果进行推理...")
    
    for i, fact in enumerate(llm_facts, 1):
        if fact.relation == "CrossProcessTransfer":
            print(f"   {i}. {fact.relation}({fact.from_process} → {fact.to_process})")
        elif fact.dst_file:
            print(f"   {i}. {fact.relation}({fact.process}, {fact.file[:30]}... → {fact.dst_file})")
        else:
            print(f"   {i}. {fact.relation}({fact.process}, {fact.file[:30]}...)")

    # 创建引擎并添加 LLM 生成的事实
    print(f"\n   🔍 运行 Datalog 推理...")
    engine = DatalogEngine()

    for fact in llm_facts:
        args = fact.to_souffle_args()
        if args:
            engine.add_fact(fact.relation, *args)

    # 从模块3结果注入补充事实（确保污点链连通）
    supplementary_facts = _inject_connected_facts_from_module3(engine, module3_result, logs)
    all_facts = llm_facts + supplementary_facts

    # 执行推理
    leak_paths = engine.query_leak()
    engine.cleanup()

    print(f"\n   ✅ ThreatDetector 完成")
    print(f"   🚨 泄露路径: {len(leak_paths)} 条")

    return {
        "leak_paths": leak_paths,
        "datalog_facts": all_facts
    }


def _build_video_frames_from_module3(module3_result: Dict[str, Any]) -> List[Dict]:
    """
    从模块3结果中构建视频帧分析数据

    数据来源:
    1. upload_events 的 extra_info（原始模块1帧分析事件数据）
    2. operation_records（所有敏感操作记录，包括非上传操作）
    """
    video_frames = []
    seen_keys = set()  # 去重

    # 1. 从 upload_events 提取帧分析数据
    upload_events = module3_result.get("upload_events", [])
    for event in upload_events:
        extra_info = event.extra_info if hasattr(event, 'extra_info') else {}
        if extra_info:
            # extra_info 就是模块1的原始帧事件数据
            dedup_key = f"{extra_info.get('time_range', '')}|{extra_info.get('app_name', '')}|{extra_info.get('operation_type', '')}"
            if dedup_key not in seen_keys:
                seen_keys.add(dedup_key)
                video_frames.append(extra_info)
        else:
            # 如果没有 extra_info，从 UploadEvent 字段构建
            dedup_key = f"{event.timestamp}|{event.app_name}|{event.operation_type}"
            if dedup_key not in seen_keys:
                seen_keys.add(dedup_key)
                video_frames.append({
                    "timestamp": event.timestamp,
                    "app_name": event.app_name,
                    "operation_type": event.operation_type,
                    "behavior_category": event.behavior_category,
                    "description": event.description,
                    "time_range": event.time_range,
                })

    # 2. 从 operation_records 补充非上传操作
    operation_records = module3_result.get("operation_records", [])
    for record in operation_records:
        dedup_key = f"{record.get('operation_time', '')}|{record.get('app_name', '')}|{record.get('operation', '')}"
        if dedup_key not in seen_keys:
            seen_keys.add(dedup_key)
            video_frames.append({
                "timestamp": record.get("operation_time", ""),
                "app_name": record.get("app_name", ""),
                "operation_type": record.get("operation", ""),
                "behavior_category": "",
                "description": record.get("description", ""),
            })

    return video_frames


def generate_datalog_facts(logs: List[Dict], video_frames: List[Dict]) -> List[DatalogFact]:
    """使用 LLM 生成 Datalog 事实"""
    from openai import OpenAI

    api_key = os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY")
    model_name = os.getenv("LLM_MODEL_NAME", "qwen-plus")
    base_url = os.getenv("LLM_BASE_URL")

    if not api_key:
        raise ValueError("❌ 未配置 LLM_API_KEY")

    client = OpenAI(api_key=api_key, base_url=base_url) if base_url else OpenAI(api_key=api_key)

    messages = PromptTemplates.get_messages(logs, video_frames)

    response = client.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=0.1
    )

    result_text = response.choices[0].message.content
    json_match = re.search(r'\[.*\]', result_text, re.DOTALL)

    if not json_match:
        raise ValueError("LLM未返回有效JSON格式")

    facts_data = json.loads(json_match.group())

    facts = []
    for fd in facts_data:
        if fd['relation'] == 'CrossProcessTransfer':
            fact = DatalogFact(
                relation=fd['relation'],
                operation_id=fd['operation_id'],
                process=fd.get('process', 'Unknown'),
                from_process=fd.get('from_process'),
                to_process=fd.get('to_process'),
                shared_data=fd.get('shared_data'),
                file=fd.get('shared_data', ''),
                timestamp=fd.get('timestamp'),
                description=fd.get('description')
            )
        else:
            fact = DatalogFact(
                relation=fd['relation'],
                operation_id=fd['operation_id'],
                process=fd.get('process', 'Unknown'),
                file=fd.get('file', ''),
                dst_file=fd.get('dst_file'),
                timestamp=fd.get('timestamp'),
                description=fd.get('description')
            )
        facts.append(fact)

    return facts


# ==================== 证据报告 ====================

def generate_evidence_report(log_file: str, video_file: str,
                             module3_result: Dict[str, Any],
                             threat_detector_result: Dict[str, Any],
                             output_dir: str = None) -> str:
    """生成完整的泄露证据报告"""

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if output_dir is None:
        output_dir = os.path.join(PROJECT_DIR, 'output')
    os.makedirs(output_dir, exist_ok=True)

    leak_paths = threat_detector_result.get("leak_paths", [])
    alert_events = module3_result.get("alert_events", [])
    info_events = module3_result.get("info_events", [])
    operation_records = module3_result.get("operation_records", [])
    statistics = module3_result.get("statistics", {})
    file_mappings = module3_result.get("file_mappings", {})

    report = {
        "report_id": f"full_evidence_{timestamp}",
        "generated_at": datetime.now().isoformat(),
        "input": {
            "log_file": log_file,
            "video_file": video_file
        },
        "summary": {
            "module3_events_processed": statistics.get("total_events_processed", 0),
            "module3_upload_events": statistics.get("upload_events_detected", 0),
            "module3_alert_events": len(alert_events),
            "module3_info_events": len(info_events),
            "module3_operation_records": len(operation_records),
            "module3_blacklist_alerts": statistics.get("blacklist_alerts", 0),
            "module4_datalog_facts": len(threat_detector_result.get("datalog_facts", [])),
            "module4_leak_paths": len(leak_paths)
        },
        "module3_risk_hunter": {
            "alert_events": [event.to_dict() for event in alert_events],
            "info_events": [event.to_dict() for event in info_events],
            "operation_records": sorted(
                operation_records,
                key=lambda item: item.get("operation_time", "")
            ),
            "file_mappings": file_mappings,
        },
        "module4_threat_detector": {
            "datalog_facts": [
                {
                    "relation": f.relation,
                    "operation_id": f.operation_id,
                    "process": f.process,
                    "file": f.file,
                    "description": f.description
                }
                for f in threat_detector_result.get("datalog_facts", [])
            ],
            "leak_paths": [lp.to_dict() for lp in leak_paths]
        },
        "conclusion": "🚨 发现数据泄露风险！" if (leak_paths or alert_events) else "✅ 未发现数据泄露"
    }

    report_file = os.path.join(output_dir, f"full_evidence_{timestamp}.json")
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    return report_file


# ==================== 主流程 ====================

def run_full_e2e_pipeline(log_file: str, video_file: str):
    """运行完整的 E2E 流程"""

    print("""
╔══════════════════════════════════════════════════════════════════════╗
║                                                                      ║
║        🔍 DataLeakDetector Full E2E Pipeline                         ║
║                                                                      ║
║   完整流程:                                                          ║
║   1. 加载数据                                                        ║
║   2. 模块3 RiskHunter (调用 模块2→模块1) → 风险报警                   ║
║   3. 模块4 ThreatDetector → Datalog 推理                             ║
║   4. 输出证据报告                                                    ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝
    """)

    # 导入模块
    import_modules()

    # 阶段1: 加载数据
    print("\n" + "=" * 80)
    print("📂 阶段1: 加载数据")
    print("=" * 80)

    logs = load_logs(log_file)
    rec_start = get_recording_start_time(logs)
    print(f"   日志文件: {os.path.basename(log_file)}")
    print(f"   视频文件: {os.path.basename(video_file)}")
    print(f"   日志事件: {len(logs)} 条")
    print(f"   录屏开始时间: {rec_start}")

    # 创建临时 INDEX.md
    output_dir = os.path.join(PROJECT_DIR, 'output', 'temp')
    index_path = create_index_file(rec_start, output_dir)

    # 阶段2: 运行模块3完整 Pipeline (内部调用 模块2 → 模块1)
    module3_result = run_module3_pipeline(
        log_file=log_file,
        video_path=video_file,
        index_path=index_path,
    )

    # 阶段3: 运行模块4 ThreatDetector
    threat_detector_result = run_threat_detector(
        logs=logs,
        module3_result=module3_result,
    )

    # 阶段4: 结果汇总
    print("\n" + "=" * 80)
    print("📊 阶段4: 检测结果汇总")
    print("=" * 80)

    leak_paths = threat_detector_result.get("leak_paths", [])
    alert_events = module3_result.get("alert_events", [])

    if leak_paths:
        print(f"\n🚨 检测到 {len(leak_paths)} 条泄露路径:\n")
        for i, leak in enumerate(leak_paths, 1):
            print(f"泄露路径 #{i}:")
            print(f"  📁 泄露文件: {leak.leaked_file}")
            print(f"  📤 泄露进程: {leak.leaking_proc}")
            print(f"  🌐 泄露渠道: {leak.leak_channel}")
            print(f"  🛤️  完整路径: {leak.full_path}")
            print()

    if alert_events:
        print(f"\n⚠️ 黑名单报警 {len(alert_events)} 个:")
        for evt in alert_events[:5]:
            print(f"   - {evt.file_name} → {evt.app_name} ({evt.app_category})")
            print(f"     外发内容: {evt.upload_content}")
            print(f"     映射链: {evt.upload_content_mapping_link}")

    # 生成报告
    report_file = generate_evidence_report(
        log_file, video_file,
        module3_result, threat_detector_result
    )
    print(f"\n📄 证据报告: {report_file}")

    print("\n" + "=" * 80)
    print("✅ Full E2E Pipeline 完成")
    print("=" * 80)

    if leak_paths or alert_events:
        print(f"\n⚠️ 发现数据泄露风险！")
    else:
        print(f"\n✅ 未发现数据泄露")

    return len(leak_paths) > 0 or len(alert_events) > 0


def main():
    parser = argparse.ArgumentParser(description='DataLeakDetector Full E2E Pipeline')
    parser.add_argument('--log', '-l', type=str, required=True, help='日志文件路径')
    parser.add_argument('--video', '-v', type=str, required=True, help='视频文件路径')

    args = parser.parse_args()

    # 验证文件存在
    if not os.path.exists(args.log):
        print(f"❌ 日志文件不存在: {args.log}")
        sys.exit(1)
    if not os.path.exists(args.video):
        print(f"❌ 视频文件不存在: {args.video}")
        sys.exit(1)

    has_risk = run_full_e2e_pipeline(
        log_file=args.log,
        video_file=args.video,
    )

    sys.exit(1 if has_risk else 0)


if __name__ == "__main__":
    main()
