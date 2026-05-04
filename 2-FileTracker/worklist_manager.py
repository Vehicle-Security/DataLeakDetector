# worklist_manager.py
"""
Worklist 管理模块
用于维护敏感文件事件的动态工作列表，供模块3调用
"""

from typing import List, Dict, Any, Set, Optional
from dataclasses import dataclass, field
from datetime import datetime
import json
from pathlib import Path

from behavior_analysis_tools import normalize_file_path

INTERNAL_ARTIFACT_BASENAMES = {
    "logs.json",
    "keyevents.json",
    "index.md",
    "global.json",
    "global.dat",
    "config.ini",
    "onceflag.ini",
    "personalsetting.xml",
    "appsettingapp.dat",
    "amcache.hve",
}

INTERNAL_ARTIFACT_SEGMENTS = (
    "/screenmonitor/",
    "/winows_monitor/",
    "/windows_monitor/",
    "/recordings/session_",
    "/logs/",
    "/video/",
    "/appdata/",
    "/cache/",
    "/cookies/",
    "/history/",
    "/temp/",
    "/tmp/",
)

INTERNAL_ARTIFACT_SUFFIXES = (
    ".sqlite",
    ".sqlite3",
    ".db",
    ".db-journal",
    ".db-wal",
    ".wal",
    ".journal",
    ".lock",
    ".dat",
    ".ini",
    ".hve",
    ".log",
    ".tmp",
    ".lnk",
    ".crdownload",
)


def is_internal_artifact_path(file_path: str) -> bool:
    normalized = normalize_file_path(file_path).casefold()
    if not normalized:
        return True

    basename = normalized.rsplit("/", 1)[-1]
    if basename in INTERNAL_ARTIFACT_BASENAMES:
        return True
    if basename.endswith(INTERNAL_ARTIFACT_SUFFIXES):
        return True

    return any(segment in normalized for segment in INTERNAL_ARTIFACT_SEGMENTS)


@dataclass()
class SensitiveFileEvent:
    """
    敏感文件事件类
    
    Attributes:
        event_id: 事件唯一标识
        original_file: 原始敏感文件路径
        current_file: 当前文件路径（可能经过重命名/复制等）
        event_type: 事件类型（opened, renamed, copied, compressed等）
        process_info: 关联进程信息
        timestamp: 事件时间戳
        is_hidden: 是否为隐藏操作（重命名、压缩等）
        raw_event: 原始事件数据
    """
    event_id: str
    original_file: str
    current_file: str
    event_type: str
    process_info: Dict[str, Any]
    timestamp: str
    is_hidden: bool = False
    raw_event: Dict[str, Any] = field(default_factory=dict)
    
    def __hash__(self):
        return hash(self.event_id)
    
    def __eq__(self, other):
        if isinstance(other, SensitiveFileEvent):
            return self.event_id == other.event_id
        return False


class WorklistManager:
    """
    工作列表管理器
    
    主要功能：
    1. 维护敏感文件列表
    2. 查询某个文件是否是敏感文件
    3. 扫描日志事件并生成工作列表
    4. 动态更新工作列表
    """
    
    def __init__(self, sensitive_files: Optional[List[str]] = None):
        """
        初始化工作列表管理器
        
        Args:
            sensitive_files: 初始敏感文件路径列表
        """
        # 工作列表（待处理的敏感文件事件）
        self.worklist: List[SensitiveFileEvent] = []
        
        # 文件映射关系：当前文件 -> 原始文件（用于追踪重命名、复制等）
        self.file_mapping: Dict[str, str] = {}
        
        # 文件直接映射关系（复数）：派生文件 -> 直接父文件（只记录上一层，用于构建完整映射链）
        self.file_mappings: Dict[str, str] = {}
        
        # 已加入工作列表的事件ID集合（避免重复处理）
        self.processed_events: Set[str] = set()

        # 同一秒内同一文件可能同时出现 browser_file_access/opened 等多条日志，
        # 对模块2/模块1而言它们描述的是同一次用户操作，合并处理即可。
        self.processed_semantic_events: Set[str] = set()
        
        # 敏感文件集合（规范化后的文件路径）
        self.sensitive_files: Set[str] = set()
        if sensitive_files:
            for file_path in sensitive_files:
                self.sensitive_files.add(self._normalize_path(file_path))
    
    def add_sensitive_file(self, file_path: str) -> None:
        """
        添加敏感文件到监控列表
        
        Args:
            file_path: 文件路径
        """
        self.sensitive_files.add(self._normalize_path(file_path))
    
    def add_sensitive_files(self, file_paths: List[str]) -> None:
        """
        批量添加敏感文件
        
        Args:
            file_paths: 文件路径列表
        """
        for path in file_paths:
            self.add_sensitive_file(path)
    
    def is_sensitive_file(self, file_path: str) -> bool:
        """
        查询某个文件是否是敏感文件
        
        Args:
            file_path: 文件路径
            
        Returns:
            是否为敏感文件
        """
        normalized_path = self._normalize_path(file_path)
        
        if normalized_path in self.sensitive_files:
            return True
        
        # 检查是否是某个敏感文件的派生文件（重命名、复制等）
        if normalized_path in self.file_mapping:
            original = self.file_mapping[normalized_path]
            return original in self.sensitive_files
        
        return False
    
    def get_original_file(self, file_path: str) -> Optional[str]:
        """
        获取文件的原始路径（如果是派生文件）
        
        Args:
            file_path: 当前文件路径
            
        Returns:
            原始文件路径，如果没有则返回None
        """
        normalized_path = self._normalize_path(file_path)
        return self.file_mapping.get(normalized_path)
    
    def scan_and_build_worklist(self, log_events: List[Dict[str, Any]]) -> int:
        """
        扫描日志事件，构建工作列表
        
        Args:
            log_events: 日志事件列表（JSON格式）
            
        Returns:
            新增的敏感事件数量
        """
        added_count = 0
        
        for event in log_events:
            event_type = event.get("event_type", "")
            file_path = event.get("file_path", "")
            timestamp = event.get("timestamp", "")
            process_info = event.get("process_info", {})
            
            event_id = self._generate_event_id(timestamp, file_path, event_type)
            
            if event_id in self.processed_events:
                continue
            
            if self._is_sensitive_event(event):
                semantic_key = self._generate_semantic_event_key(event)
                if semantic_key in self.processed_semantic_events:
                    self.processed_events.add(event_id)
                    continue

                sensitive_event = self._create_sensitive_event(event, event_id)
                self.worklist.append(sensitive_event)
                self.processed_events.add(event_id)
                self.processed_semantic_events.add(semantic_key)
                added_count += 1
        
        return added_count
    
    def get_next_event(self) -> Optional[SensitiveFileEvent]:
        """
        从工作列表中获取下一个待处理事件
        
        Returns:
            下一个敏感文件事件，如果列表为空则返回None
        """
        if self.worklist:
            return self.worklist.pop(0)
        return None
    
    def peek_next_event(self) -> Optional[SensitiveFileEvent]:
        """
        查看下一个待处理事件但不移除
        
        Returns:
            下一个敏感文件事件，如果列表为空则返回None
        """
        if self.worklist:
            return self.worklist[0]
        return None
    
    def add_event(self, event: SensitiveFileEvent) -> None:
        """
        手动添加事件到工作列表
        
        Args:
            event: 敏感文件事件
        """
        if event.event_id not in self.processed_events:
            self.worklist.append(event)
            self.processed_events.add(event.event_id)  # 标记为已加入，避免重复添加
            print(f"   ✅ 成功添加新事件ID: {event.event_id}")
        else:
            print(f"   ⚠️ 事件ID已存在，跳过添加: {event.event_id}")
    
    def is_empty(self) -> bool:
        """
        检查工作列表是否为空
        
        Returns:
            工作列表是否为空
        """
        return len(self.worklist) == 0
    
    def size(self) -> int:
        """
        获取工作列表大小
        
        Returns:
            工作列表中的事件数量
        """
        return len(self.worklist)
    
    def update_file_mapping(self, original_file: str, new_file: str) -> None:
        """
        更新文件映射关系（用于追踪重命名、复制等操作）
        
        同时维护两个映射字典：
        - file_mapping: 派生文件 -> 最初源文件（直接追溯，用于快速查询）
        - file_mappings: 派生文件 -> 直接父文件（只记录上一层，用于构建完整映射链）
        
        Args:
            original_file: 原始文件路径
            new_file: 新文件路径
        """
        original_normalized = self._normalize_path(original_file)
        new_normalized = self._normalize_path(new_file)
        
        # 更新 file_mappings：只记录直接父子关系
        self.file_mappings[new_normalized] = original_normalized
        
        # 更新 file_mapping：追溯到最初的源文件
        # 如果原始文件本身是派生文件，追溯到最初的源文件
        if original_normalized in self.file_mapping:
            root_file = self.file_mapping[original_normalized]
            self.file_mapping[new_normalized] = root_file
        else:
            self.file_mapping[new_normalized] = original_normalized
    
    def get_mapping_chain(self, file_path: str) -> Optional[str]:
        """
        获取文件的完整映射链（从最初源文件到当前文件）
        
        Args:
            file_path: 当前文件路径
            
        Returns:
            映射链字符串（例如："file_a.docx -> file_b.docx -> file_c.docx"）
            如果不是派生文件，返回 None
        """
        normalized_path = self._normalize_path(file_path)
        
        # 如果不是派生文件，返回 None
        if normalized_path not in self.file_mappings:
            return None
        
        # 从当前文件向上追溯，构建映射链
        chain = []
        current = normalized_path
        
        # 向上追溯最多10层，防止死循环
        for _ in range(10):
            if current in self.file_mappings:
                parent = self.file_mappings[current]
                chain.insert(0, parent)
                current = parent
            else:
                break
        
        # 添加当前文件
        chain.append(normalized_path)
        
        # 返回映射链字符串
        return " -> ".join(chain)

    def get_direct_file_mappings(self) -> Dict[str, str]:
        """
        获取直接文件映射关系（子文件 -> 直接父文件）

        Returns:
            直接映射字典
        """
        return dict(sorted(self.file_mappings.items()))

    def get_full_file_mapping_chains(self) -> Dict[str, str]:
        """
        获取所有派生文件的完整映射链

        Returns:
            字典，key为当前派生文件，value为完整映射链字符串
        """
        chains: Dict[str, str] = {}
        for derived_file in sorted(self.file_mappings.keys()):
            chain = self.get_mapping_chain(derived_file)
            if chain:
                chains[derived_file] = chain
        return chains

    def export_file_mappings(self) -> Dict[str, Any]:
        """
        导出文件映射关系（用于输出JSON）

        Returns:
            包含直接映射关系和完整映射链的字典
        """
        return {
            "direct_file_mappings": self.get_direct_file_mappings(),
            "full_file_mapping_chains": self.get_full_file_mapping_chains(),
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        获取工作列表统计信息
        
        Returns:
            统计信息字典
        """
        return {
            "sensitive_files_count": len(self.sensitive_files),
            "worklist_size": len(self.worklist),
            "processed_events_count": len(self.processed_events),
            "file_mappings_count": len(self.file_mapping),
            "direct_mappings_count": len(self.file_mappings),
            "event_types": self._count_event_types()
        }
    
    def clear_worklist(self) -> None:
        """
        清空工作列表（不清空敏感文件列表）
        """
        self.worklist.clear()
    
    def reset(self) -> None:
        """
        重置所有状态
        """
        self.sensitive_files.clear()
        self.worklist.clear()
        self.file_mapping.clear()
        self.file_mappings.clear()
        self.processed_events.clear()
        self.processed_semantic_events.clear()
    
    # ==================== 私有方法 ====================
    
    def _normalize_path(self, file_path: str) -> str:
        """
        规范化文件路径（去除多余的斜杠等）
        
        Args:
            file_path: 原始文件路径
            
        Returns:
            规范化后的路径
        """
        return normalize_file_path(file_path)
    
    def _generate_event_id(self, timestamp: str, file_path: str, event_type: str) -> str:
        """
        生成事件唯一ID
        
        Args:
            timestamp: 时间戳
            file_path: 文件路径
            event_type: 事件类型
            
        Returns:
            事件ID
        """
        #return f"{timestamp}_{event_type}_{hash(file_path)}"
        # 只保留时间戳到秒级别，避免同一秒内同一文件的重复事件被多次处理
        timestamp_sec = timestamp.split('.')[0] if '.' in timestamp else timestamp
        normalized_path = self._normalize_path(file_path)
        return f"{timestamp_sec}_{event_type}_{hash(normalized_path)}"

    def _generate_semantic_event_key(self, event: Dict[str, Any]) -> str:
        timestamp = str(event.get("timestamp", ""))
        timestamp_sec = timestamp.split('.')[0] if '.' in timestamp else timestamp
        file_path = self._normalize_path(event.get("file_path", ""))
        original_file = self._normalize_path(event.get("original_file", "")) or file_path
        return f"{timestamp_sec}|{original_file}|{file_path}"
    
    def _is_sensitive_event(self, event: Dict[str, Any]) -> bool:
        """
        判断事件是否涉及敏感文件
        
        Args:
            event: 事件数据
            
        Returns:
            是否为敏感事件
        """
        file_path = event.get("file_path", "")
        event_type = event.get("event_type", "")

        if is_internal_artifact_path(file_path):
            return False
        
        # Worklist 目前关注所有下面日志的事件类型
        """ # 只关注文件打开、修改、重命名、复制、转换格式等操作
        relevant_types = ["opened", "modified", "renamed", "copied", "deleted", "moved", "upload_detected", "file_selected", "compressed", "converted"]
        
        日志event_type           说明	
        created	             文件创建
        opened               文件打开	
        modified	         文件修改/访问
        deleted	             文件删除	
        moved                文件移动	
        renamed	             文件重命名	
        upload_detected      检测到上传行为	
        file_selected        用户通过文件对话框选择文件	
        app_switch           应用切换（切换到黑名单桌面应用）	
        website_visit	     访问黑名单网站
        
        if event_type not in relevant_types:
            return False
         """
        
        return self.is_sensitive_file(file_path)
    
    def _create_sensitive_event(
        self, 
        event: Dict[str, Any], 
        event_id: str
    ) -> SensitiveFileEvent:
        """
        从原始事件创建敏感文件事件对象
        
        Args:
            event: 原始事件数据
            event_id: 事件ID
            
        Returns:
            敏感文件事件对象
        """
        file_path = event.get("file_path", "")
        normalized_path = self._normalize_path(file_path)
        
        original_file = self.get_original_file(file_path) or normalized_path
        
        # 隐藏行为判断由 behavior_analysis_graph 模块完成
        is_hidden = False
        
        return SensitiveFileEvent(
            event_id=event_id,
            original_file=original_file,
            current_file=normalized_path,
            event_type=event.get("event_type", ""),
            process_info=event.get("process_info", {}),
            timestamp=event.get("timestamp", ""),
            is_hidden=is_hidden,
            raw_event=event
        )
    
    def _count_event_types(self) -> Dict[str, int]:
        """
        统计工作列表中各类型事件的数量
        
        Returns:
            事件类型统计
        """
        type_counts = {}
        for event in self.worklist:
            event_type = event.event_type
            type_counts[event_type] = type_counts.get(event_type, 0) + 1
        return type_counts


def load_log_from_json(json_file: str) -> List[Dict[str, Any]]:
    """
    从JSON文件加载日志事件
    
    Args:
        json_file: JSON文件路径
        
    Returns:
        事件列表
    """
    with open(json_file, 'r', encoding='utf-8') as f:
        return json.load(f)
