"""
污点传播追踪器 - Taint Tracker
自动追踪敏感数据的流动和转换
"""

from typing import Dict, List, Set, Optional
from datetime import datetime, timedelta
import logging

from ..core.event_bus import Event, EventBus
from ..core.memory import Memory, Node, Relationship

logger = logging.getLogger(__name__)


class TaintRule:
    """
    污点传播规则
    定义什么样的操作会导致污点传播
    """
    
    def __init__(
        self, 
        name: str,
        source_event: str,  # 源事件类型
        target_event: str,  # 目标事件类型
        propagation_type: str  # 传播类型：copy, convert, derive
    ):
        self.name = name
        self.source_event = source_event
        self.target_event = target_event
        self.propagation_type = propagation_type
    
    def matches(self, event: Event, previous_event: Optional[Event]) -> bool:
        """
        检查事件是否匹配此规则
        
        Args:
            event: 当前事件
            previous_event: 前一个事件
            
        Returns:
            True如果匹配
        """
        if not previous_event:
            return False
        
        return (
            previous_event.event_type == self.source_event and
            event.event_type == self.target_event
        )


class TaintTracker:
    """
    污点追踪器
    
    核心功能：
    1. 自动识别敏感数据源（如数据库查询结果）
    2. 追踪数据流动（复制、转换、传输）
    3. 自动标记污点
    4. 检测跨模态转换（文本->图片、文本->音频等）
    
    典型场景：
    - 用户从数据库复制敏感数据 -> 标记剪贴板为污点
    - 用户粘贴到文档 -> 传播污点到文档
    - 用户将文档转为PDF -> 传播污点到PDF
    - 用户上传PDF -> 触发告警
    """
    
    def __init__(self, event_bus: EventBus, memory: Memory):
        self.event_bus = event_bus
        self.memory = memory
        
        # 污点传播规则库
        self.rules: List[TaintRule] = []
        self._init_default_rules()
        
        # 敏感数据源定义
        self.sensitive_sources = {
            "database_query",
            "confidential_file_open",
            "secure_storage_access"
        }
        
        # 跨模态转换检测
        self.modal_conversions = {
            ("text", "audio"): ["text_to_speech", "tts", "audio_convert"],
            ("text", "image"): ["screenshot", "text_to_image", "ocr_reverse"],
            ("document", "pdf"): ["pdf_export", "print_to_pdf"],
        }
        
        # 最近事件缓存（用于关联分析）
        self.recent_events: List[Event] = []
        self.max_recent = 100
        
        logger.info("污点追踪器已初始化")
    
    def _init_default_rules(self):
        """初始化默认的污点传播规则"""
        self.rules = [
            TaintRule("copy_propagation", "file_read", "file_write", "copy"),
            TaintRule("clipboard_propagation", "copy_data", "paste_data", "copy"),
            TaintRule("conversion_propagation", "file_open", "file_save_as", "convert"),
            TaintRule("screenshot_propagation", "app_open", "screenshot", "derive"),
        ]
    
    def start(self):
        """启动污点追踪器"""
        # 订阅所有事件
        self.event_bus.subscribe("*", self.track_event)
        logger.info("污点追踪器已启动")
    
    def track_event(self, event: Event):
        """
        追踪单个事件
        
        Args:
            event: 待追踪的事件
        """
        # 添加到最近事件
        self.recent_events.append(event)
        if len(self.recent_events) > self.max_recent:
            self.recent_events.pop(0)
        
        # 检查是否是敏感数据源
        if self._is_sensitive_source(event):
            self._mark_as_sensitive(event)
        
        # 检查污点传播
        self._check_propagation(event)
        
        # 检查跨模态转换
        self._check_modal_conversion(event)
    
    def _is_sensitive_source(self, event: Event) -> bool:
        """
        判断事件是否涉及敏感数据源
        
        Args:
            event: 事件
            
        Returns:
            True如果是敏感源
        """
        # 检查事件类型
        if event.event_type in self.sensitive_sources:
            return True
        
        # 检查事件元数据
        if event.metadata.get("sensitive") is True:
            return True
        
        # 检查目标是否是已知的敏感文件/应用
        if event.target:
            sensitive_keywords = ["confidential", "secret", "database", "customer"]
            return any(kw in event.target.lower() for kw in sensitive_keywords)
        
        return False
    
    def _mark_as_sensitive(self, event: Event):
        """
        标记实体为敏感/污点
        
        Args:
            event: 事件
        """
        logger.info(f"🔴 检测到敏感数据源: {event}")
        
        # 标记actor（执行者）
        if event.actor:
            self.memory.mark_tainted(event.actor, taint_source="sensitive_access")
            
            # 创建或更新actor节点
            actor_node = Node(
                node_id=event.actor,
                node_type="User",
                properties={
                    "is_tainted": True,
                    "taint_source": "sensitive_access",
                    "first_tainted_at": datetime.now().isoformat()
                }
            )
            self.memory.add_node(actor_node)
        
        # 标记target（如果是数据容器，如剪贴板、变量）
        if event.target and event.metadata.get("is_data_container"):
            self.memory.mark_tainted(event.target, taint_source=event.actor)
            
            target_node = Node(
                node_id=event.target,
                node_type="DataContainer",
                properties={
                    "is_tainted": True,
                    "taint_source": event.actor
                }
            )
            self.memory.add_node(target_node)
    
    def _check_propagation(self, event: Event):
        """
        检查污点传播
        
        Args:
            event: 当前事件
        """
        # 获取最近的事件（时间窗口：10秒）
        time_window = timedelta(seconds=10)
        recent = [
            e for e in self.recent_events[-20:]
            if (event.timestamp - e.timestamp) < time_window
        ]
        
        if not recent:
            return
        
        # 检查是否匹配传播规则
        for rule in self.rules:
            for prev_event in reversed(recent):
                if rule.matches(event, prev_event):
                    self._propagate_taint(prev_event, event, rule.propagation_type)
                    break
    
    def _propagate_taint(self, source_event: Event, target_event: Event, prop_type: str):
        """
        执行污点传播
        
        Args:
            source_event: 源事件
            target_event: 目标事件
            prop_type: 传播类型
        """
        # 检查源是否被污染
        source_id = source_event.target or source_event.actor
        if not source_id or not self.memory.check_taint(source_id):
            return
        
        # 传播到目标
        target_id = target_event.target or target_event.actor
        if not target_id:
            return
        
        logger.warning(f"🔴 污点传播: {source_id} -> {target_id} ({prop_type})")
        
        self.memory.propagate_taint(source_id, target_id, prop_type)
        
        # 创建目标节点
        target_node = Node(
            node_id=target_id,
            node_type=self._infer_node_type(target_event),
            properties={
                "is_tainted": True,
                "taint_source": source_id,
                "propagation_type": prop_type
            }
        )
        self.memory.add_node(target_node)
    
    def _check_modal_conversion(self, event: Event):
        """
        检查跨模态转换
        
        这是发现"文本转音频"等隐蔽手段的关键
        
        Args:
            event: 当前事件
        """
        # 查找actor的最近行为序列
        if not event.actor:
            return
        
        actor_events = [
            e for e in self.recent_events[-50:]
            if e.actor == event.actor
        ]
        
        if len(actor_events) < 2:
            return
        
        # 分析行为序列，查找模态转换模式
        for (src_modal, tgt_modal), keywords in self.modal_conversions.items():
            if self._detect_conversion_pattern(actor_events, keywords):
                self._handle_modal_conversion(event, src_modal, tgt_modal)
    
    def _detect_conversion_pattern(self, events: List[Event], keywords: List[str]) -> bool:
        """
        检测转换模式
        
        Args:
            events: 事件序列
            keywords: 关键词列表
            
        Returns:
            True如果检测到转换模式
        """
        # 简化检测：查找事件类型或目标中是否包含关键词
        for event in events[-10:]:
            event_str = f"{event.event_type} {event.target or ''}".lower()
            if any(kw in event_str for kw in keywords):
                return True
        return False
    
    def _handle_modal_conversion(self, event: Event, src_modal: str, tgt_modal: str):
        """
        处理检测到的跨模态转换
        
        Args:
            event: 触发事件
            src_modal: 源模态
            tgt_modal: 目标模态
        """
        logger.warning(f"🔴 检测到跨模态转换: {src_modal} -> {tgt_modal}")
        
        # 如果actor被污染，自动传播到转换产物
        if event.actor and self.memory.check_taint(event.actor):
            if event.target:
                self.memory.propagate_taint(
                    event.actor, 
                    event.target, 
                    f"modal_conversion_{src_modal}_to_{tgt_modal}"
                )
                
                logger.critical(
                    f"🚨 污染的跨模态转换: {event.actor} -> {event.target} "
                    f"({src_modal} 转 {tgt_modal})"
                )
    
    def _infer_node_type(self, event: Event) -> str:
        """
        根据事件推断节点类型
        
        Args:
            event: 事件
            
        Returns:
            节点类型
        """
        if event.event_type in ["file_write", "file_create", "file_save"]:
            return "File"
        elif event.event_type in ["app_open", "process_start"]:
            return "App"
        elif event.event_type in ["url_open", "http_request"]:
            return "Website"
        else:
            return "Entity"
    
    def get_taint_summary(self) -> Dict:
        """
        获取污点追踪摘要
        
        Returns:
            污点统计信息
        """
        tainted_nodes = self.memory.query_nodes(properties={"is_tainted": True})
        
        summary = {
            "total_tainted": len(tainted_nodes),
            "by_type": {},
            "recent_propagations": []
        }
        
        for node in tainted_nodes:
            node_type = node.node_type
            summary["by_type"][node_type] = summary["by_type"].get(node_type, 0) + 1
        
        return summary
