"""
共享感知层 - Event Bus
统一的事件总线系统，负责接收和分发所有原始事件
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Callable, Optional
from datetime import datetime
from queue import Queue, Empty
import threading
import logging

logger = logging.getLogger(__name__)


@dataclass
class Event:
    """事件对象，表示系统中发生的一个动作"""
    
    event_type: str  # 事件类型: user_click, file_write, app_open, file_upload等
    actor: str  # 动作执行者: userA, processB等
    target: Optional[str] = None  # 动作目标: file_path, app_name, url等
    metadata: Dict[str, Any] = field(default_factory=dict)  # 额外信息
    timestamp: datetime = field(default_factory=datetime.now)
    event_id: Optional[str] = None  # 事件唯一ID
    
    def __post_init__(self):
        if self.event_id is None:
            self.event_id = f"{self.event_type}_{self.timestamp.timestamp()}"
    
    def __repr__(self):
        return f"Event({self.actor}, {self.event_type}, {self.target})"


class EventBus:
    """
    事件总线 - Agent的感知层
    
    核心功能：
    1. 接收所有原始事件（OCR识别、系统日志等）
    2. 维护事件队列，确保事件按序处理
    3. 支持事件订阅机制，让N2和N3引擎订阅感兴趣的事件
    4. 提供事件历史查询功能
    """
    
    def __init__(self, max_history: int = 10000):
        self.event_queue = Queue()
        self.subscribers: Dict[str, List[Callable]] = {}  # 事件类型 -> 处理器列表
        self.event_history: List[Event] = []  # 事件历史记录
        self.max_history = max_history
        self.running = False
        self.worker_thread = None
        self._lock = threading.Lock()
        
        logger.info("事件总线已初始化")
    
    def publish(self, event: Event):
        """
        发布一个新事件到总线
        
        Args:
            event: Event对象
        """
        logger.debug(f"发布事件: {event}")
        self.event_queue.put(event)
        
        # 记录到历史
        with self._lock:
            self.event_history.append(event)
            if len(self.event_history) > self.max_history:
                self.event_history.pop(0)
    
    def subscribe(self, event_type: str, handler: Callable[[Event], None]):
        """
        订阅特定类型的事件
        
        Args:
            event_type: 事件类型（如 "file_write", "app_open"）或 "*" 表示所有事件
            handler: 事件处理回调函数
        """
        if event_type not in self.subscribers:
            self.subscribers[event_type] = []
        self.subscribers[event_type].append(handler)
        logger.info(f"已订阅事件类型: {event_type}")
    
    def unsubscribe(self, event_type: str, handler: Callable[[Event], None]):
        """取消订阅"""
        if event_type in self.subscribers:
            self.subscribers[event_type].remove(handler)
    
    def start(self):
        """启动事件总线的处理线程"""
        if self.running:
            logger.warning("事件总线已在运行中")
            return
        
        self.running = True
        self.worker_thread = threading.Thread(target=self._process_events, daemon=True)
        self.worker_thread.start()
        logger.info("事件总线已启动")
    
    def stop(self):
        """停止事件总线"""
        self.running = False
        if self.worker_thread:
            self.worker_thread.join(timeout=5)
        logger.info("事件总线已停止")
    
    def _process_events(self):
        """
        事件处理主循环
        从队列中取出事件，分发给所有订阅者
        """
        while self.running:
            try:
                event = self.event_queue.get(timeout=1)
                self._dispatch_event(event)
            except Empty:
                continue
            except Exception as e:
                logger.error(f"处理事件时出错: {e}", exc_info=True)
    
    def _dispatch_event(self, event: Event):
        """
        将事件分发给订阅者
        
        Args:
            event: 待分发的事件
        """
        # 分发给特定类型的订阅者
        if event.event_type in self.subscribers:
            for handler in self.subscribers[event.event_type]:
                try:
                    handler(event)
                except Exception as e:
                    logger.error(f"{event.event_type}处理器错误: {e}", exc_info=True)
        
        # 分发给通配符订阅者
        if "*" in self.subscribers:
            for handler in self.subscribers["*"]:
                try:
                    handler(event)
                except Exception as e:
                    logger.error(f"通配符处理器错误: {e}", exc_info=True)
    
    def query_history(
        self, 
        event_type: Optional[str] = None,
        actor: Optional[str] = None,
        time_range: Optional[tuple] = None,
        limit: int = 100
    ) -> List[Event]:
        """
        查询事件历史
        
        Args:
            event_type: 筛选特定事件类型
            actor: 筛选特定执行者
            time_range: 时间范围 (start_time, end_time)
            limit: 返回结果数量限制
            
        Returns:
            符合条件的事件列表
        """
        with self._lock:
            results = self.event_history.copy()
        
        # 应用筛选条件
        if event_type:
            results = [e for e in results if e.event_type == event_type]
        
        if actor:
            results = [e for e in results if e.actor == actor]
        
        if time_range:
            start_time, end_time = time_range
            results = [e for e in results if start_time <= e.timestamp <= end_time]
        
        # 返回最新的limit条
        return results[-limit:]
    
    def get_stats(self) -> Dict[str, Any]:
        """获取事件总线统计信息"""
        with self._lock:
            total_events = len(self.event_history)
            event_type_counts = {}
            for event in self.event_history:
                event_type_counts[event.event_type] = event_type_counts.get(event.event_type, 0) + 1
        
        return {
            "total_events": total_events,
            "queue_size": self.event_queue.qsize(),
            "subscribers_count": sum(len(handlers) for handlers in self.subscribers.values()),
            "event_type_distribution": event_type_counts
        }
