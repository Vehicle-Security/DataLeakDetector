# behavior_analysis_graph.py
"""
模块2 LangGraph图定义
基于 LangGraph 的隐藏行为分析工作流
"""

import os
import json
from typing import Dict, Any
from dotenv import load_dotenv, find_dotenv

from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI

from behavior_analysis_state import BehaviorAnalysisState
from behavior_analysis_nodes import (
    initialize_node,
    analyze_frames_node,
    extract_operations_node,
    create_new_events_node,
    update_worklist_node,
    finalize_node,
    should_create_events
)
from worklist_manager import WorklistManager, SensitiveFileEvent

load_dotenv(find_dotenv())


def _should_render_graph_debug() -> bool:
    """
    调试图渲染仅在显式开启时执行，避免运行时依赖外部 mermaid 服务。
    """
    return os.getenv("RENDER_BEHAVIOR_GRAPH_DEBUG", "").strip().lower() in {"1", "true", "yes", "on"}


class BehaviorAnalysisGraph:
    """
    隐藏行为分析工作流图
    """
    
    def __init__(self, worklist_manager: WorklistManager):
        """
        初始化分析图
        
        Args:
            worklist_manager: Worklist 管理器实例
        """
        self.worklist_manager = worklist_manager
        
        self.llm = ChatOpenAI(
            model=os.getenv("MODEL_NAME", "gpt-4"),
            base_url=os.getenv("OPENAI_BASE_URL"),
            api_key=os.getenv("OPENAI_API_KEY"),
            temperature=float(os.getenv("TEMPERATURE", "0.01")),
            streaming=False
        )
        print("===========使用的ai模型名称：", os.getenv("MODEL_NAME", "gpt-4"),"==========")
        
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """
        构建 LangGraph 工作流
        """
        workflow = StateGraph(BehaviorAnalysisState)
        
        # 使用 lambda 包装节点函数，传入 worklist_manager
        workflow.add_node("initialize", lambda state: initialize_node(state, self.worklist_manager))
        workflow.add_node("analyze_frames", lambda state: analyze_frames_node(state, self.worklist_manager))
        workflow.add_node("extract_operations", lambda state: extract_operations_node(state, self.worklist_manager))
        workflow.add_node("create_new_events", lambda state: create_new_events_node(state, self.worklist_manager))
        workflow.add_node("update_worklist", lambda state: update_worklist_node(state, self.worklist_manager))
        workflow.add_node("finalize", lambda state: finalize_node(state, self.worklist_manager))
        
        workflow.set_entry_point("initialize")
        
        workflow.add_edge("initialize", "analyze_frames")
        workflow.add_edge("analyze_frames", "extract_operations")
        workflow.add_conditional_edges(
            "extract_operations",
            should_create_events,
            {
                "create_events": "create_new_events",
                "skip": "finalize"
            }
        )
        workflow.add_edge("create_new_events", "update_worklist")
        workflow.add_edge("update_worklist", "finalize")
        workflow.add_edge("finalize", END)
        
        return workflow.compile()
    
    # ==================== 运行主要方法 ====================
    
    def analyze_event(
        self,
        event: SensitiveFileEvent,
        index_path: str,
        video_path: str,
        log_events: list = None,
        search_duration: int = 30
    ) -> Dict[str, Any]:
        """
        分析单个敏感事件的隐藏行为
        
        Args:
            event: 敏感文件事件
            index_path: INDEX.md 路径
            video_path: 视频路径
            log_events: 日志事件列表（用于查找跨目录文件的实际路径）
            search_duration: 视频搜索时长（秒），默认30秒
            
        Returns:
            分析结果
        """
        print(f"\n🔍 [BehaviorAnalysis] 开始分析事件: {event.event_id}")
        
        initial_state = {
            "current_event": event,
            "index_path": index_path,
            "video_path": video_path,
            "log_events": log_events or [],  # 添加日志事件列表
            "search_duration": search_duration,
            "frame_analysis_result": None,
            "hidden_operations": [],
            "file_mappings": [],
            "new_events": [],
            "has_hidden_behavior": False,
            "analysis_complete": False,
            "error_message": None,
            "messages": []
        }

        if _should_render_graph_debug():
            try:
                graph_png = self.graph.get_graph().draw_mermaid_png()
                with open("behavior_analysis_graph.png", "wb") as f:
                    f.write(graph_png)
            except Exception as exc:
                print(f"   ⚠️ 跳过行为分析图渲染: {exc}")
        
        result = self.graph.invoke(initial_state)
        
        return result


# ==================== graph便捷调用函数 ====================

def analyze_sensitive_event_behavior(
    event: SensitiveFileEvent,
    index_path: str,
    video_path: str,
    worklist_manager: WorklistManager,
    log_events: list = None,
    search_duration: int = 30
) -> Dict[str, Any]:
    """
    分析单个敏感事件的隐藏行为（便捷函数）
    
    Args:
        event: 敏感文件事件
        index_path: INDEX.md 路径
        video_path: 视频路径
        worklist_manager: Worklist 管理器
        log_events: 日志事件列表（用于查找跨目录文件的实际路径）
        search_duration: 视频搜索时长（秒），默认30秒
        
    Returns:
        分析结果
    """
    graph = BehaviorAnalysisGraph(worklist_manager)
    return graph.analyze_event(event, index_path, video_path, log_events, search_duration)
