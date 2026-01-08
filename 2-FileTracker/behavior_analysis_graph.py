# behavior_analysis_graph.py
"""
基于 LangGraph 的隐藏行为分析工作流
"""

import os
import json
from typing import Dict, Any
from datetime import datetime
from dotenv import load_dotenv

from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from behavior_analysis_state import BehaviorAnalysisState
from behavior_analysis_tools import analyze_frame_behavior, extract_hidden_operations
from behavior_analysis_prompts import (
    BEHAVIOR_ANALYSIS_SYSTEM_PROMPT,
    get_extract_hidden_operations_prompt
)
from worklist_manager import WorklistManager, SensitiveFileEvent

load_dotenv()


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
        
        # 初始化 LLM
        self.llm = ChatOpenAI(
            model=os.getenv("MODEL_NAME", "gpt-4"),
            base_url=os.getenv("OPENAI_BASE_URL"),
            api_key=os.getenv("OPENAI_API_KEY"),
            temperature=float(os.getenv("TEMPERATURE", "0.01")),
            streaming=False
        )
        
        # 构建图
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """
        构建 LangGraph 工作流
        """
        workflow = StateGraph(BehaviorAnalysisState)
        
        # 添加节点
        workflow.add_node("initialize", self._initialize_node)
        workflow.add_node("analyze_frames", self._analyze_frames_node)
        workflow.add_node("extract_operations", self._extract_operations_node)
        workflow.add_node("create_new_events", self._create_new_events_node)
        workflow.add_node("update_worklist", self._update_worklist_node)
        workflow.add_node("finalize", self._finalize_node)
        
        # 设置入口
        workflow.set_entry_point("initialize")
        
        # 添加边
        workflow.add_edge("initialize", "analyze_frames")
        workflow.add_edge("analyze_frames", "extract_operations")
        workflow.add_conditional_edges(
            "extract_operations",
            self._should_create_events,
            {
                "create_events": "create_new_events",
                "skip": "finalize"
            }
        )
        workflow.add_edge("create_new_events", "update_worklist")
        workflow.add_edge("update_worklist", "finalize")
        workflow.add_edge("finalize", END)
        
        return workflow.compile()
    
    # ==================== 辅助方法 ====================
    
    def _infer_full_path(self, base_dir: str, filename: str) -> str:
        """
        推断并标准化文件的完整路径
        
        Args:
            base_dir: 基础目录路径
            filename: 文件名（可能是完整路径或仅文件名）
            
        Returns:
            标准化后的完整路径
        """
        # 如果已经是绝对路径，直接使用
        if os.path.isabs(filename) or filename.startswith('/'):
            full_path = filename
        else:
            # 拼接基础目录和文件名
            full_path = os.path.join(base_dir, filename) if base_dir else filename
        
        # 标准化路径：统一使用 / 分隔符，去除多余的斜杠
        full_path = full_path.replace("\\", "/")
        full_path = full_path.replace("//", "/")
        
        return full_path
    
    def _find_file_path_in_logs(self, filename: str, time_range: str, log_events: list) -> str:
        """
        从日志中查找文件的实际完整路径
        
        单次遍历策略（高效）：
        1. 遍历日志时记录：完全匹配（带扩展名）和不带扩展名匹配
        2. 找到完全匹配立即返回
        3. 遍历完没有完全匹配，返回不带扩展名匹配（如果有）
        
        Args:
            filename: 文件名（不含路径）
            time_range: 时间范围字符串（如 "2026-01-05 10:00:00 - 2026-01-05 10:10:00"）
            log_events: 日志事件列表
            
        Returns:
            找到的完整文件路径，如果未找到则返回空字符串
        """
        if not log_events:
            return ""
        
        # 解析时间范围
        start_time, end_time = datetime.min, datetime.max
        try:
            time_parts = time_range.split(" - ")
            if len(time_parts) == 2:
                start_time = datetime.strptime(time_parts[0].strip(), "%Y-%m-%d %H:%M:%S")
                end_time = datetime.strptime(time_parts[1].strip(), "%Y-%m-%d %H:%M:%S")
        except:
            pass
        
        # 预计算目标文件的不带扩展名版本
        target_name_no_ext = os.path.splitext(filename)[0]
        
        # 记录候选匹配（用于不带扩展名的情况）
        no_ext_match = None
        no_ext_in_range = False
        
        # 单次遍历日志
        for event in log_events:
            try:
                # 检查文件路径
                file_path = event.get("file_path", "")
                if not file_path:
                    continue
                
                event_filename = os.path.basename(file_path)
                
                # 完全匹配（带扩展名）- 找到立即返回
                if event_filename == filename:
                    # 判断是否在时间范围内
                    in_time_range = False
                    event_time_str = event.get("timestamp", "")
                    if event_time_str:
                        try:
                            event_time = datetime.strptime(event_time_str, "%Y-%m-%dT%H:%M:%S.%f")
                            in_time_range = start_time <= event_time <= end_time
                        except:
                            try:
                                event_time = datetime.strptime(event_time_str, "%Y-%m-%d %H:%M:%S")
                                in_time_range = start_time <= event_time <= end_time
                            except:
                                pass
                    
                    range_info = "在时间范围内" if in_time_range else "不在时间范围内"
                    print(f"      ✅ 带扩展名匹配文件路径（{range_info}）")
                    return file_path
                
                # 记录不带扩展名的匹配（作为备选）
                if not no_ext_match:
                    event_name_no_ext = os.path.splitext(event_filename)[0]
                    if event_name_no_ext == target_name_no_ext:
                        no_ext_match = file_path
                        
                        # 判断是否在时间范围内
                        event_time_str = event.get("timestamp", "")
                        if event_time_str:
                            try:
                                event_time = datetime.strptime(event_time_str, "%Y-%m-%dT%H:%M:%S.%f")
                                no_ext_in_range = start_time <= event_time <= end_time
                            except:
                                try:
                                    event_time = datetime.strptime(event_time_str, "%Y-%m-%d %H:%M:%S")
                                    no_ext_in_range = start_time <= event_time <= end_time
                                except:
                                    pass
                        
            except Exception as e:
                continue
        
        # 遍历完成，如果有不带扩展名的匹配就返回
        if no_ext_match:
            range_info = "在时间范围内" if no_ext_in_range else "不在时间范围内"
            print(f"      ✅ 不带扩展名匹配文件路径（{range_info}）")
            return no_ext_match
        
        return ""
    
    # ==================== 运行主要方法 ====================
    
    def analyze_event(
        self,
        event: SensitiveFileEvent,
        index_path: str,
        video_path: str,
        log_events: list = None
    ) -> Dict[str, Any]:
        """
        分析单个敏感事件的隐藏行为
        
        Args:
            event: 敏感文件事件
            index_path: INDEX.md 路径
            video_path: 视频路径
            log_events: 日志事件列表（用于查找跨目录文件的实际路径）
            
        Returns:
            分析结果
        """
        print(f"\n🔍 [BehaviorAnalysis] 开始分析事件: {event.event_id}")
        
        # 初始化状态
        initial_state = {
            "current_event": event,
            "index_path": index_path,
            "video_path": video_path,
            "log_events": log_events or [],  # 添加日志事件列表
            "frame_analysis_result": None,
            "hidden_operations": [],
            "file_mappings": [],
            "new_events": [],
            "has_hidden_behavior": False,
            "analysis_complete": False,
            "error_message": None,
            "messages": []
        }

        graph_png = self.graph.get_graph().draw_mermaid_png()
        with open("behavior_analysis_graph.png", "wb") as f:
            f.write(graph_png)
        
        # 运行图
        result = self.graph.invoke(initial_state)
        
        return result
    
    # ==================== 节点实现 ====================
    
    def _initialize_node(self, state: BehaviorAnalysisState) -> Dict[str, Any]:
        """
        初始化节点
        """
        print("\n🚀 [Initialize] 初始化隐藏行为分析...")
        
        event = state["current_event"]
        print(f"   - 事件ID: {event.event_id}")
        print(f"   - 文件: {event.current_file}")
        print(f"   - 时间: {event.timestamp}")
        print(f"   - 事件类型: {event.event_type}")
        
        # 初始化消息
        messages = [
            SystemMessage(content=BEHAVIOR_ANALYSIS_SYSTEM_PROMPT)
        ]
        
        return {
            "messages": messages
        }
    
    def _analyze_frames_node(self, state: BehaviorAnalysisState) -> Dict[str, Any]:
        """
        分析视频帧节点
        """
        print("\n🎬 [AnalyzeFrames] 调用模块1分析视频帧...")
        
        event = state["current_event"]
        
        try:
            # 调用工具分析帧
            result = analyze_frame_behavior.invoke({
                "event_timestamp": event.timestamp,
                "current_file": event.current_file,
                "index_path": state["index_path"],
                "video_path": state["video_path"],
                "search_duration": 30
            })
            
            print(f"   ✅ 帧分析完成")
            
            return {
                "frame_analysis_result": result
            }
            
        except Exception as e:
            error_msg = f"帧分析失败: {str(e)}"
            print(f"   ❌ {error_msg}")
            return {
                "frame_analysis_result": None,
                "error_message": error_msg,
                "analysis_complete": True
            }
    
    def _extract_operations_node(self, state: BehaviorAnalysisState) -> Dict[str, Any]:
        """
        提取隐藏操作节点
        """
        print("\n🔎 [ExtractOperations] 提取隐藏操作...")
        
        frame_result = state.get("frame_analysis_result")
        
        if not frame_result or frame_result.get("total_events", 0) == 0:
            print("   ⚠️ 没有发现相关事件")
            return {
                "has_hidden_behavior": False,
                "hidden_operations": [],
                "file_mappings": []
            }
        
        try:
            # 调用工具提取隐藏操作
            extraction_result = extract_hidden_operations.invoke({
                "frame_analysis_result": frame_result
            })
            
            has_hidden = extraction_result.get("has_hidden_behavior", False)
            operations = extraction_result.get("hidden_operations", [])
            mappings = extraction_result.get("file_mappings", [])
            
            print(f"   ✅ 提取完成: 发现 {len(operations)} 个隐藏操作")
            
            return {
                "has_hidden_behavior": has_hidden,
                "hidden_operations": operations,
                "file_mappings": mappings
            }
            
        except Exception as e:
            error_msg = f"提取隐藏操作失败: {str(e)}"
            print(f"   ❌ {error_msg}")
            return {
                "has_hidden_behavior": False,
                "hidden_operations": [],
                "file_mappings": [],
                "error_message": error_msg
            }
    
    def _create_new_events_node(self, state: BehaviorAnalysisState) -> Dict[str, Any]:
        """
        创建新的敏感事件节点
        """
        print("\n📝 [CreateNewEvents] 创建新的敏感事件...")
        
        operations = state.get("hidden_operations", [])
        current_event = state["current_event"]
        log_events = state.get("log_events", [])
        new_events = []
        
        for op in operations:
            # 从模块1返回的是文件名，需要推断完整路径
            new_filename = op["new_file"]
            operation_type = op["operation_type"]
            time_range = op.get("time_range", "")
            
            # 获取当前文件的目录
            current_dir = os.path.dirname(current_event.current_file)
            
            # 方案1：先尝试从日志中查找文件的实际路径（处理跨目录情况）
            new_file_path = ""
            if log_events:
                print(f"   🔍 在 {len(log_events)} 条日志中搜索文件: {new_filename}")
                print(f"   🔍 时间范围: {time_range}")
                new_file_path = self._find_file_path_in_logs(new_filename, time_range, log_events)
                if new_file_path:
                    print(f"   📍 从日志找到实际路径: {new_file_path}")
                else:
                    print(f"   ⚠️ 日志中未找到匹配的文件路径")
            else:
                print(f"   ⚠️ log_events 为空，无法从日志查找")
            
            # 方案2：如果日志中未找到，则使用原目录推断（同目录情况）
            if not new_file_path:
                new_file_path = self._infer_full_path(current_dir, new_filename)
                print(f"   📍 推断路径（同目录）: {new_file_path}")
            
            # 使用隐藏行为的实际发生时间（从involved_timestamps取中间那个）
            involved_timestamps = op.get("involved_timestamps", [])
            if involved_timestamps:
                # 使用中间时间戳作为事件发生时间
                mid_index = len(involved_timestamps) // 2
                timestamp_str = involved_timestamps[mid_index]
                # 转换为ISO格式（如果需要）
                try:
                    dt = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S")
                    timestamp = dt.strftime("%Y-%m-%dT%H:%M:%S.%f")
                except:
                    timestamp = timestamp_str if 'T' in timestamp_str else timestamp_str.replace(' ', 'T') + '.000000'
            else:
                # 如果没有时间戳，使用原事件的时间戳
                timestamp = current_event.timestamp
            
            event_id = f"derived_{operation_type}_{hash(new_file_path)}_{timestamp}"
            
            new_event = SensitiveFileEvent(
                event_id=event_id,
                original_file=current_event.original_file,  # 追溯到原始文件
                current_file=new_file_path,  # 使用完整路径
                event_type=f"derived_from_{operation_type}",
                process_info={"app_name": op["app_name"]},
                timestamp=timestamp,
                is_hidden=True,
                raw_event={
                    "operation_type": operation_type,
                    "description": op["description"],
                    "time_range": op["time_range"]
                }
            )
            
            new_events.append(new_event)
            print(f"   ✅ 创建事件: {new_file_path} (来自 {operation_type})")
        
        return {
            "new_events": new_events
        }
    
    def _update_worklist_node(self, state: BehaviorAnalysisState) -> Dict[str, Any]:
        """
        更新 worklist 节点
        """
        print("\n📋 [UpdateWorklist] 更新 worklist...")
        
        new_events = state.get("new_events", [])
        file_mappings = state.get("file_mappings", [])
        current_event = state["current_event"]
        current_dir = os.path.dirname(current_event.current_file)
        
        # 添加新事件到 worklist 并将派生文件加入敏感文件列表
        new_sensitive_files = []
        for event in new_events:
            self.worklist_manager.add_event(event)
            print(f"   ✅ 添加到 worklist: {event.current_file}")
            
            # 将派生文件加入敏感文件列表
            if event.current_file not in self.worklist_manager.sensitive_files:
                self.worklist_manager.add_sensitive_file(event.current_file)
                new_sensitive_files.append(event.current_file)
                print(f"   ✅ 添加到敏感文件列表: {event.current_file}")
            
            # 直接使用 new_event 中已推断好的路径更新文件映射
            # original_file: 追溯的原始文件路径
            # current_file: 派生后的新文件路径（可能来自日志查找或同目录推断）
            self.worklist_manager.update_file_mapping(
                original_file=event.original_file,
                new_file=event.current_file
            )
            print(f"   ✅ 添加映射: {event.original_file} -> {event.current_file}")
        
        # 如果有新的敏感文件，重新扫描日志
        if new_sensitive_files:
            print(f"\n   🔄 重新扫描日志以查找新敏感文件的相关操作...")
            # 从 state 中获取原始日志（如果有的话）
            # 注意：这需要在初始化时将日志数据传入state
            # TODO:这里先输出提示，具体实现需要调整工作流
            print(f"   ⚠️ 提示：发现 {len(new_sensitive_files)} 个新敏感文件，建议重新扫描完整日志")
        
        return {}
    
    def _finalize_node(self, state: BehaviorAnalysisState) -> Dict[str, Any]:
        """
        完成节点
        """
        print("\n✅ [Finalize] 分析完成")
        
        has_hidden = state.get("has_hidden_behavior", False)
        operations_count = len(state.get("hidden_operations", []))
        new_events_count = len(state.get("new_events", []))
        
        print(f"   - 发现隐藏行为: {'是' if has_hidden else '否'}")
        print(f"   - 隐藏操作数: {operations_count}")
        print(f"   - 新增事件数: {new_events_count}")
        
        return {
            "analysis_complete": True
        }
    
    # ==================== 条件判断 ====================
    
    def _should_create_events(self, state: BehaviorAnalysisState) -> str:
        """
        判断是否需要创建新事件
        """
        if state.get("has_hidden_behavior", False):
            return "create_events"
        else:
            return "skip"


# ==================== 便捷函数 ====================

def analyze_sensitive_event_behavior(
    event: SensitiveFileEvent,
    index_path: str,
    video_path: str,
    worklist_manager: WorklistManager,
    log_events: list = None
) -> Dict[str, Any]:
    """
    分析单个敏感事件的隐藏行为（便捷函数）
    
    Args:
        event: 敏感文件事件
        index_path: INDEX.md 路径
        video_path: 视频路径
        worklist_manager: Worklist 管理器
        log_events: 日志事件列表（用于查找跨目录文件的实际路径）
        
    Returns:
        分析结果
    """
    graph = BehaviorAnalysisGraph(worklist_manager)
    return graph.analyze_event(event, index_path, video_path, log_events)
