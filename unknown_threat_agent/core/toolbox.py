"""
共享行动层 - Toolbox
统一的工具接口，提供Agent可以调用的所有原子能力
"""

from typing import Any, Dict, List, Optional, Callable
from dataclasses import dataclass
from abc import ABC, abstractmethod
import logging
import subprocess
import time
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class ToolResult:
    """工具执行结果"""
    success: bool
    data: Any
    error: Optional[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class Tool(ABC):
    """工具基类"""
    
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
    
    @abstractmethod
    def execute(self, **kwargs) -> ToolResult:
        """执行工具"""
        pass


class VideoSearchTool(Tool):
    """
    视频搜索工具 - 对接需求1的视频搜索引擎
    调用ViT大模型分析UI界面
    """
    
    def __init__(self, search_engine_path: Optional[str] = None):
        super().__init__(
            name="video_search",
            description="使用ViT模型搜索和分析视频帧"
        )
        self.search_engine_path = search_engine_path or "../deformed_image_search"
    
    def execute(
        self, 
        video_path: Optional[str] = None,
        time_range: Optional[tuple] = None,
        prompt: Optional[str] = None,
        **kwargs
    ) -> ToolResult:
        """
        执行视频搜索
        
        Args:
            video_path: 视频文件路径
            time_range: 时间范围 (start, end)
            prompt: 搜索提示词（如 "upload button", "text to audio"）
            
        Returns:
            ToolResult包含搜索结果
        """
        try:
            logger.info(f"执行视频搜索，提示词: {prompt}")
            
            # 这里应该调用需求1的实际接口
            # 目前返回模拟结果
            # TODO: 集成实际的视频搜索引擎
            
            result_data = {
                "found_ui_elements": [],
                "screenshots": [],
                "confidence": 0.0
            }
            
            # 模拟：如果提示词包含关键词，则返回发现结果
            if prompt:
                if any(keyword in prompt.lower() for keyword in ["upload", "convert", "audio", "text"]):
                    result_data = {
                        "found_ui_elements": ["text_to_audio_button", "upload_interface"],
                        "screenshots": ["frame_1234.png"],
                        "confidence": 0.89
                    }
            
            return ToolResult(
                success=True,
                data=result_data,
                metadata={"tool": "video_search", "prompt": prompt}
            )
        
        except Exception as e:
            logger.error(f"视频搜索失败: {e}", exc_info=True)
            return ToolResult(success=False, data=None, error=str(e))


class ProcessMonitorTool(Tool):
    """进程监控工具"""
    
    def __init__(self):
        super().__init__(
            name="monitor_process",
            description="监控进程活动和文件操作"
        )
        self.monitoring_tasks: Dict[str, Dict] = {}
    
    def execute(
        self,
        action: str = "start",  # start, stop, query
        user: Optional[str] = None,
        process_name: Optional[str] = None,
        file_type: Optional[List[str]] = None,
        **kwargs
    ) -> ToolResult:
        """
        执行进程监控
        
        Args:
            action: 监控动作 (start, stop, query)
            user: 监控的用户
            process_name: 监控的进程名
            file_type: 监控的文件类型列表
            
        Returns:
            ToolResult
        """
        try:
            if action == "start":
                task_id = f"{user}_{process_name}_{int(time.time())}"
                self.monitoring_tasks[task_id] = {
                    "user": user,
                    "process": process_name,
                    "file_types": file_type or [],
                    "started_at": time.time(),
                    "events": []
                }
                logger.info(f"已启动监控: {task_id}")
                return ToolResult(
                    success=True,
                    data={"task_id": task_id},
                    metadata={"action": "start"}
                )
            
            elif action == "stop":
                task_id = kwargs.get("task_id")
                if task_id in self.monitoring_tasks:
                    del self.monitoring_tasks[task_id]
                    logger.info(f"已停止监控: {task_id}")
                return ToolResult(success=True, data={"stopped": task_id})
            
            elif action == "query":
                return ToolResult(
                    success=True,
                    data={"active_tasks": list(self.monitoring_tasks.keys())}
                )
            
        except Exception as e:
            logger.error(f"进程监控失败: {e}", exc_info=True)
            return ToolResult(success=False, data=None, error=str(e))


class FileOperationTool(Tool):
    """文件操作工具"""
    
    def __init__(self):
        super().__init__(
            name="file_operation",
            description="监控和分析文件操作"
        )
    
    def execute(
        self,
        operation: str,  # check_exists, get_metadata, monitor_creation
        file_path: Optional[str] = None,
        **kwargs
    ) -> ToolResult:
        """
        执行文件操作
        
        Args:
            operation: 操作类型
            file_path: 文件路径
            
        Returns:
            ToolResult
        """
        try:
            if operation == "check_exists":
                exists = Path(file_path).exists() if file_path else False
                return ToolResult(
                    success=True,
                    data={"exists": exists, "path": file_path}
                )
            
            elif operation == "get_metadata":
                if file_path and Path(file_path).exists():
                    stat = Path(file_path).stat()
                    return ToolResult(
                        success=True,
                        data={
                            "size": stat.st_size,
                            "created": stat.st_ctime,
                            "modified": stat.st_mtime,
                            "extension": Path(file_path).suffix
                        }
                    )
                return ToolResult(success=False, data=None, error="File not found")
            
            elif operation == "monitor_creation":
                # 在实际应用中，这里应该设置文件系统监控
                logger.info(f"监控文件创建: {kwargs}")
                return ToolResult(
                    success=True,
                    data={"monitoring": True},
                    metadata=kwargs
                )
        
        except Exception as e:
            logger.error(f"文件操作失败: {e}", exc_info=True)
            return ToolResult(success=False, data=None, error=str(e))


class NetworkMonitorTool(Tool):
    """网络监控工具"""
    
    def __init__(self):
        super().__init__(
            name="network_monitor",
            description="监控网络活动和连接"
        )
    
    def execute(
        self,
        action: str,  # check_connection, monitor_upload, analyze_url
        url: Optional[str] = None,
        **kwargs
    ) -> ToolResult:
        """
        执行网络监控
        
        Args:
            action: 监控动作
            url: URL地址
            
        Returns:
            ToolResult
        """
        try:
            if action == "analyze_url":
                # 简单的URL分析
                is_suspicious = False
                if url:
                    suspicious_keywords = ["convert", "upload", "anonymous", "temp"]
                    is_suspicious = any(kw in url.lower() for kw in suspicious_keywords)
                
                return ToolResult(
                    success=True,
                    data={
                        "url": url,
                        "suspicious": is_suspicious,
                        "category": "unknown"
                    }
                )
            
            elif action == "monitor_upload":
                logger.info(f"监控上传活动: {kwargs}")
                return ToolResult(success=True, data={"monitoring": True})
        
        except Exception as e:
            logger.error(f"网络监控失败: {e}", exc_info=True)
            return ToolResult(success=False, data=None, error=str(e))


class LLMReasoningTool(Tool):
    """
    LLM推理工具 - 用于N3的推理（Reason）部分
    调用大语言模型进行威胁分析和假设生成
    """
    
    def __init__(self, model_name: str = "qwen", api_key: Optional[str] = None):
        super().__init__(
            name="llm_reasoning",
            description="使用LLM进行威胁分析和假设生成"
        )
        self.model_name = model_name
        self.api_key = api_key
    
    def execute(
        self,
        context: str,
        question: str,
        **kwargs
    ) -> ToolResult:
        """
        执行LLM推理
        
        Args:
            context: 上下文信息
            question: 推理问题
            
        Returns:
            ToolResult包含推理结果
        """
        try:
            logger.info(f"LLM推理问题: {question}")
            
            # ========== 真实LLM API调用 ==========
            if self.api_key and self.api_key.strip():
                try:
                    import dashscope
                    from dashscope import Generation
                    
                    # 构建中文提示词
                    prompt = f"""你是一个专业的网络安全威胁分析专家。请分析以下上下文信息，判断是否存在安全威胁。

上下文信息：
{context}

分析问题：
{question}

请以JSON格式返回分析结果，包含以下字段：
- hypothesis: 你的假设（用中文描述可能的威胁场景）
- confidence: 置信度（0-1之间的数字）
- suggested_actions: 建议采取的行动（中文字符串列表，至少3条）
- threat_level: 威胁等级（low/medium/high之一）

只返回JSON格式数据，不要包含其他说明文字。"""
                    
                    logger.info(f"调用通义千问API，模型: {self.model_name}")
                    
                    # 调用通义千问API
                    dashscope.api_key = self.api_key
                    response = Generation.call(
                        model=self.model_name,
                        prompt=prompt,
                        max_tokens=1500,
                        temperature=0.7,
                        top_p=0.8
                    )
                    
                    # 检查响应
                    if response.status_code == 200:
                        result_text = response.output.text.strip()
                        logger.debug(f"LLM原始响应长度: {len(result_text)}字符")
                        
                        # 解析JSON响应
                        import json
                        import re
                        
                        # 提取JSON（可能在代码块中）
                        json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', result_text, re.DOTALL)
                        if json_match:
                            json_str = json_match.group(0)
                            reasoning_result = json.loads(json_str)
                            logger.info(f"LLM推理成功，假设: {reasoning_result.get('hypothesis', '')[:80]}...")
                        else:
                            # 无法解析JSON，使用文本内容
                            reasoning_result = {
                                "hypothesis": result_text[:200] if result_text else "无法获取分析结果",
                                "confidence": 0.6,
                                "suggested_actions": ["继续监控", "收集更多证据", "人工复查"],
                                "threat_level": "medium"
                            }
                            logger.warning("LLM返回内容无法解析为JSON，使用默认结构")
                        
                        return ToolResult(
                            success=True,
                            data=reasoning_result,
                            metadata={
                                "model": self.model_name,
                                "prompt_length": len(prompt),
                                "api_used": True,
                                "response_length": len(result_text)
                            }
                        )
                    else:
                        logger.error(f"通义千问API调用失败: {response.code} - {response.message}")
                        raise Exception(f"API返回错误: {response.code}")
                        
                except ImportError as e:
                    logger.warning(f"dashscope库未安装，回退到模拟模式。安装命令: pip install dashscope")
                except Exception as api_error:
                    logger.warning(f"LLM API调用异常，回退到模拟模式: {str(api_error)[:100]}")
            else:
                logger.info("未配置API密钥或密钥为空，使用模拟模式")
            
            # ========== 模拟模式（回退方案） ==========
            logger.info("使用规则基础的模拟推理")
            
            # 基于上下文关键词的智能推理
            context_lower = context.lower()
            threat_level = "low"
            hypothesis = "正常行为，未发现明显威胁迹象"
            suggested_actions = ["继续常规监控", "记录行为日志"]
            confidence = 0.4
            
            # 规则1: 污点用户访问未知资源
            if "tainted" in context_lower and ("unknown" in context_lower or "suspicious" in context_lower):
                threat_level = "medium"
                hypothesis = "污点用户访问未知或可疑资源，存在潜在数据外泄风险"
                suggested_actions = [
                    "调用视频搜索工具分析界面元素",
                    "监控该用户的后续文件操作",
                    "追踪污点传播路径"
                ]
                confidence = 0.65
            
            # 规则2: 跨模态转换（高危）
            if "tainted" in context_lower and any(kw in context_lower for kw in ["audio", "convert", "modal", "transform"]):
                threat_level = "high"
                hypothesis = "检测到污点数据的跨模态转换行为，疑似使用文本转音频等方式规避检测"
                suggested_actions = [
                    "立即监控音频或图像文件的创建",
                    "标记所有转换产物为污点",
                    "追踪文件的后续操作（上传、分享等）",
                    "准备生成威胁告警"
                ]
                confidence = 0.80
            
            # 规则3: 污点数据外泄（极高危）
            if "tainted" in context_lower and ("upload" in context_lower or "external" in context_lower or "share" in context_lower):
                threat_level = "high"
                hypothesis = "检测到污点数据外泄行为，用户正在或已经将敏感数据传输到外部"
                suggested_actions = [
                    "立即阻断网络上传操作",
                    "记录完整的威胁链路径",
                    "生成高级别安全告警",
                    "通知安全运营中心",
                    "隔离受影响的用户账户"
                ]
                confidence = 0.90
            
            reasoning_result = {
                "hypothesis": hypothesis,
                "confidence": confidence,
                "suggested_actions": suggested_actions,
                "threat_level": threat_level
            }
            
            return ToolResult(
                success=True,
                data=reasoning_result,
                metadata={
                    "model": "rule_based_fallback",
                    "prompt_length": len(context),
                    "api_used": False
                }
            )
        
        except Exception as e:
            logger.error(f"LLM推理失败: {e}", exc_info=True)
            return ToolResult(success=False, data=None, error=str(e))


class Toolbox:
    """
    工具箱 - Agent的行动层
    
    提供统一接口访问所有工具
    """
    
    def __init__(self):
        self.tools: Dict[str, Tool] = {}
        self._register_default_tools()
        logger.info("工具箱已初始化")
    
    def _register_default_tools(self):
        """注册默认工具"""
        from config import AGENT_CONFIG
        
        self.register_tool(VideoSearchTool())
        self.register_tool(ProcessMonitorTool())
        self.register_tool(FileOperationTool())
        self.register_tool(NetworkMonitorTool())
        
        # 获取API密钥并传入LLM工具
        api_key = AGENT_CONFIG["n3_engine"]["llm_api_key"]
        model_name = AGENT_CONFIG["n3_engine"]["llm_model"]
        self.register_tool(LLMReasoningTool(api_key=api_key, model_name=model_name))
    
    def register_tool(self, tool: Tool):
        """
        注册一个新工具
        
        Args:
            tool: Tool对象
        """
        self.tools[tool.name] = tool
        logger.info(f"已注册工具: {tool.name}")
    
    def call(self, tool_name: str, **kwargs) -> ToolResult:
        """
        调用工具
        
        Args:
            tool_name: 工具名称
            **kwargs: 工具参数
            
        Returns:
            ToolResult
        """
        if tool_name not in self.tools:
            logger.error(f"未找到工具: {tool_name}")
            return ToolResult(
                success=False,
                data=None,
                error=f"Tool '{tool_name}' not registered"
            )
        
        tool = self.tools[tool_name]
        logger.debug(f"调用工具: {tool_name}，参数: {kwargs}")
        
        try:
            result = tool.execute(**kwargs)
            return result
        except Exception as e:
            logger.error(f"工具执行错误: {e}", exc_info=True)
            return ToolResult(success=False, data=None, error=str(e))
    
    def list_tools(self) -> List[Dict[str, str]]:
        """
        列出所有可用工具
        
        Returns:
            工具信息列表
        """
        return [
            {"name": tool.name, "description": tool.description}
            for tool in self.tools.values()
        ]
    
    def get_tool(self, tool_name: str) -> Optional[Tool]:
        """获取工具对象"""
        return self.tools.get(tool_name)
