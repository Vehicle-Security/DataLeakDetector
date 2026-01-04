# nodes.py
import json
import re
import os
from typing import Dict, Any, List
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from state import TrackerState, Resource, EvidenceChain
from tools import (
    extract_resource_from_description,
    trace_resource_flow,
)

# 加载环境变量
load_dotenv()

# 工具映射
tools_map = {
    "extract_resource_from_description": extract_resource_from_description,
    "trace_resource_flow": trace_resource_flow,
}

# 初始化 LLM
reasoning_llm = ChatOpenAI(
    model=os.getenv("MODEL_NAME"),
    base_url=os.getenv("OPENAI_BASE_URL"),
    api_key=os.getenv("OPENAI_API_KEY"),
    temperature=float(os.getenv("TEMPERATURE", "0.1")),
    streaming=False
).bind_tools(list(tools_map.values()))

SYSTEM_PROMPT = """
你是一名数据泄露防护（DLP）专家，负责追踪敏感资源在多个操作中的流转和变化。

【输入数据】
每个操作包含以下字段（已由 RiskSieve 识别）：
- operation_id: 操作唯一标识
- operation_type: 操作类型（如 file_access, file_compress, file_upload 等）
- resource_name: 主要涉及的资源名称
- app_name: 执行操作的应用
- start_time: 操作时间
- raw_description: 原始文字描述

【你的任务】
1. 识别资源之间的转换关系（如：将A压缩为B，A就是B的源文件）
2. 追踪资源的完整流转路径（从最初的源文件到最终的外发）
3. 识别风险指标（重命名、压缩、加密等隐蔽行为）
4. 构建清晰的证据链

【可用工具】

1. extract_resource_from_description - 从描述中提取资源转换关系
   参数: {"operation_description": "...", "operation_type": "..."}
   用途: 当操作涉及资源转换时（如压缩、重命名），提取源资源和目标资源
   示例: "将机密文件.pdf压缩为backup.zip" → 源: 机密文件.pdf, 目标: backup.zip

2. trace_resource_flow - 追踪资源的完整流转路径
   参数: {"target_resource": "资源名", "all_operations": [...], "max_depth": 10}
   用途: 从某个资源开始，向前追溯源头，向后追踪派生，构建完整链路
   示例: trace_resource_flow("backup.zip") → 
         源链: 机密文件.pdf → backup.zip
         派生链: backup.zip → (上传到云盘)

【工作流程建议】
1. 浏览所有操作，识别最终的外发操作（upload, send等）
2. 对于外发操作的资源，使用 trace_resource_flow 追踪其来源
3. 对于涉及转换的操作，使用 extract_resource_from_description 提取源资源
4. 基于工具返回的信息，构建完整的证据链

【重要原则】
- 优先使用结构化字段（operation_type, resource_name）
- 只在需要提取转换关系或追踪完整链路时才调用工具
- 分析要简洁高效，避免重复调用相同工具

【输出格式】
在分析过程中，按 ReAct 模式输出：
Thought: 当前需要确认什么信息
Action: 工具名称
Action Input: {"参数名": "参数值"}
Observation: (工具返回结果)

当完成所有操作分析后，输出：
Final Answer: {
    "tracked_resources": [
        {
            "resource_id": "res_001",
            "resource_name": "敏感文件.pdf",
            "resource_type": "document",
            "first_seen": "操作ID",
            "last_seen": "操作ID",
            "derived_resources": ["压缩包.zip", "重命名文件.pdf"]
        }
    ],
    "evidence_chains": [
        {
            "chain_id": "chain_001",
            "root_resource": "敏感文件.pdf",
            "operations": ["op_001", "op_002", "op_003"],
            "risk_indicators": ["file_obfuscation", "cross_application_transfer"]
        }
    ],
    "summary": "追踪分析总结"
}

注意：
- 不要与日志或视频直接交互，只分析 RiskSieve 的输出
- 重点关注资源的流转和变化，而非行为风险判断（风险判断由 ThreatHunter 完成）
- Action Input 必须是纯 JSON，不要用 Markdown 代码块
"""


def initialize_node(state: TrackerState) -> Dict[str, Any]:
    """
    初始化节点：设置初始状态和系统提示
    """
    print("\n🚀 [Initialize] EvidenceTracer 启动")
    
    input_ops = state["input_operations"]
    print(f"📥 接收到 {len(input_ops)} 个敏感操作片段")
    
    # 构建初始提示
    ops_summary = "【输入的敏感操作片段】\n"
    for i, op in enumerate(input_ops):
        ops_summary += f"\n操作 {i+1} [ID: {op['operation_id']}]:\n"
        ops_summary += f"  - 类型: {op['operation_type']}\n"
        ops_summary += f"  - 应用: {op['app_name']}\n"
        ops_summary += f"  - 资源: {op['resource_name']}\n"
        ops_summary += f"  - 时间: {op['start_time']}\n"
        ops_summary += f"  - 描述: {op['raw_description']}\n"
    
    initial_messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=ops_summary + "\n请开始分析这些操作，追踪资源流转。")
    ]
    
    return {
        "messages": initial_messages,
        "current_operation_index": 0,
        "tracked_resources": {},
        "evidence_chains": [],
        "analysis_results": [],
        "is_complete": False
    }


def reasoning_node(state: TrackerState) -> Dict[str, Any]:
    """
    推理节点：LLM 决定下一步行动或输出最终结果
    """
    print("\n🧠 [Reasoning] 分析资源追踪策略...")
    
    messages = state["messages"]
    
    # 调用 LLM 进行推理
    response = reasoning_llm.invoke(messages)
    
    print(f"💭 LLM 输出: {response.content[:200]}...")
    
    return {"messages": [response]}


def action_node(state: TrackerState) -> Dict[str, Any]:
    """
    执行节点：解析 LLM 输出并调用相应工具
    """
    print("\n⚡ [Action] 执行工具调用...")
    
    last_message = state["messages"][-1]
    
    # 检查是否有 tool_calls
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        tool_call = last_message.tool_calls[0]
        tool_name = tool_call['name']
        tool_args = tool_call['args']
        
        print(f"🔧 调用工具: {tool_name}")
        print(f"📝 参数: {json.dumps(tool_args, ensure_ascii=False)[:200]}")
        
        # 执行工具
        if tool_name in tools_map:
            tool_func = tools_map[tool_name]
            
            # 特殊处理：某些工具需要从 state 获取数据
            if tool_name == "find_resource_relationships":
                tool_args["all_operations"] = state["input_operations"]
            elif tool_name == "build_evidence_chain":
                tool_args["operations"] = state["input_operations"]
            
            try:
                result = tool_func.invoke(tool_args)
                result_str = json.dumps(result, ensure_ascii=False, indent=2)
                print(f"✅ 工具返回: {result_str[:300]}...")
                
                return {"current_tool_output": result_str}
            except Exception as e:
                error_msg = f"工具执行失败: {str(e)}"
                print(f"❌ {error_msg}")
                return {"current_tool_output": error_msg}
        else:
            error_msg = f"未知工具: {tool_name}"
            print(f"❌ {error_msg}")
            return {"current_tool_output": error_msg}
    
    # 如果没有 tool_calls，尝试解析文本格式的 Action
    content = last_message.content
    action_match = re.search(r'Action:\s*(\w+)', content)
    input_match = re.search(r'Action Input:\s*(\{.*?\})', content, re.DOTALL)
    
    if action_match and input_match:
        tool_name = action_match.group(1)
        try:
            tool_args = json.loads(input_match.group(1))
            
            print(f"🔧 调用工具: {tool_name}")
            print(f"📝 参数: {json.dumps(tool_args, ensure_ascii=False)[:200]}")
            
            if tool_name in tools_map:
                tool_func = tools_map[tool_name]
                
                # 特殊处理
                if tool_name == "find_resource_relationships":
                    tool_args["all_operations"] = state["input_operations"]
                elif tool_name == "build_evidence_chain":
                    tool_args["operations"] = state["input_operations"]
                
                result = tool_func.invoke(tool_args)
                result_str = json.dumps(result, ensure_ascii=False, indent=2)
                print(f"✅ 工具返回: {result_str[:300]}...")
                
                return {"current_tool_output": result_str}
        except Exception as e:
            error_msg = f"工具执行失败: {str(e)}"
            print(f"❌ {error_msg}")
            return {"current_tool_output": error_msg}
    
    return {"current_tool_output": "无需执行工具"}


def observation_node(state: TrackerState) -> Dict[str, Any]:
    """
    观察节点：将工具输出反馈给 LLM
    """
    print("\n👁️  [Observation] 反馈工具结果...")
    
    tool_output = state.get("current_tool_output", "无输出")
    
    observation_msg = HumanMessage(
        content=f"Observation: {tool_output}\n\n请继续分析或输出最终结果。"
    )
    
    return {"messages": [observation_msg]}


def finalize_node(state: TrackerState) -> Dict[str, Any]:
    """
    终结节点：提取最终结果并保存到 JSON 文件
    """
    print("\n🏁 [Finalize] 提取最终分析结果...")
    
    last_message = state["messages"][-1]
    content = last_message.content
    
    # 提取 Final Answer
    try:
        if "Final Answer:" in content:
            json_part = content.split("Final Answer:")[-1].strip()
            json_part = json_part.replace("```json", "").replace("```", "").strip()
            json_part = re.sub(r',\s*}', '}', json_part)
            json_part = re.sub(r',\s*]', ']', json_part)
            
            final_result = json.loads(json_part)
            
            # 保存到 JSON 文件
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = "output"
            os.makedirs(output_dir, exist_ok=True)
            output_file = os.path.join(output_dir, f"evidence_trace_{timestamp}.json")
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(final_result, f, ensure_ascii=False, indent=2)
            
            print(f"✅ 分析结果已保存到: {output_file}")
            
            return {
                "final_output": final_result,
                "is_complete": True
            }
    except Exception as e:
        print(f"❌ 解析 Final Answer 失败: {e}")
        return {
            "final_output": {
                "error": "无法解析最终结果",
                "raw_content": content
            },
            "is_complete": True
        }
    
    return {"is_complete": False}
