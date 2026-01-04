# tools.py
import os
import json
from typing import List, Dict, Any
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 初始化 LLM（用于工具内部调用）
tool_llm = ChatOpenAI(
    model=os.getenv("MODEL_NAME"),
    base_url=os.getenv("OPENAI_BASE_URL"),
    api_key=os.getenv("OPENAI_API_KEY"),
    temperature=0.0,  # 工具调用需要确定性
    streaming=False
)


@tool
def extract_resource_from_description(operation_description: str, operation_type: str) -> Dict[str, Any]:
    """
    使用 LLM 从操作描述中提取资源转换关系
    
    Args:
        operation_description: 操作的文字描述
        operation_type: 操作类型（如 file_compress, file_rename 等）
        
    Returns:
        包含源资源、目标资源的 JSON 对象
    """
    print(f"   [Tool] 使用 LLM 提取资源转换关系...")
    
    prompt = f"""
分析以下操作描述，提取资源转换关系。

操作类型: {operation_type}
操作描述: {operation_description}

请严格按照以下 JSON 格式输出，不要添加任何额外文字：
{{
    "source_resources": ["源文件1", "源文件2"],  // 操作的输入资源（如果有）
    "target_resource": "目标文件",  // 操作的输出资源（如果有）
    "transformation_type": "描述转换类型"  // 如：压缩、重命名、复制等
}}

规则：
1. 如果描述中提到"将A转换/压缩/重命名为B"，则 A 是源，B 是目标
2. 如果是单向操作（如只是打开文件），source 和 target 可能相同
3. 只提取文件名，不要添加路径
4. 如果没有找到，字段值为 null 或空列表
"""
    
    try:
        response = tool_llm.invoke([
            SystemMessage(content="你是一个专业的文件操作分析助手，擅长从描述中提取资源信息。"),
            HumanMessage(content=prompt)
        ])
        
        content = response.content.strip()
        # 移除可能的 markdown 代码块标记
        content = content.replace("```json", "").replace("```", "").strip()
        
        result = json.loads(content)
        print(f"   ✅ LLM 提取结果: {json.dumps(result, ensure_ascii=False)}")
        return result
        
    except Exception as e:
        print(f"   ❌ LLM 提取失败: {e}")
        return {
            "source_resources": [],
            "target_resource": None,
            "transformation_type": "unknown"
        }


@tool
def trace_resource_flow(
    target_resource: str,
    all_operations: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    使用 LLM 追踪资源的完整流转路径
    
    让 LLM 分析所有操作，推理出目标资源的：
    1. 来源（它是从哪个文件转换来的？）
    2. 流转（经历了哪些操作？）
    3. 派生（它又变成了什么？）
    4. 风险（是否存在隐蔽行为？）
    
    Args:
        target_resource: 要追踪的目标资源名称
        all_operations: 所有操作列表
        
    Returns:
        资源流转的完整链路分析
    """
    print(f"   [Tool] 使用 LLM 追踪资源流转: {target_resource}")
    
    # 构造操作序列描述
    ops_summary = "操作序列:\n"
    for i, op in enumerate(all_operations, 1):
        ops_summary += f"\n{i}. [{op.get('operation_id')}] {op.get('operation_type')}\n"
        ops_summary += f"   - 资源: {op.get('resource_name')}\n"
        ops_summary += f"   - 应用: {op.get('app_name')}\n"
        ops_summary += f"   - 时间: {op.get('start_time')}\n"
        ops_summary += f"   - 描述: {op.get('raw_description')}\n"
    
    prompt = f"""
请分析以下操作序列，追踪资源 "{target_resource}" 的完整流转路径。

{ops_summary}

请严格按照以下 JSON 格式输出分析结果，不要添加任何额外文字：
{{
    "target_resource": "{target_resource}",
    "source_chain": [
        {{
            "resource": "源文件名",
            "operation_id": "相关操作ID",
            "relationship": "关系描述（如：压缩自、重命名自）"
        }}
    ],
    "flow_steps": [
        {{
            "step": 1,
            "operation_id": "op_xxx",
            "action": "动作描述",
            "resource_before": "操作前的资源",
            "resource_after": "操作后的资源"
        }}
    ],
    "derived_chain": [
        {{
            "resource": "派生文件名",
            "operation_id": "相关操作ID",
            "relationship": "关系描述（如：被压缩为、被上传到）"
        }}
    ],
    "risk_indicators": ["风险标签1", "风险标签2"],
    "summary": "用一句话总结资源的完整流转路径"
}}

分析要点：
1. 向前追溯：{target_resource} 是从哪个文件转换来的？
2. 流转过程：经历了哪些操作？每步资源如何变化？
3. 向后追踪：{target_resource} 又被转换成了什么？
4. 风险识别：是否存在重命名、压缩、加密、外发等风险行为？

风险标签参考：
- file_obfuscation: 文件脱敏（重命名、压缩）
- encryption_before_transfer: 传输前加密
- data_exfiltration: 数据外发
- cross_application_transfer: 跨应用传输
- format_change: 格式转换
"""
    
    try:
        response = tool_llm.invoke([
            SystemMessage(content="你是一个专业的数据泄露分析专家，擅长追踪文件流转路径和识别隐蔽的数据外发行为。"),
            HumanMessage(content=prompt)
        ])
        
        content = response.content.strip()
        # 移除可能的 markdown 代码块标记
        content = content.replace("```json", "").replace("```", "").strip()
        
        result = json.loads(content)
        print(f"   ✅ LLM 追踪结果: {result.get('summary', 'N/A')}")
        return result
        
    except Exception as e:
        print(f"   ❌ LLM 追踪失败: {e}")
        return {
            "target_resource": target_resource,
            "source_chain": [],
            "flow_steps": [],
            "derived_chain": [],
            "risk_indicators": [],
            "summary": f"追踪失败: {str(e)}"
        }


# 工具列表，供 LLM 调用
tools_list = [
    extract_resource_from_description,
    trace_resource_flow,
]
