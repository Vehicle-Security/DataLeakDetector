# graph.py
"""
EvidenceTracer 工作流图定义
基于 LangGraph 构建资源追踪的多步推理流程
"""

import os
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from state import TrackerState
from nodes import (
    initialize_node,
    reasoning_node,
    action_node,
    observation_node,
    finalize_node,
)

# 加载环境变量
load_dotenv()


def should_continue(state: TrackerState) -> str:
    """
    条件边逻辑：判断工作流的下一步
    
    Returns:
        - "finalize": 完成分析，提取最终结果
        - "action": 执行工具调用
        - "continue": 继续推理
    """
    last_message = state["messages"][-1]
    content = last_message.content
    
    # 检查是否输出了 Final Answer
    if "Final Answer" in content or "Final Answer:" in content:
        print("📋 检测到 Final Answer，准备终结...")
        return "finalize"
    
    # 检查是否有工具调用
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        print("🔧 检测到工具调用...")
        return "action"
    
    # 检查文本格式的 Action
    if "Action:" in content and "Action Input:" in content:
        print("🔧 检测到文本格式的工具调用...")
        return "action"
    
    # 默认继续推理
    return "continue"


def after_action(state: TrackerState) -> str:
    """
    执行工具后的路由：总是返回观察节点
    """
    return "observe"


def after_observation(state: TrackerState) -> str:
    """
    观察后的路由：返回推理节点继续分析
    """
    return "reason"


def after_finalize(state: TrackerState) -> str:
    """
    终结节点后的路由：检查是否真正完成
    """
    if state.get("is_complete"):
        print("✅ 分析完成，结束工作流")
        return "end"
    else:
        print("⚠️  未完成，继续推理")
        return "reason"


# ============================================
# 构建 LangGraph 工作流
# ============================================

def build_evidence_tracer_graph():
    """
    构建 EvidenceTracer 的状态图
    """
    workflow = StateGraph(TrackerState)
    
    # 添加节点
    workflow.add_node("initialize", initialize_node)
    workflow.add_node("reason", reasoning_node)
    workflow.add_node("action", action_node)
    workflow.add_node("observe", observation_node)
    workflow.add_node("finalize", finalize_node)
    
    # 设置入口点
    workflow.set_entry_point("initialize")
    
    # 添加边
    # initialize -> reason
    workflow.add_edge("initialize", "reason")
    
    # reason -> (finalize | action | reason)
    workflow.add_conditional_edges(
        "reason",
        should_continue,
        {
            "finalize": "finalize",
            "action": "action",
            "continue": "reason",
        }
    )
    
    # action -> observe
    workflow.add_edge("action", "observe")
    
    # observe -> reason
    workflow.add_edge("observe", "reason")
    
    # finalize -> (END | reason)
    workflow.add_conditional_edges(
        "finalize",
        after_finalize,
        {
            "end": END,
            "reason": "reason",
        }
    )
    
    # 编译图
    app = workflow.compile()
    
    return app


# ============================================
# 主执行函数
# ============================================

def run_evidence_tracer(input_operations: list, max_iterations: int = 20):
    """
    运行 EvidenceTracer 分析
    
    Args:
        input_operations: 来自 RiskSieve 的敏感操作片段列表
        max_iterations: 最大迭代次数，防止无限循环
        
    Returns:
        分析结果字典
    """
    print("=" * 80)
    print("🔍 EvidenceTracer - 敏感资源跨过程追踪")
    print("=" * 80)
    
    # 初始化状态
    initial_state = TrackerState(
        messages=[],
        input_operations=input_operations,
        current_operation_index=0,
        tracked_resources={},
        evidence_chains=[],
        current_tool_output=None,
        analysis_results=[],
        final_output=None,
        is_complete=False,
    )
    
    # 构建图
    app = build_evidence_tracer_graph()
    
    # 执行工作流
    try:
        iteration = 0
        final_state = initial_state
        
        for state in app.stream(initial_state):
            iteration += 1
            print(f"\n{'='*60}")
            print(f"迭代 {iteration}/{max_iterations}")
            print(f"{'='*60}")
            
            final_state = state
            
            # 检查是否完成
            if any(s.get("is_complete") for s in state.values()):
                print("\n✅ 工作流完成")
                break
            
            if iteration >= max_iterations:
                print("\n⚠️  达到最大迭代次数，强制终止")
                break
        
        # 提取最终结果
        result = None
        for node_state in final_state.values():
            if isinstance(node_state, dict) and "final_output" in node_state:
                result = node_state["final_output"]
                break
        
        if not result:
            # 尝试从最后的消息中提取
            print("\n⚠️  未找到 final_output，尝试从消息中提取...")
            result = {
                "status": "incomplete",
                "message": "分析未完成或结果格式不正确",
                "iterations": iteration
            }
        
        print("\n" + "=" * 80)
        print("📊 最终结果")
        print("=" * 80)
        print(result)
        
        return result
        
    except Exception as e:
        print(f"\n❌ 执行过程中出错: {e}")
        import traceback
        traceback.print_exc()
        return {
            "status": "error",
            "error": str(e)
        }


# ============================================
# 单独导出编译后的应用
# ============================================

app = build_evidence_tracer_graph()
graph_png = app.get_graph().draw_mermaid_png()
with open("agent_workflow_graph.png", "wb") as f:
    f.write(graph_png)


if __name__ == "__main__":
    # 测试用例
    test_operations = [
        {
            "operation_id": "op_001",
            "operation_type": "file_access",
            "resource_name": "机密报告.pdf",
            "app_name": "Adobe Reader",
            "start_time": "10:23:15",
            "end_time": "10:23:45",
            "keyframes": ["/path/to/frame1.jpg"],
            "raw_description": "用户在 Adobe Reader 中打开了名为'机密报告.pdf'的文件，并浏览了其中的内容。"
        },
        {
            "operation_id": "op_002",
            "operation_type": "file_compress",
            "resource_name": "report.zip",
            "app_name": "7-Zip",
            "start_time": "10:25:10",
            "end_time": "10:25:20",
            "keyframes": ["/path/to/frame2.jpg"],
            "raw_description": "用户使用 7-Zip 将'机密报告.pdf'压缩为'report.zip'，并设置了密码。"
        },
        {
            "operation_id": "op_003",
            "operation_type": "file_upload",
            "resource_name": "report.zip",
            "app_name": "Chrome",
            "start_time": "10:26:00",
            "end_time": "10:26:30",
            "keyframes": ["/path/to/frame3.jpg"],
            "raw_description": "用户在 Chrome 浏览器中将'report.zip'上传到外部云存储服务。"
        }
    ]
    
    print("\n🧪 运行测试用例...")
    result = run_evidence_tracer(test_operations, max_iterations=10)
