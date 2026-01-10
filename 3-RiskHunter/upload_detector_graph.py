# upload_detector_graph.py
"""
模块3 LangGraph图定义
基于LangGraph构建上传检测Agent
"""

from langgraph.graph import StateGraph, END
from upload_detector_state import UploadDetectorState
from upload_detector_nodes import (
    initialize_node,
    process_event_node,
    analyze_upload_node,
    finalize_node,
    should_continue_processing
)


def create_upload_detector_graph():
    """
    创建上传检测图
    
    流程：
    1. initialize: 初始化WorklistManager，扫描日志构建worklist
    2. process_event: 从worklist获取事件，调用模块2分析（模块2会调用模块1）
    3. analyze_upload: 分析是否为上传行为，判断是否报警
    4. 判断是否继续：
       - 如果worklist不为空，返回步骤2
       - 如果worklist为空，进入步骤5
    5. finalize: 生成报告，显示统计信息
    """
    
    workflow = StateGraph(UploadDetectorState)
    
    workflow.add_node("initialize", initialize_node)
    workflow.add_node("process_event", process_event_node)
    workflow.add_node("analyze_upload", analyze_upload_node)
    workflow.add_node("finalize", finalize_node)
    
    workflow.set_entry_point("initialize")
    
    workflow.add_edge("initialize", "process_event")
    workflow.add_edge("process_event", "analyze_upload")
    
    workflow.add_conditional_edges(
        "analyze_upload",
        should_continue_processing,
        {
            "continue": "process_event",  # 继续处理下一个事件
            "end": "finalize"  # 结束处理
        }
    )
    
    workflow.add_edge("finalize", END)
    
    app = workflow.compile()

    graph_png = app.get_graph().draw_mermaid_png()
    with open("upload_detector_graph.png", "wb") as f:
        f.write(graph_png)
    
    return app


if __name__ == "__main__":
    """测试图构建"""
    print("=" * 80)
    print("测试上传检测图构建")
    print("=" * 80)
    
    try:
        app = create_upload_detector_graph()
        print("✅ 图构建成功")
        
        # 打印图结构
        print("\n图结构:")
        print(app.get_graph().draw_ascii())
        
    except Exception as e:
        print(f"❌ 图构建失败: {e}")
        import traceback
        traceback.print_exc()
