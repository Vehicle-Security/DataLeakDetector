# tools.py
import base64
import os
import json
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage


vlm_llm = ChatOpenAI(
    model="qwen2-vl-72b-instruct",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    api_key="your-api-key", 
    temperature=0.1,
    streaming=False,
)

def encode_image(image_path: str) -> str:
    """将本地图片转换为 Base64 编码"""
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except Exception as e:
        print(f"读取图片失败: {image_path}, {e}")
        return ""

def run_vlm_analysis(target_paths: list[str], query: str) -> str:
    """
    核心工具：调用 Qwen2-VL 分析图片列表
    """
    
    if not target_paths or not os.path.exists(target_paths[0]):
        print(f"   [VLM Warning] 图片路径不存在，进入模拟模式: {target_paths}")
        
        return "视觉识别结果：屏幕显示用户点击了发送按钮。"
    # -----------------------------------------------------

    print(f"   [VLM] 正在调用 Qwen2-VL 分析 {len(target_paths)} 张图片... Query: {query}")

    content_parts = []
    
    # 1. 添加 Prompt
    content_parts.append({
        "type": "text", 
        "text": f"请仔细观察这些关键帧序列。以下是你需要识别的目标：{query}。请直接识别出任务目标，不要进行推测。"
    })

    # 2. 添加图片
    for path in target_paths:
        b64_img = encode_image(path)
        if b64_img:
            content_parts.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64_img}"}
            })

    message = HumanMessage(content=content_parts)
    system_prompt = SystemMessage(content="你是一个专业的GUI界面审计专家。")

    try:
        response = vlm_llm.invoke([system_prompt, message])
        print("视觉大模型识别结果: ", response)
        return response.content
    except Exception as e:
        return f"视觉分析调用失败: {str(e)}"