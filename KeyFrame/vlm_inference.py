from langchain_core.messages  import HumanMessage
from langchain_openai import ChatOpenAI

def api_inference_video(model_name, prompt=None, images=None, contents=None):
    """
    通用 VLM 调用：
    ✅ prompt + 多张图片
    ✅ prompt + 单张图片
    ✅ 或者直接传入 contents（更灵活）

    Args:
        model_name: 模型名称
        prompt: 文本提示词
        images: base64 字符串列表（"data:image/jpeg;base64,xxx"）
        contents: 已经构造好的 content 列表（可直接传入）
    """

    llm = ChatOpenAI(
        model=model_name,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        api_key="sk-62823ac2c480482084d040855d2e5a15",
        streaming=False,      # ✅ 批量时禁用 streaming，否则 content 会乱
    )

    # ✅ 如果直接传入 contents，则优先使用
    if contents is None:
        contents = []

        if prompt:
            contents.append({"type": "text", "text": prompt})

        # ✅ 多张图片
        if images:
            for img in images:
                contents.append({
                    "type": "image_url",
                    "image_url": img     # 必须是 "data:image/jpeg;base64,xxx"
                })

    messages = [HumanMessage(content=contents)]

    try:
        resp = llm.invoke(messages)
        return resp.content        # 最终返回字符串
    except Exception as e:
        print(f"API调用错误: {e}")
        return None
