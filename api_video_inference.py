
from langchain.schema import HumanMessage
from langchain_openai import ChatOpenAI
from pydantic import SecretStr
from prompt import inference_prompt

def api_inference_video( model_name, prompt,key_frames=None):

    llm = ChatOpenAI(
        model=model_name,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        api_key=SecretStr("sk-ebaed1219b52430ba9447180c9b795d3"),
        streaming=True,

    )

    # 构建消息内容
    content = [
        {"type": "text",
         "text": prompt}
    ]

    # 添加图片
    if key_frames:
        for img_base64 in key_frames:
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"}
            })

    #print(content)


    messages = [HumanMessage(content=content)]

    try:
        response = llm.invoke(messages)
        return response.content
    except Exception as e:
        print(f"API调用错误: {e}")
        return None




