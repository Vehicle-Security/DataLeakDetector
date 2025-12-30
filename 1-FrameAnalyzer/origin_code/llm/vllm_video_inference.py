import cv2
import base64
from natsort import natsorted
import os
import requests
import json
#from prompt.prompt import inference_prompt

import cv2
import base64
from natsort import natsorted
import os
import requests
import json
from openai import OpenAI

# 初始化 OpenAI 客户端
client = OpenAI(
    api_key="token-abc123",  # 与 vLLM 服务启动时设置的 api-key 一致
    base_url="http://localhost:8000/v1",
)

def local_inference_video_alternative(model_name, prompt, key_frames=None):
    """
    使用 OpenAI 客户端进行多模态推理
    """
    try:
        if key_frames is not None and len(key_frames) > 0:
            # 构建多模态消息
            content = [{"type": "text", "text": prompt}]
            
            for frame in key_frames:
                content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{frame}"
                    }
                })
            
            messages = [{"role": "user", "content": content}]
        else:
            # 纯文本消息
            messages = [{"role": "user", "content": prompt}]
        
        #print("调试信息 - 发送请求...")
        
        # 使用 OpenAI 客户端调用
        chat_response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            max_tokens=512,
            temperature=0.7,
            # 如果有需要，可以添加 extra_body
            # extra_body={"stop_token_ids": [151645, 151643]}
        )
        
        #print("调试信息 - 响应接收成功")
        answer = chat_response.choices[0].message.content
        print("回答:", answer)
        return answer
        
    except Exception as e:
        print(f"调用服务时发生错误: {str(e)}")
        return None
    
# if __name__ == "__main__":
#     # 假设您已经启动了vLLM服务
#     # 读取图片
#     image_path = "/home/tjl/projects/Agent_data_leak/output2/关键帧_29.00s.jpg"
#     with open(image_path, 'rb') as file:
#             base64_image = base64.b64encode(file.read()).decode('utf-8')
    
#     result = local_inference_video_alternative(
#             model_name="Qwen/Qwen2.5-VL-7B-Instruct",
#             prompt="请详细描述这张图片中的内容",
#             key_frames=[base64_image]
#     )
        
#     if result:
#         print("推理成功！")
        
#     else:
#         print("推理失败")
        