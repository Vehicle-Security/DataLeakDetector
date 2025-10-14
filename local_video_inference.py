import cv2
import base64
from natsort import natsorted
import os
from ollama import Client
from prompt import inference_prompt


def local_inference_video( model_name, prompt, key_frames = None):



    # Ollama 格式的消息
    if key_frames is not  None:
        messages = [
            {
                "role": "user",
                "content": prompt,
                "images": key_frames  # 直接传入base64编码的图片列表
            }
        ]

    else :
        messages = [
            {
                "role": "user",
                "content": prompt,
            }
        ]

    response = Client().chat(
        model=model_name,
        messages=messages
    )

    print(response['message']['content'])

    return response['message']['content']