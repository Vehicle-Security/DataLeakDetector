import os
from natsort import natsorted
import base64
from api_video_inference import *
from prompt import *
from final_decision import *
from local_video_inference import local_inference_video

def image_to_base64(image_path):
    """将图像文件转换为Base64编码字符串"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def process_directory(directory, model_name, use_api=False):
    """处理指定目录下的所有图像"""
    result = []
    sub_dirs = [d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))]
    sub_dirs = natsorted(sub_dirs)


    for dir_name in sub_dirs:
        sub_dir_path = os.path.join(directory, dir_name)
        print(f"Processing directory: {sub_dir_path}")
        # 获取所有图像文件并使用natsort进行自然排序
        image_files = [f for f in os.listdir(sub_dir_path) if f.endswith(('.png', '.jpg', '.jpeg'))]
        image_files = natsorted(image_files)
        images = [os.path.join(sub_dir_path, f) for f in image_files]
        image_base64s = [image_to_base64(img) for img in images]
        # # 将所有图像的Base64编码发送给AI
        if use_api:
            res = api_inference_video(model_name, image_prompt, image_base64s)
        else:
            res = local_inference_video(model_name,image_prompt, image_base64s)



        print(res)
        result.append(res)

    return result







