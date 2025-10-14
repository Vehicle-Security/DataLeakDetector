import logging
import copy
from frame_detect import *
from local_video_inference import local_inference_video
from api_video_inference import api_inference_video
from natsort import natsorted

video_path = "/Users/tujiali/Desktop/v2.mov"
keyFrame = KeyFrameDetection(video_path, sensitivity_threshold=0.09)
# 处理视频并检测关键帧
keyFrame.process_video_stream()
key_frames_based  = keyFrame.key_frames_b64


# def extract_key_frames():
#     """提取视频关键帧"""
#     root = "./res"
#     frames = []
#     for file in natsorted(os.listdir(root)):
#         if file.lower().endswith((".jpg", ".png")):
#             image = os.path.join(root, file)
#             with open(image, "rb") as image_file:
#                 img_base64 = base64.b64encode(image_file.read()).decode('utf-8')
#                 frames.append(img_base64)
#     return frames
#
# # 使用关键帧
# key_frames_based = extract_key_frames()
#
# #选择本地模型
# model_name  = "minicpm-v:8b"
# res_local = local_inference_video(key_frames_based, model_name)
# print(res_local)

# # 调用其他大模型api
# model_name = "qwen2.5-vl-32b-instruct"
# res_api = api_inference_video(key_frames_based, model_name)
# print(res_api)