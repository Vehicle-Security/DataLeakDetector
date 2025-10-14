import cv2
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import List, Dict, Optional, Tuple
import json
import os
import base64
import shutil


class KeyFrameDetection:
    def __init__(self, video_source: str, sensitivity_threshold: float = 0.7):
        self.video_source = video_source
        self.sensitivity_threshold = sensitivity_threshold
        self.cap = cv2.VideoCapture(video_source)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.key_frames = []  # 存储检测到的关键帧
        self.key_frames_b64: List[str] = []

        # 初始化日志
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger("KeyFrameAgent")





    def extract_frame(self, frame_number: int) -> np.ndarray:
        """提取指定帧"""
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = self.cap.read()
        if ret:
            return frame
        return None

    def compute_frame_difference(self, frame1: np.ndarray, frame2: np.ndarray) -> float:
        """计算两帧之间的差异度"""
        if frame1 is None or frame2 is None:
            return 1.0

        # 转换为灰度图
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

        # 计算结构相似性
        from skimage.metrics import structural_similarity as ssim
        score, _ = ssim(gray1, gray2, full=True)

        return 1 - score  # 返回差异度

    def binary_search_key_frames(self, start_frame: int, end_frame: int,
                                 min_interval: int = 30) -> List[int]:
        """
        使用二分搜索在指定范围内查找关键帧
        """
        if end_frame - start_frame <= min_interval:
            return [start_frame]

        mid_frame = (start_frame + end_frame) // 2

        # 提取关键帧进行比较
        frame_start = self.extract_frame(start_frame)
        frame_mid = self.extract_frame(mid_frame)
        frame_end = self.extract_frame(end_frame)

        # 计算差异
        diff_start_mid = self.compute_frame_difference(frame_start, frame_mid)
        diff_mid_end = self.compute_frame_difference(frame_mid, frame_end)
        print("diff_start_mid", diff_start_mid)
        print("diff_mid_end", diff_mid_end)


        key_frames = []

        # 如果中间帧与起始帧差异大，说明有关键变化
        if diff_start_mid > self.sensitivity_threshold:
            key_frames.extend(self.binary_search_key_frames(start_frame, mid_frame, min_interval))

        # 如果中间帧与结束帧差异大，继续搜索后半段
        if diff_mid_end > self.sensitivity_threshold:
            key_frames.extend(self.binary_search_key_frames(mid_frame, end_frame, min_interval))

        # 如果两端差异都不大，只保留起始帧
        if not key_frames:
            key_frames.append(start_frame)

        return key_frames





    def process_video_stream(self):
        """处理整个视频流"""
        self.logger.info(f"开始处理视频流，总帧数: {self.total_frames}")


        all_key_frames = self.binary_search_key_frames(0, self.total_frames - 1)


        output_dir = "res"
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
            os.makedirs(output_dir)
        else:
            os.makedirs(output_dir)

        for frame_num in all_key_frames:
            frame = self.extract_frame(frame_num)
            if frame is None:
                continue

            # _, buf = cv2.imencode('.jpg', frame)
            # # bytes -> base64
            # b64_str = base64.b64encode(buf.tobytes()).decode('utf-8')
            # self.key_frames_b64.append(b64_str)

            # 保存关键帧截图
            filename = os.path.join(output_dir, f"frame_{frame_num}.jpg")
            cv2.imwrite(filename, frame)




