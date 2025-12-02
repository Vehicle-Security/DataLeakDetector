"""
视频处理模块
使用FFmpeg提取视频帧，支持时序子采样
"""
import cv2
import os
import subprocess
from typing import List, Tuple, Optional
from pathlib import Path


class VideoProcessor:
    """视频处理器，负责视频帧提取和管理"""
    
    def __init__(self, fps=10, frame_format='jpg'):
        """
        初始化视频处理器
        
        Args:
            fps: 提取帧的帧率（每秒帧数）
            frame_format: 保存帧的图像格式
        """
        self.fps = fps
        self.frame_format = frame_format
    
    def extract_frames(self, video_path: str, output_dir: str, 
                      use_ffmpeg=True) -> List[Tuple[float, str]]:
        """
        从视频中提取帧
        
        Args:
            video_path: 视频文件路径
            output_dir: 帧输出目录
            use_ffmpeg: 是否使用FFmpeg（更快），否则使用OpenCV
            
        Returns:
            帧信息列表 [(timestamp, frame_path), ...]
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"视频文件不存在: {video_path}")
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        if use_ffmpeg and self._check_ffmpeg():
            return self._extract_frames_ffmpeg(video_path, output_dir)
        else:
            return self._extract_frames_opencv(video_path, output_dir)
    
    def _check_ffmpeg(self) -> bool:
        """检查FFmpeg是否可用"""
        try:
            subprocess.run(['ffmpeg', '-version'], 
                         stdout=subprocess.PIPE, 
                         stderr=subprocess.PIPE,
                         check=True)
            return True
        except:
            return False
    
    def _extract_frames_ffmpeg(self, video_path: str, 
                              output_dir: str) -> List[Tuple[float, str]]:
        """
        使用FFmpeg提取帧（推荐方法，速度快）
        
        Args:
            video_path: 视频文件路径
            output_dir: 帧输出目录
            
        Returns:
            帧信息列表
        """
        # 构建FFmpeg命令
        # -r 设置输出帧率
        # -q:v 2 设置高质量输出
        output_pattern = os.path.join(output_dir, f'frame_%06d.{self.frame_format}')
        
        cmd = [
            'ffmpeg',
            '-i', video_path,
            '-vf', f'fps={self.fps}',  # 设置提取帧率
            '-q:v', '2',  # 高质量
            '-y',  # 覆盖已存在的文件
            output_pattern
        ]
        
        try:
            # 执行FFmpeg命令
            subprocess.run(cmd, 
                         stdout=subprocess.PIPE, 
                         stderr=subprocess.PIPE,
                         check=True)
            
            # 获取视频信息以计算时间戳
            cap = cv2.VideoCapture(video_path)
            video_fps = cap.get(cv2.CAP_PROP_FPS)
            cap.release()
            
            # 收集提取的帧
            frames = []
            frame_files = sorted([f for f in os.listdir(output_dir) 
                                if f.startswith('frame_') and f.endswith(f'.{self.frame_format}')])
            
            for idx, frame_file in enumerate(frame_files):
                frame_path = os.path.join(output_dir, frame_file)
                # 根据提取的fps计算时间戳
                timestamp = idx / self.fps
                frames.append((timestamp, frame_path))
            
            return frames
            
        except subprocess.CalledProcessError as e:
            print(f"FFmpeg提取帧失败: {e}")
            print("回退到OpenCV方法...")
            return self._extract_frames_opencv(video_path, output_dir)
    
    def _extract_frames_opencv(self, video_path: str, 
                              output_dir: str) -> List[Tuple[float, str]]:
        """
        使用OpenCV提取帧（备用方法）
        
        Args:
            video_path: 视频文件路径
            output_dir: 帧输出目录
            
        Returns:
            帧信息列表
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"无法打开视频: {video_path}")
        
        # 获取视频信息
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / video_fps
        
        # 计算帧间隔
        frame_interval = int(video_fps / self.fps)
        
        frames = []
        frame_count = 0
        saved_count = 0
        
        while True:
            ret, frame = cap.read()
            
            if not ret:
                break
            
            # 按间隔保存帧
            if frame_count % frame_interval == 0:
                timestamp = frame_count / video_fps
                frame_filename = f'frame_{saved_count:06d}.{self.frame_format}'
                frame_path = os.path.join(output_dir, frame_filename)
                
                # 使用支持中文路径的方法保存
                is_success, buffer = cv2.imencode(f'.{self.frame_format}', frame)
                if is_success:
                    with open(frame_path, 'wb') as f:
                        f.write(buffer)
                    frames.append((timestamp, frame_path))
                    saved_count += 1
                else:
                    print(f"警告: 帧 {frame_count} 编码失败")
            
            frame_count += 1
        
        cap.release()
        
        return frames
    
    def get_frame_at_timestamp(self, video_path: str, 
                              timestamp: float) -> Optional[Tuple[str, any]]:
        """
        提取视频中指定时间戳的帧
        
        Args:
            video_path: 视频文件路径
            timestamp: 时间戳（秒）
            
        Returns:
            (temp_path, frame_image) 或 None
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            return None
        
        # 定位到指定时间
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_number = int(timestamp * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            return None
        
        return frame
    
    def get_video_info(self, video_path: str) -> dict:
        """
        获取视频基本信息
        
        Args:
            video_path: 视频文件路径
            
        Returns:
            包含视频信息的字典
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"无法打开视频: {video_path}")
        
        info = {
            'fps': cap.get(cv2.CAP_PROP_FPS),
            'total_frames': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            'duration': cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS)
        }
        
        cap.release()
        
        return info
