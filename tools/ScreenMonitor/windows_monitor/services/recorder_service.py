# -*- coding: utf-8 -*-
"""
recorder_service.py - 屏幕录制服务
职责：封装屏幕录制逻辑，提供简洁的 start/stop 接口

对应架构角色：Service（服务层）
"""

import os
import shutil
import subprocess
import threading
import time
from typing import Optional

from core.utils import app_logger


class RecorderService:
    """
    屏幕录制服务
    
    使用 ffmpeg 进行屏幕录制（无需额外依赖）
    如果 ffmpeg 不可用，则使用 PIL 截图方案
    """
    
    def __init__(self, fps: int = 4):
        """
        Args:
            fps: 录制帧率
        """
        self.fps = fps
        self.recording = False
        self.process: Optional[subprocess.Popen] = None
        self.output_path: Optional[str] = None
        self.ffmpeg_path = self._resolve_ffmpeg_path()
        self._lock = threading.Lock()
    
    def start(self, output_path: str) -> bool:
        """
        开始录制
        
        Args:
            output_path: 输出视频路径（.mp4）
            
        Returns:
            True 如果启动成功
        """
        with self._lock:
            if self.recording:
                app_logger.warning("录制已在进行中")
                return False
            
            self.output_path = output_path
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # 尝试使用 ffmpeg
            if self._start_ffmpeg(output_path):
                self.recording = True
                app_logger.info(f"📹 开始屏幕录制: {output_path}")
                return True
            
            app_logger.warning("ffmpeg 不可用，跳过屏幕录制")
            return False
    
    def stop(self) -> bool:
        """
        停止录制
        
        Returns:
            True 如果停止成功
        """
        with self._lock:
            if not self.recording:
                return False
            
            self.recording = False
            
            if self.process:
                try:
                    # 发送 q 键停止 ffmpeg
                    self.process.stdin.write(b'q')
                    self.process.stdin.flush()
                    self.process.wait(timeout=5)
                except Exception:
                    self.process.terminate()
                finally:
                    self.process = None
            
            app_logger.info(f"📹 屏幕录制已停止: {self.output_path}")
            return True
    
    def is_recording(self) -> bool:
        """检查是否正在录制"""
        return self.recording
    
    def _start_ffmpeg(self, output_path: str) -> bool:
        """使用 ffmpeg 开始录制"""
        try:
            ffmpeg_executable = self.ffmpeg_path or self._resolve_ffmpeg_path()
            if not ffmpeg_executable:
                return False

            # Windows 使用 gdigrab
            cmd = [
                ffmpeg_executable,
                '-f', 'gdigrab',
                '-framerate', str(self.fps),
                '-i', 'desktop',
                '-c:v', 'libx264',
                '-preset', 'veryfast',
                '-crf', '32',
                '-pix_fmt', 'yuv420p',
                '-y',
                output_path
            ]
            
            self.process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            
            # 等待一下确认启动成功
            time.sleep(0.5)
            if self.process.poll() is not None:
                return False
            
            return True
        except FileNotFoundError:
            return False
        except Exception as e:
            app_logger.error(f"ffmpeg 启动失败: {e}")
            return False

    def _resolve_ffmpeg_path(self) -> Optional[str]:
        """解析可用的 ffmpeg 可执行文件路径。"""
        for candidate in ("ffmpeg", "ffmpeg.exe"):
            resolved = shutil.which(candidate)
            if resolved:
                return resolved

        try:
            import imageio_ffmpeg

            bundled = imageio_ffmpeg.get_ffmpeg_exe()
            if bundled and os.path.exists(bundled):
                app_logger.info(f"📦 使用 imageio_ffmpeg 内置二进制: {bundled}")
                return bundled
        except Exception as e:
            app_logger.warning(f"解析 imageio_ffmpeg 内置 ffmpeg 失败: {e}")

        return None
