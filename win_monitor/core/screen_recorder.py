# -*- coding: utf-8 -*-
"""
屏幕录制器 - 使用ffmpeg进行屏幕录制
"""
import subprocess
import threading
import time
import os
import sys
from datetime import datetime
import platform


class ScreenRecorder:
    """使用ffmpeg进行屏幕录制"""
    
    def __init__(self, config=None):
        """
        初始化录制器
        
        Args:
            config: 配置对象，包含录制参数
        """
        self.config = config or {}
        self.is_recording_flag = False
        self.ffmpeg_process = None
        self.output_path = None
        self.start_time = None
        self.recording_thread = None
        
        # 默认配置
        self.fps = self.config.get("recording", {}).get("fps", 10)
        self.quality = self.config.get("recording", {}).get("quality", 23)  # CRF值，越小质量越高
        self.codec = self.config.get("recording", {}).get("codec", "libx264")
        
        # 检查ffmpeg是否可用
        self._check_ffmpeg()
    
    def _check_ffmpeg(self):
        """检查ffmpeg是否安装并可用"""
        try:
            result = subprocess.run(
                ["ffmpeg", "-version"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode != 0:
                raise RuntimeError("ffmpeg不可用")
            print("[RECORDER] ffmpeg检测成功")
        except (subprocess.SubprocessError, FileNotFoundError) as e:
            raise RuntimeError(f"ffmpeg未安装或不在PATH中: {e}")
    
    def start_recording(self, output_dir="./recordings", fps=None, resolution=None, 
                       filename=None, monitor=0):
        """
        开始录制屏幕
        
        Args:
            output_dir: 输出目录
            fps: 帧率，默认使用配置值
            resolution: 分辨率，格式如"1920x1080"，None表示使用屏幕分辨率
            filename: 输出文件名，None表示自动生成
            monitor: 监视器索引（0=主显示器）
            
        Returns:
            str: 输出文件路径
        """
        if self.is_recording_flag:
            raise RuntimeError("录制已在进行中")
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 生成输出文件名
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"screen_recording_{timestamp}.mp4"
        
        self.output_path = os.path.join(output_dir, filename)
        
        # 使用传入的fps或默认值
        fps = fps or self.fps
        
        # 构建ffmpeg命令
        system = platform.system()
        
        if system == "Windows":
            # Windows使用gdigrab
            input_args = [
                "-f", "gdigrab",
                "-framerate", str(fps),
                "-i", "desktop"
            ]
        elif system == "Linux":
            # Linux使用x11grab
            input_args = [
                "-f", "x11grab",
                "-framerate", str(fps),
                "-i", f":{monitor}.0"
            ]
        elif system == "Darwin":  # macOS
            # macOS使用avfoundation
            input_args = [
                "-f", "avfoundation",
                "-framerate", str(fps),
                "-i", str(monitor)
            ]
        else:
            raise RuntimeError(f"不支持的操作系统: {system}")
        
        # 视频编码参数（优化以提高兼容性）
        output_args = [
            "-c:v", self.codec,
            "-preset", "ultrafast",  # 快速编码
            "-crf", str(self.quality),
            "-pix_fmt", "yuv420p",  # 兼容性
            "-movflags", "+faststart",  # 优化MP4文件头
            "-y"  # 覆盖已存在文件
        ]
        
        # 如果指定了分辨率
        if resolution:
            output_args.extend(["-s", resolution])
        
        # 完整命令
        cmd = ["ffmpeg", "-loglevel", "warning"] + input_args + output_args + [self.output_path]
        
        print(f"[RECORDER] 开始录制: {self.output_path}")
        print(f"[RECORDER] 帧率: {fps} FPS, 质量: CRF {self.quality}")
        print(f"[RECORDER] 命令: {' '.join(cmd[:10])}...")  # 显示部分命令
        
        try:
            # 启动ffmpeg进程 - 使用bytes模式而不是text模式
            self.ffmpeg_process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,  # 不捕获stdout
                stderr=subprocess.PIPE,
                bufsize=10**8,  # 100MB缓冲区
                creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0
            )
            
            self.is_recording_flag = True
            self.start_time = time.time()
            
            # 启动监控线程
            self.recording_thread = threading.Thread(target=self._monitor_recording, daemon=True)
            self.recording_thread.start()
            
            print(f"[RECORDER] 录制已启动，PID: {self.ffmpeg_process.pid}")
            return self.output_path
            
        except Exception as e:
            print(f"[RECORDER] 启动录制失败: {e}")
            self.is_recording_flag = False
            raise
    
    def _monitor_recording(self):
        """监控录制进程"""
        while self.is_recording_flag and self.ffmpeg_process:
            if self.ffmpeg_process.poll() is not None:
                # 进程已结束
                if self.is_recording_flag:
                    print("[RECORDER] 录制进程意外结束")
                    self.is_recording_flag = False
                break
            time.sleep(1)
    
    def stop_recording(self):
        """停止录制"""
        if not self.is_recording_flag:
            print("[RECORDER] 当前没有正在进行的录制")
            return None
        
        print("[RECORDER] 正在停止录制...")
        self.is_recording_flag = False
        
        if self.ffmpeg_process:
            try:
                # 发送'q'命令优雅地停止ffmpeg（使用bytes模式）
                try:
                    if self.ffmpeg_process.stdin:
                        self.ffmpeg_process.stdin.write(b'q\n')
                        self.ffmpeg_process.stdin.flush()
                        self.ffmpeg_process.stdin.close()
                    print("[RECORDER] 已发送停止命令")
                except Exception as e:
                    print(f"[RECORDER] 发送停止命令失败: {e}")
                
                # 等待进程正常结束（增加到20秒）
                print("[RECORDER] 等待ffmpeg完成视频处理...")
                try:
                    stdout, stderr = self.ffmpeg_process.communicate(timeout=20)
                    print("[RECORDER] ✓ ffmpeg正常退出")
                    
                    # 显示任何警告信息
                    if stderr:
                        stderr_text = stderr.decode('utf-8', errors='ignore') if isinstance(stderr, bytes) else stderr
                        if 'error' in stderr_text.lower():
                            print(f"[RECORDER] ⚠️ FFmpeg警告: {stderr_text[:200]}")
                    
                except subprocess.TimeoutExpired:
                    print("[RECORDER] ⚠️ 进程超时，尝试终止...")
                    self.ffmpeg_process.terminate()
                    try:
                        self.ffmpeg_process.wait(timeout=5)
                        print("[RECORDER] 进程已终止")
                    except subprocess.TimeoutExpired:
                        print("[RECORDER] ⚠️ 强制终止进程")
                        self.ffmpeg_process.kill()
                        self.ffmpeg_process.wait()
                
                # 额外等待确保文件写入完成
                time.sleep(1.0)
                
                duration = self.get_recording_duration()
                print(f"[RECORDER] ✓ 录制已停止，时长: {duration:.1f} 秒")
                print(f"[RECORDER] ✓ 视频已保存: {self.output_path}")
                
                # 验证视频文件
                if os.path.exists(self.output_path):
                    file_size = os.path.getsize(self.output_path)
                    print(f"[RECORDER] 文件大小: {file_size / 1024 / 1024:.2f} MB")
                    if file_size < 10240:  # 小于10KB可能有问题
                        print(f"[RECORDER] ⚠️ 警告: 文件大小异常，视频可能损坏")
                    else:
                        # 快速验证视频格式
                        try:
                            result = subprocess.run(
                                ["ffmpeg", "-v", "error", "-i", self.output_path, "-f", "null", "-"],
                                capture_output=True,
                                timeout=5
                            )
                            if result.returncode == 0:
                                print(f"[RECORDER] ✓ 视频文件格式验证通过")
                            else:
                                error_msg = result.stderr.decode('utf-8', errors='ignore')
                                print(f"[RECORDER] ⚠️ 视频可能有问题: {error_msg[:100]}")
                        except Exception as e:
                            print(f"[RECORDER] 无法验证视频: {e}")
                else:
                    print(f"[RECORDER] ⚠️ 警告: 视频文件不存在")
                
                return self.output_path
                
            except Exception as e:
                print(f"[RECORDER] 停止录制时出错: {e}")
                import traceback
                traceback.print_exc()
                if self.ffmpeg_process and self.ffmpeg_process.poll() is None:
                    self.ffmpeg_process.kill()
            finally:
                self.ffmpeg_process = None
        
        return self.output_path
    
    def get_recording_duration(self):
        """
        获取录制时长（秒）
        
        Returns:
            float: 录制时长
        """
        if self.start_time is None:
            return 0.0
        
        if self.is_recording_flag:
            return time.time() - self.start_time
        else:
            # 已停止，返回总时长
            return time.time() - self.start_time if self.start_time else 0.0
    
    def is_recording(self):
        """
        检查是否正在录制
        
        Returns:
            bool: True表示正在录制
        """
        return self.is_recording_flag
    
    def get_output_path(self):
        """
        获取输出文件路径
        
        Returns:
            str: 输出文件路径，如果未开始录制则返回None
        """
        return self.output_path


if __name__ == "__main__":
    # 简单测试
    print("=== 屏幕录制器测试 ===")
    
    recorder = ScreenRecorder()
    
    try:
        # 录制5秒
        output = recorder.start_recording(output_dir="./test_recordings", fps=10)
        print(f"录制中... 输出文件: {output}")
        
        time.sleep(5)
        
        recorder.stop_recording()
        print("测试完成")
        
    except KeyboardInterrupt:
        print("\n中断录制...")
        recorder.stop_recording()
    except Exception as e:
        print(f"测试失败: {e}")
        import traceback
        traceback.print_exc()
