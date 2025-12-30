"""
文件管理模块
管理输入输出文件夹，实现同名文件替换逻辑
"""
import os
import shutil
from pathlib import Path
from typing import Optional


class FileManager:
    """文件管理器，负责输入输出文件的组织和管理"""
    
    def __init__(self, base_dir: str = None):
        """
        初始化文件管理器
        
        Args:
            base_dir: 基础目录，默认为当前工作目录
        """
        if base_dir is None:
            base_dir = os.getcwd()
        
        self.base_dir = Path(base_dir)
        self.inputs_dir = self.base_dir / 'inputs'
        self.outputs_dir = self.base_dir / 'outputs'
        self.frames_dir = self.outputs_dir / 'frames'
        self.results_dir = self.outputs_dir / 'results'  # 新增：匹配结果目录
        
        # 创建必要的目录
        self._initialize_directories()
    
    def _initialize_directories(self):
        """初始化目录结构"""
        self.inputs_dir.mkdir(parents=True, exist_ok=True)
        self.outputs_dir.mkdir(parents=True, exist_ok=True)
        self.frames_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)  # 新增
    
    def save_input_image(self, source_path: str, filename: Optional[str] = None) -> str:
        """
        保存输入的变形图，如果同名文件存在则替换
        
        Args:
            source_path: 源图像文件路径
            filename: 保存的文件名，如果为None则使用原文件名
            
        Returns:
            保存后的文件路径
        """
        if filename is None:
            filename = os.path.basename(source_path)
        
        dest_path = self.inputs_dir / filename
        
        # 如果文件已存在，先删除
        if dest_path.exists():
            dest_path.unlink()
        
        # 复制文件
        shutil.copy2(source_path, dest_path)
        
        return str(dest_path)
    
    def get_output_dir_for_query(self, query_image_name: str) -> str:
        """
        获取或创建特定查询图像的输出目录
        
        Args:
            query_image_name: 查询图像的文件名（不含扩展名）
            
        Returns:
            输出目录路径
        """
        # 移除文件扩展名
        name_without_ext = Path(query_image_name).stem
        
        # 在 results 目录下创建以查询图命名的文件夹
        output_dir = self.results_dir / name_without_ext
        
        # 创建目录(如果不存在)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        return str(output_dir)
    
    def get_frames_dir_for_video(self, video_name: str, check_existing: bool = True) -> tuple:
        """
        获取或创建特定视频的帧输出目录
        
        Args:
            video_name: 视频文件名（不含扩展名）
            check_existing: 是否检查已存在的帧文件夹
            
        Returns:
            (frames_dir_path, frames_exist) 元组
            - frames_dir_path: 帧输出目录路径
            - frames_exist: 帧是否已存在
        """
        # 移除文件扩展名
        name_without_ext = Path(video_name).stem
        
        frames_dir = self.frames_dir / name_without_ext
        
        # 检查目录是否已存在且包含帧文件
        frames_exist = False
        if check_existing and frames_dir.exists():
            # 检查是否有帧文件
            frame_files = list(frames_dir.glob('frame_*.jpg')) + list(frames_dir.glob('frame_*.png'))
            if len(frame_files) > 0:
                frames_exist = True
                print(f"发现已存在的帧目录: {frames_dir}，包含 {len(frame_files)} 个帧")
        
        # 如果不存在或不检查，创建目录
        if not frames_exist:
            if frames_dir.exists():
                shutil.rmtree(frames_dir)
            frames_dir.mkdir(parents=True, exist_ok=True)
        
        return str(frames_dir), frames_exist
    
    def save_input_video(self, source_path: str, filename: Optional[str] = None) -> str:
        """
        保存输入的视频，如果同名文件存在则替换
        
        Args:
            source_path: 源视频文件路径
            filename: 保存的文件名，如果为None则使用原文件名
            
        Returns:
            保存后的文件路径
        """
        if filename is None:
            filename = os.path.basename(source_path)
        
        dest_path = self.inputs_dir / filename
        
        # 如果文件已存在，先删除
        if dest_path.exists():
            dest_path.unlink()
        
        # 复制文件
        shutil.copy2(source_path, dest_path)
        
        return str(dest_path)
    
    def save_result_image(self, image_data, query_name: str, 
                         result_filename: str) -> str:
        """
        保存结果可视化图像
        
        Args:
            image_data: 图像数据（numpy数组或PIL Image）
            query_name: 查询图像名称
            result_filename: 结果文件名
            
        Returns:
            保存的文件路径
        """
        import cv2
        import numpy as np
        from PIL import Image
        
        # 获取查询的输出目录
        output_dir = self.get_output_dir_for_query(query_name)
        
        # 创建visualizations子目录
        visualizations_dir = os.path.join(output_dir, 'visualizations')
        os.makedirs(visualizations_dir, exist_ok=True)
        
        # 构建完整路径
        output_path = os.path.join(visualizations_dir, result_filename)
        
        # print(f"      [DEBUG] 保存路径: {output_path}")
        # print(f"      [DEBUG] 目录存在: {os.path.exists(visualizations_dir)}")
        
        # 根据数据类型保存
        if isinstance(image_data, np.ndarray):
            # print(f"      [DEBUG] 图像类型: numpy.ndarray, shape={image_data.shape}")
            # 使用支持中文路径的方法
            is_success, buffer = cv2.imencode('.jpg', image_data)
            # print(f"      [DEBUG] 编码结果: {is_success}, buffer shape={buffer.shape if is_success else 'N/A'}")
            if is_success:
                with open(output_path, 'wb') as f:
                    bytes_written = f.write(buffer.tobytes())
                # print(f"      [DEBUG] 写入字节数: {bytes_written}")
                # print(f"      [DEBUG] 文件存在: {os.path.exists(output_path)}")
                # if os.path.exists(output_path):
                #     print(f"      [DEBUG] 文件大小: {os.path.getsize(output_path)}")
            else:
                raise ValueError("图像编码失败")
        elif isinstance(image_data, Image.Image):
            image_data.save(output_path)
        else:
            raise ValueError("不支持的图像数据类型")
        
        return output_path
    
    def save_raw_frame(self, image_data, query_name: str, 
                      frame_filename: str) -> str:
        """
        保存原始匹配帧到raw_frames子目录
        
        Args:
            image_data: 图像数据（numpy数组）
            query_name: 查询图像名称
            frame_filename: 帧文件名
            
        Returns:
            保存的文件路径
        """
        import cv2
        import numpy as np
        
        # 获取查询的输出目录
        output_dir = self.get_output_dir_for_query(query_name)
        
        # 创建raw_frames子目录
        raw_frames_dir = os.path.join(output_dir, 'raw_frames')
        os.makedirs(raw_frames_dir, exist_ok=True)
        
        # 构建完整路径
        output_path = os.path.join(raw_frames_dir, frame_filename)
        
        # 保存图像
        if isinstance(image_data, np.ndarray):
            is_success, buffer = cv2.imencode('.jpg', image_data)
            if is_success:
                with open(output_path, 'wb') as f:
                    f.write(buffer.tobytes())
            else:
                raise ValueError("图像编码失败")
        else:
            raise ValueError("不支持的图像数据类型")
        
        return output_path
    
    def clean_outputs(self):
        """清空所有输出目录"""
        if self.outputs_dir.exists():
            for item in self.outputs_dir.iterdir():
                if item.is_dir():
                    shutil.rmtree(item)
                else:
                    item.unlink()
    
    def clean_frames(self):
        """清空所有帧目录"""
        if self.frames_dir.exists():
            for item in self.frames_dir.iterdir():
                if item.is_dir():
                    shutil.rmtree(item)
    
    def get_input_files(self, extension: str = None) -> list:
        """
        获取输入目录中的所有文件
        
        Args:
            extension: 文件扩展名过滤（如'.jpg'），None表示所有文件
            
        Returns:
            文件路径列表
        """
        if not self.inputs_dir.exists():
            return []
        
        files = []
        for item in self.inputs_dir.iterdir():
            if item.is_file():
                if extension is None or item.suffix.lower() == extension.lower():
                    files.append(str(item))
        
        return sorted(files)
    
    def get_output_files(self, query_name: str, extension: str = None) -> list:
        """
        获取特定查询的输出文件
        
        Args:
            query_name: 查询图像名称
            extension: 文件扩展名过滤
            
        Returns:
            文件路径列表
        """
        name_without_ext = Path(query_name).stem
        output_dir = self.results_dir / name_without_ext
        
        if not output_dir.exists():
            return []
        
        files = []
        for item in output_dir.iterdir():
            if item.is_file():
                if extension is None or item.suffix.lower() == extension.lower():
                    files.append(str(item))
        
        return sorted(files)
    
    def create_result_json_path(self, query_name: str) -> str:
        """
        创建结果JSON文件的路径
        
        Args:
            query_name: 查询图像名称
            
        Returns:
            JSON文件路径
        """
        output_dir = self.get_output_dir_for_query(query_name)
        return os.path.join(output_dir, 'results.json')
