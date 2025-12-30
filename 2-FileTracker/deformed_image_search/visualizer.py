"""
可视化模块
在匹配帧上绘制检测框，生成可视化结果图片
"""
import cv2
import numpy as np
from typing import Optional, Tuple
from PIL import Image, ImageDraw, ImageFont


class Visualizer:
    """可视化器，负责生成匹配结果的可视化图像"""
    
    def __init__(self, box_color=(0, 255, 0), box_thickness=3, 
                 text_color=(255, 255, 255), text_bg_color=(0, 255, 0)):
        """
        初始化可视化器
        
        Args:
            box_color: 边界框颜色 (B, G, R)
            box_thickness: 边界框线条粗细
            text_color: 文本颜色
            text_bg_color: 文本背景颜色
        """
        self.box_color = box_color
        self.box_thickness = box_thickness
        self.text_color = text_color
        self.text_bg_color = text_bg_color
    
    def draw_match_result(self, frame_image, bounding_box: Optional[np.ndarray], 
                         score: float, timestamp: float,
                         query_image=None, show_query=True) -> np.ndarray:
        """
        在匹配帧上绘制检测框和信息
        
        Args:
            frame_image: 帧图像（numpy数组或文件路径）
            bounding_box: 查询图在帧上的投影边界框 [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
            score: 匹配分数
            timestamp: 时间戳
            query_image: 查询图像（可选，用于对比显示）
            show_query: 是否在结果中显示查询图
            
        Returns:
            可视化结果图像（numpy数组）
        """
        # 加载帧图像
        if isinstance(frame_image, str):
            # 使用支持中文路径的方法
            with open(frame_image, 'rb') as f:
                img_array = np.frombuffer(f.read(), dtype=np.uint8)
                frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                if frame is None:
                    raise ValueError(f"无法读取图像: {frame_image}")
        elif isinstance(frame_image, np.ndarray):
            frame = frame_image.copy()
        else:
            raise ValueError("不支持的图像格式")
        
        # 绘制边界框
        if bounding_box is not None:
            # 绘制多边形边界框
            cv2.polylines(frame, [bounding_box], True, self.box_color, 
                         self.box_thickness)
            
            # 在角点处绘制圆点
            for point in bounding_box:
                cv2.circle(frame, tuple(point), 5, (0, 0, 255), -1)
        
        # 添加文本信息
        info_text = f"Score: {score:.2f} | Time: {timestamp:.2f}s"
        self._draw_text_with_background(frame, info_text, (10, 30))
        
        # 如果需要显示查询图，创建组合视图
        if show_query and query_image is not None:
            frame = self._create_comparison_view(query_image, frame, 
                                                 score, timestamp)
        
        return frame
    
    def _draw_text_with_background(self, image: np.ndarray, text: str, 
                                   position: Tuple[int, int], 
                                   font_scale=0.8, font_thickness=2):
        """
        在图像上绘制带背景的文本
        
        Args:
            image: 目标图像
            text: 文本内容
            position: 文本位置 (x, y)
            font_scale: 字体缩放
            font_thickness: 字体粗细
        """
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        # 获取文本尺寸
        (text_width, text_height), baseline = cv2.getTextSize(
            text, font, font_scale, font_thickness
        )
        
        x, y = position
        
        # 绘制背景矩形
        cv2.rectangle(image, 
                     (x - 5, y - text_height - 10),
                     (x + text_width + 5, y + baseline),
                     self.text_bg_color, -1)
        
        # 绘制文本
        cv2.putText(image, text, (x, y - 5), font, font_scale, 
                   self.text_color, font_thickness)
    
    def _create_comparison_view(self, query_image, matched_frame: np.ndarray,
                               score: float, timestamp: float) -> np.ndarray:
        """
        创建查询图和匹配帧的对比视图
        
        Args:
            query_image: 查询图像（numpy数组或文件路径）
            matched_frame: 匹配的帧图像
            score: 匹配分数
            timestamp: 时间戳
            
        Returns:
            组合视图图像
        """
        # 加载查询图像
        if isinstance(query_image, str):
            # 使用支持中文路径的方法
            with open(query_image, 'rb') as f:
                img_array = np.frombuffer(f.read(), dtype=np.uint8)
                query = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                if query is None:
                    raise ValueError(f"无法读取查询图像: {query_image}")
        elif isinstance(query_image, np.ndarray):
            query = query_image.copy()
        else:
            raise ValueError("不支持的查询图像格式")
        
        # 调整查询图大小以匹配帧图高度
        frame_height = matched_frame.shape[0]
        query_aspect = query.shape[1] / query.shape[0]
        new_query_width = int(frame_height * query_aspect)
        query_resized = cv2.resize(query, (new_query_width, frame_height))
        
        # 在查询图上添加标签
        label_query = query_resized.copy()
        self._draw_text_with_background(label_query, "Query Image", (10, 30))
        
        # 创建分隔线
        separator = np.ones((frame_height, 20, 3), dtype=np.uint8) * 255
        
        # 水平拼接图像
        combined = np.hstack([label_query, separator, matched_frame])
        
        return combined
    
    def create_result_grid(self, results: list, max_cols=3) -> Optional[np.ndarray]:
        """
        创建多个匹配结果的网格视图
        
        Args:
            results: 结果列表，每个元素为(frame_image, bbox, score, timestamp)
            max_cols: 每行最多显示的结果数
            
        Returns:
            网格视图图像
        """
        if not results:
            return None
        
        # 绘制每个结果
        result_images = []
        for frame_img, bbox, score, timestamp in results:
            vis_img = self.draw_match_result(frame_img, bbox, score, 
                                            timestamp, show_query=False)
            result_images.append(vis_img)
        
        # 计算网格布局
        n_results = len(result_images)
        n_cols = min(max_cols, n_results)
        n_rows = (n_results + n_cols - 1) // n_cols
        
        # 统一所有图像的尺寸
        target_height = 400
        resized_images = []
        for img in result_images:
            aspect = img.shape[1] / img.shape[0]
            target_width = int(target_height * aspect)
            resized = cv2.resize(img, (target_width, target_height))
            resized_images.append(resized)
        
        # 找到最大宽度
        max_width = max(img.shape[1] for img in resized_images)
        
        # 将所有图像填充到相同宽度
        padded_images = []
        for img in resized_images:
            if img.shape[1] < max_width:
                padding = np.ones((target_height, max_width - img.shape[1], 3), 
                                dtype=np.uint8) * 255
                img = np.hstack([img, padding])
            padded_images.append(img)
        
        # 创建网格
        rows = []
        for i in range(n_rows):
            start_idx = i * n_cols
            end_idx = min(start_idx + n_cols, n_results)
            row_images = padded_images[start_idx:end_idx]
            
            # 如果最后一行不够，用空白填充
            while len(row_images) < n_cols:
                blank = np.ones((target_height, max_width, 3), dtype=np.uint8) * 255
                row_images.append(blank)
            
            row = np.hstack(row_images)
            rows.append(row)
        
        # 垂直堆叠所有行
        grid = np.vstack(rows)
        
        return grid
    
    def add_match_info_overlay(self, image: np.ndarray, 
                              total_matches: int, inliers: int,
                              global_similarity: float) -> np.ndarray:
        """
        在图像上添加详细的匹配信息覆盖层
        
        Args:
            image: 目标图像
            total_matches: 总匹配点数
            inliers: 内点数量
            global_similarity: 全局相似度
            
        Returns:
            添加了信息的图像
        """
        img = image.copy()
        
        # 准备信息文本
        info_lines = [
            f"Total Matches: {total_matches}",
            f"Inliers: {inliers}",
            f"Global Similarity: {global_similarity:.3f}"
        ]
        
        # 绘制每行信息
        y_offset = 70
        for line in info_lines:
            self._draw_text_with_background(img, line, (10, y_offset))
            y_offset += 35
        
        return img
