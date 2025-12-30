"""
几何校验模块
使用RANSAC算法进行特征匹配的几何一致性验证，估计单应性变换矩阵
"""
import cv2
import numpy as np
from typing import Tuple, Optional, List


class GeometricVerifier:
    """几何校验器，使用RANSAC算法验证特征匹配的几何一致性"""
    
    def __init__(self, ransac_reproj_threshold=5.0, ransac_max_iters=2000, 
                 ransac_confidence=0.995, min_inliers=8):
        """
        初始化几何校验器
        
        Args:
            ransac_reproj_threshold: RANSAC重投影误差阈值（像素）
            ransac_max_iters: RANSAC最大迭代次数
            ransac_confidence: RANSAC置信度
            min_inliers: 最小内点数量阈值
        """
        self.ransac_reproj_threshold = ransac_reproj_threshold
        self.ransac_max_iters = ransac_max_iters
        self.ransac_confidence = ransac_confidence
        self.min_inliers = min_inliers
    
    def verify_matches(self, keypoints1: List, keypoints2: List, 
                      matches: List) -> Tuple[int, Optional[np.ndarray], Optional[np.ndarray]]:
        """
        使用RANSAC验证匹配点的几何一致性
        
        Args:
            keypoints1: 第一张图像的关键点列表
            keypoints2: 第二张图像的关键点列表
            matches: 匹配点对列表
            
        Returns:
            (inlier_count, homography_matrix, inlier_mask) 元组
            - inlier_count: 内点数量（几何校验分数）
            - homography_matrix: 单应性变换矩阵 (3x3)
            - inlier_mask: 内点掩码，标记哪些匹配是内点
        """
        if len(matches) < 4:
            return 0, None, None
        
        # 提取匹配点的坐标
        src_pts = np.float32([keypoints1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        
        # 使用RANSAC估计单应性矩阵
        try:
            homography, mask = cv2.findHomography(
                src_pts, 
                dst_pts,
                cv2.RANSAC,
                self.ransac_reproj_threshold,
                maxIters=self.ransac_max_iters,
                confidence=self.ransac_confidence
            )
            
            if homography is None:
                return 0, None, None
            
            # 统计内点数量
            inlier_count = int(np.sum(mask))
            
            return inlier_count, homography, mask
            
        except Exception as e:
            print(f"RANSAC几何校验失败: {e}")
            return 0, None, None
    
    def compute_bounding_box(self, query_shape: Tuple[int, int], 
                           homography: np.ndarray) -> Optional[np.ndarray]:
        """
        计算查询图像在目标图像上的投影边界框
        
        Args:
            query_shape: 查询图像的形状 (height, width)
            homography: 单应性变换矩阵
            
        Returns:
            投影后的边界框坐标 [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
        """
        if homography is None:
            return None
        
        h, w = query_shape[:2]
        
        # 查询图像的四个角点
        corners = np.float32([
            [0, 0],
            [w, 0],
            [w, h],
            [0, h]
        ]).reshape(-1, 1, 2)
        
        # 使用单应性矩阵变换角点
        try:
            transformed_corners = cv2.perspectiveTransform(corners, homography)
            return transformed_corners.reshape(-1, 2).astype(np.int32)
        except Exception as e:
            print(f"边界框变换失败: {e}")
            return None
    
    def calculate_geometric_score(self, inlier_count: int, 
                                  total_matches: int) -> float:
        """
        计算几何校验分数
        
        Args:
            inlier_count: 内点数量
            total_matches: 总匹配点数量
            
        Returns:
            几何分数 [0, 100]
        """
        if total_matches == 0:
            return 0.0
        
        # 基础分数：内点比例
        ratio = inlier_count / total_matches
        base_score = ratio * 50
        
        # 绝对内点数量加成
        if inlier_count >= self.min_inliers:
            # 使用对数函数平滑增长
            bonus = min(50, 50 * np.log1p(inlier_count - self.min_inliers) / np.log1p(100))
        else:
            bonus = 0
        
        score = base_score + bonus
        return min(100.0, score)
    
    def calculate_hybrid_score(self, global_similarity: float, 
                              inlier_count: int,
                              total_matches: int = None) -> float:
        """
        计算混合评分（全局特征相似度 + 几何校验分数）
        优先考虑几何验证质量，全局相似度作为辅助
        
        Args:
            global_similarity: 全局特征余弦相似度 [0, 1]
            inlier_count: 几何校验内点数量
            total_matches: 总匹配点数量（可选）
            
        Returns:
            混合分数 [0, 100]
        """
        # 如果几何校验失败
        if inlier_count < self.min_inliers:
            # 主要依靠全局相似度，但分数不超过50
            return min(50.0, global_similarity * 50)
        
        # 几何校验成功 - 几何质量占主导
        # 1. 内点数量分数（0-60分）- 使用对数函数平滑增长
        inlier_score = min(60, 60 * np.log1p(inlier_count) / np.log1p(100))
        
        # 2. 内点比例加成（0-20分）
        if total_matches and total_matches > 0:
            ratio = inlier_count / total_matches
            ratio_bonus = ratio * 20
        else:
            ratio_bonus = 0
        
        # 3. 全局相似度调整（0-20分）
        global_bonus = global_similarity * 20
        
        # 最终分数 = 内点分数 + 比例加成 + 全局加成
        final_score = inlier_score + ratio_bonus + global_bonus
        
        return min(100.0, final_score)
    
    def is_valid_homography(self, homography: np.ndarray, 
                           min_determinant=0.01, max_determinant=100.0) -> bool:
        """
        检查单应性矩阵是否有效
        
        Args:
            homography: 单应性矩阵
            min_determinant: 行列式最小值（默认0.01，允许10倍缩小）
            max_determinant: 行列式最大值（默认100.0，允许10倍放大）
            
        Returns:
            是否有效
        """
        if homography is None:
            return False
        
        # 检查矩阵尺寸
        if homography.shape != (3, 3):
            return False
        
        # 检查行列式（避免过度变形）
        # 行列式 = scale_x * scale_y（近似）
        # 允许范围：0.01-100，即支持10倍缩放（缩略图、高清图等场景）
        det = np.linalg.det(homography[:2, :2])
        if det < min_determinant or det > max_determinant:
            return False
        
        return True
