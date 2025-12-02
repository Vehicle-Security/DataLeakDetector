"""
局部特征提取模块
使用SIFT/RootSIFT算法提取图像的局部关键点和描述符
"""
import cv2
import numpy as np
from typing import Tuple, Union, Optional
from PIL import Image


class LocalFeatureExtractor:
    """局部特征提取器，基于SIFT/RootSIFT"""
    
    def __init__(self, use_root_sift=True, n_features=0, n_octave_layers=3, 
                 contrast_threshold=0.04, edge_threshold=10, sigma=1.6):
        """
        初始化局部特征提取器
        
        Args:
            use_root_sift: 是否使用RootSIFT（性能更优的SIFT变体）
            n_features: 保留的最佳特征数量，0表示不限制
            n_octave_layers: 每个金字塔层的层数
            contrast_threshold: 对比度阈值，用于过滤弱特征
            edge_threshold: 边缘阈值，用于去除边缘响应
            sigma: 高斯核的标准差
        """
        self.use_root_sift = use_root_sift
        
        # 创建SIFT检测器
        self.sift = cv2.SIFT_create(
            nfeatures=n_features,
            nOctaveLayers=n_octave_layers,
            contrastThreshold=contrast_threshold,
            edgeThreshold=edge_threshold,
            sigma=sigma
        )
    
    def extract_features(self, image: Union[str, Image.Image, np.ndarray]) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        提取图像的局部特征（关键点和描述符）
        
        Args:
            image: 输入图像，可以是文件路径、PIL Image对象或numpy数组
            
        Returns:
            (keypoints, descriptors) 元组
            - keypoints: 关键点数组，每个关键点包含位置、尺度、方向等信息
            - descriptors: 描述符数组，shape为(N, 128)，N为关键点数量
        """
        # 加载图像
        if isinstance(image, str):
            # 使用支持中文路径的方法
            with open(image, 'rb') as f:
                img_array = np.frombuffer(f.read(), dtype=np.uint8)
                img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                if img is None:
                    raise ValueError(f"无法读取图像: {image}")
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        elif isinstance(image, Image.Image):
            img = np.array(image.convert('RGB'))
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        elif isinstance(image, np.ndarray):
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image
        else:
            raise ValueError("不支持的图像格式")
        
        # 检测关键点和提取描述符
        keypoints, descriptors = self.sift.detectAndCompute(gray, None)
        
        if descriptors is None:
            return None, None
        
        # 应用RootSIFT变换
        if self.use_root_sift:
            descriptors = self._root_sift(descriptors)
        
        return keypoints, descriptors
    
    @staticmethod
    def _root_sift(descriptors: np.ndarray, eps=1e-7) -> np.ndarray:
        """
        将SIFT描述符转换为RootSIFT描述符
        RootSIFT通过L1归一化后取平方根，再进行L2归一化，能提供更好的匹配性能
        
        Args:
            descriptors: SIFT描述符数组
            eps: 避免除零的小常数
            
        Returns:
            RootSIFT描述符数组
        """
        # L1归一化
        descriptors /= (descriptors.sum(axis=1, keepdims=True) + eps)
        
        # 对每个元素取平方根
        descriptors = np.sqrt(descriptors)
        
        # L2归一化
        descriptors /= (np.linalg.norm(descriptors, axis=1, keepdims=True) + eps)
        
        return descriptors
    
    @staticmethod
    def match_features(descriptors1: np.ndarray, descriptors2: np.ndarray, 
                      ratio_threshold=0.75, cross_check=True) -> list:
        """
        使用FLANN匹配器匹配两组描述符
        
        Args:
            descriptors1: 第一组描述符
            descriptors2: 第二组描述符
            ratio_threshold: Lowe's ratio test的阈值
            cross_check: 是否进行交叉检查
            
        Returns:
            良好的匹配点对列表
        """
        if descriptors1 is None or descriptors2 is None:
            return []
        
        # 创建FLANN匹配器
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        
        # K-NN匹配（k=2用于ratio test）
        try:
            matches = flann.knnMatch(descriptors1, descriptors2, k=2)
        except:
            # 如果特征点太少，使用暴力匹配
            bf = cv2.BFMatcher()
            matches = bf.knnMatch(descriptors1, descriptors2, k=2)
        
        # 应用Lowe's ratio test
        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < ratio_threshold * n.distance:
                    good_matches.append(m)
        
        # 交叉检查（可选）
        if cross_check and len(good_matches) > 0:
            # 反向匹配
            try:
                matches_reverse = flann.knnMatch(descriptors2, descriptors1, k=2)
            except:
                bf = cv2.BFMatcher()
                matches_reverse = bf.knnMatch(descriptors2, descriptors1, k=2)
            
            good_matches_reverse = []
            for match_pair in matches_reverse:
                if len(match_pair) == 2:
                    m, n = match_pair
                    if m.distance < ratio_threshold * n.distance:
                        good_matches_reverse.append(m)
            
            # 只保留双向匹配的点
            cross_checked_matches = []
            for m1 in good_matches:
                for m2 in good_matches_reverse:
                    if m1.queryIdx == m2.trainIdx and m1.trainIdx == m2.queryIdx:
                        cross_checked_matches.append(m1)
                        break
            
            good_matches = cross_checked_matches
        
        return good_matches
