"""
全局特征提取模块
使用预训练的ResNet-50模型提取图像的全局特征向量，并计算余弦相似度
"""
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import cv2
from typing import Union


class GlobalFeatureExtractor:
    """全局特征提取器，基于ResNet-50"""
    
    def __init__(self, model_name='resnet50'):
        """
        初始化全局特征提取器
        
        Args:
            model_name: 使用的预训练模型名称，默认为resnet50
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 加载预训练模型
        if model_name == 'resnet50':
            try:
                # 尝试使用新版本API
                from torchvision.models import ResNet50_Weights
                self.model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
            except (ImportError, AttributeError):
                # 回退到旧版本API
                self.model = models.resnet50(pretrained=True)
            # 移除最后的全连接层，保留全局平均池化后的2048维特征
            self.model = nn.Sequential(*list(self.model.children())[:-1])
            self.feature_dim = 2048
        else:
            raise ValueError(f"不支持的模型: {model_name}")
        
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # 图像预处理转换
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    def extract_features(self, image: Union[str, Image.Image, np.ndarray]) -> np.ndarray:
        """
        提取图像的全局特征向量
        
        Args:
            image: 输入图像，可以是文件路径、PIL Image对象或numpy数组
            
        Returns:
            归一化后的特征向量 (2048维)
        """
        # 加载和预处理图像
        if isinstance(image, str):
            img = Image.open(image).convert('RGB')
        elif isinstance(image, np.ndarray):
            # OpenCV 图像是 BGR 格式，需要转换为 RGB
            if len(image.shape) == 3 and image.shape[2] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(image).convert('RGB')
        elif isinstance(image, Image.Image):
            img = image.convert('RGB')
        else:
            raise ValueError("不支持的图像格式")
        
        # 应用预处理
        img_tensor = self.transform(img).unsqueeze(0).to(self.device)
        
        # 提取特征
        with torch.no_grad():
            features = self.model(img_tensor)
        
        # 展平特征向量并转换为numpy数组
        features = features.squeeze().cpu().numpy()
        
        # L2归一化
        features = features / (np.linalg.norm(features) + 1e-8)
        
        return features
    
    @staticmethod
    def cosine_similarity(features1: np.ndarray, features2: np.ndarray) -> float:
        """
        计算两个特征向量之间的余弦相似度
        
        Args:
            features1: 第一个特征向量
            features2: 第二个特征向量
            
        Returns:
            余弦相似度分数 [0, 1]，越接近1越相似
        """
        # 确保向量已归一化
        features1 = features1 / (np.linalg.norm(features1) + 1e-8)
        features2 = features2 / (np.linalg.norm(features2) + 1e-8)
        
        # 计算余弦相似度（范围 [-1, 1]）
        similarity = np.dot(features1, features2)
        
        # 裁剪到 [0, 1]，负值表示完全不相关，直接设为0
        similarity = max(0.0, float(similarity))
        
        return similarity
