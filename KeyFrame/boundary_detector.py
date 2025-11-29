import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import cv2
import numpy as np

# === 配置参数 ===
TAU_LOW = 0.88
K_STABLE_FRAMES = 3
ADJACENT_SIMILARITY_THRESHOLD = 0.98
MIN_PROGRESS_FRAMES = 1

class OperationBoundaryDetector:
    """操作边界检测器"""
    
    def __init__(self, device=None):
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._load_model()
        self.preprocess = self._get_preprocess()
        self.feature_cache = {}
        
    def _load_model(self):
        """加载预训练的ResNet-50模型"""
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        model = nn.Sequential(*list(model.children())[:-1])
        model.eval()
        model.to(self.device)
        return model
    
    def _get_preprocess(self):
        """获取图像预处理流程"""
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def extract_features(self, frame, roi_coords=None):
        """提取图像特征向量，支持ROI裁剪"""
        cache_key = (frame.tobytes(), tuple(roi_coords) if roi_coords else None)
        if cache_key in self.feature_cache:
            return self.feature_cache[cache_key]
        
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        if roi_coords is not None:
            height, width = frame.shape[:2]
            x1 = int(roi_coords[0] * width / 1000)
            y1 = int(roi_coords[1] * height / 1000)
            x2 = int(roi_coords[2] * width / 1000)
            y2 = int(roi_coords[3] * height / 1000)
            
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(width, x2), min(height, y2)
            
            if x2 > x1 and y2 > y1:
                image = image.crop((x1, y1, x2, y2))
        
        input_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            features = self.model(input_tensor)
            feature_vector = features.squeeze().cpu().numpy().flatten()
            
        self.feature_cache[cache_key] = feature_vector
        return feature_vector
        
    def calculate_similarity(self, feature1, feature2):
        """计算两个特征向量的余弦相似度"""
        if feature1 is None or feature2 is None:
            return 0.0
            
        feature1 = feature1.flatten().astype(np.float64)
        feature2 = feature2.flatten().astype(np.float64)
        
        norm1 = np.linalg.norm(feature1)
        norm2 = np.linalg.norm(feature2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        similarity = np.dot(feature1, feature2) / (norm1 * norm2)
        
        return float(similarity)
        
    def find_operation_boundaries(self, ref_frames, all_frames, roi_coords=None):
        """寻找操作的开始和结束帧"""
        print(f"🔍 寻找操作边界...")
        print(f"基准帧数量: {len(ref_frames)}, ROI: {roi_coords}")

        # 提取平均全局特征
        ref_features_list = [self.extract_features(frame_info['frame']) for frame_info in ref_frames]
        ref_feature_global = np.mean(ref_features_list, axis=0)

        all_frame_indices = [f['frame_index'] for f in all_frames]
        ref_frame_indices = [f['frame_index'] for f in ref_frames]
        ref_start_index = all_frame_indices.index(min(ref_frame_indices))
        ref_end_index = all_frame_indices.index(max(ref_frame_indices))

        # 向前搜索（寻找开始帧 T_start）
        print("\n向前搜索开始帧 (T_start)...")
        start_frame_index = ref_start_index
        
        start_feat = self.extract_features(all_frames[ref_start_index]['frame'])
        last_similarity = self.calculate_similarity(ref_feature_global, start_feat)
        
        low_similarity_counter = 0
        REQUIRED_LOW_SIMILARITY_FRAMES = 2 

        for i in range(ref_start_index - 1, -1, -1):
            current_frame_info = all_frames[i]
            current_feature = self.extract_features(current_frame_info['frame'])
            current_similarity = self.calculate_similarity(ref_feature_global, current_feature)

            if current_similarity < TAU_LOW:
                low_similarity_counter += 1
                if low_similarity_counter >= REQUIRED_LOW_SIMILARITY_FRAMES:
                    start_frame_index = i + REQUIRED_LOW_SIMILARITY_FRAMES
                    start_frame_index = min(start_frame_index, ref_start_index)
                    print(f"  帧 {current_frame_info['frame_index']}: 连续 {REQUIRED_LOW_SIMILARITY_FRAMES} 帧低于绝对阈值，T_start确定为 {all_frames[start_frame_index]['frame_index']}")
                    break
            else:
                low_similarity_counter = 0 
                start_frame_index = i
            
            last_similarity = current_similarity
        
        if start_frame_index < 0: start_frame_index = 0
        if low_similarity_counter > 0 and i == -1: start_frame_index = 0

        # 向后搜索（寻找结束帧 T_end）
        print("\n向后搜索结束帧 (T_end)...")
        end_frame_index = ref_end_index

        end_feat = self.extract_features(all_frames[ref_end_index]['frame'])
        last_similarity = self.calculate_similarity(ref_feature_global, end_feat)

        current_idx = ref_end_index + 1
        stable_frame_counter = 0 
        prev_roi_feature = self.extract_features(all_frames[ref_end_index]['frame'], roi_coords)
        frames_passed_since_ref_end = 0

        while current_idx < len(all_frames):
            current_frame_info = all_frames[current_idx]
            current_frame_index = current_frame_info['frame_index']
            current_feature_global = self.extract_features(current_frame_info['frame'])

            frames_passed_since_ref_end += 1

            # 全局相似度检查
            global_similarity = self.calculate_similarity(current_feature_global, ref_feature_global)
            
            if global_similarity < TAU_LOW:
                end_frame_index = current_idx - 1
                print(f"  帧 {current_frame_index}: 全局相似度低于绝对阈值，T_end确定为 {all_frames[end_frame_index]['frame_index']}")
                break

            last_similarity = global_similarity

            # 局部相邻相似度检查
            if frames_passed_since_ref_end < MIN_PROGRESS_FRAMES:
                current_roi_feature = self.extract_features(current_frame_info['frame'], roi_coords)
                prev_roi_feature = current_roi_feature
            else:
                current_roi_feature = self.extract_features(current_frame_info['frame'], roi_coords)
                s_adj_c = self.calculate_similarity(current_roi_feature, prev_roi_feature)

                if s_adj_c >= ADJACENT_SIMILARITY_THRESHOLD: 
                    stable_frame_counter += 1
                    if stable_frame_counter >= K_STABLE_FRAMES:
                        end_frame_index = current_idx - K_STABLE_FRAMES 
                        print(f"  帧 {current_frame_index}: 局部连续 {K_STABLE_FRAMES} 帧稳定，T_end确定为 {all_frames[end_frame_index]['frame_index']}")
                        break
                else:
                    stable_frame_counter = 0 

                prev_roi_feature = current_roi_feature

            current_idx += 1

        if current_idx == len(all_frames):
            end_frame_index = len(all_frames) - 1

        # 结果整合
        start_frame_info = all_frames[start_frame_index]
        end_frame_info = all_frames[end_frame_index]

        print(f"\n=== 操作边界检测结果 ===")
        print(f"开始帧: {start_frame_info['frame_index']}, 时间: {start_frame_info['timestamp']:.2f}s")
        print(f"结束帧: {end_frame_info['frame_index']}, 时间: {end_frame_info['timestamp']:.2f}s")

        return start_frame_info, end_frame_info

def extend_operation_boundaries(sensitive_operations, all_frames):
    """为每个敏感操作组扩展时间边界"""
    detector = OperationBoundaryDetector()
    extended_operations = []
    
    if not sensitive_operations:
        return extended_operations
    
    for group in sensitive_operations:
        print(f"\n🎯 处理操作组: {group['app_name']} - {group['operation_type']}")
        
        ref_frames = []
        for frame_info in group['frames']:
            original_frame = next(
                (f for f in all_frames if f['frame_index'] == frame_info['frame_index']), 
                None
            )
            if original_frame:
                ref_frames.append(original_frame)
        
        if not ref_frames:
            print(f"⚠️ 未找到基准帧，跳过该组")
            continue
        
        roi_coords = None
        for frame_info in group['frames']:
            if frame_info.get('roi_bbox') not in ([0, 0, 0, 0], None):
                roi_coords = frame_info['roi_bbox']
                break
        
        try:
            start_frame_info, end_frame_info = detector.find_operation_boundaries(
                ref_frames=ref_frames,
                all_frames=all_frames,
                roi_coords=roi_coords
            )
            
            extended_operation = {
                'group_id': group['group_id'],
                'app_name': group['app_name'],
                'operation_type': group['operation_type'],
                'original_frames': group['frames'],
                'extended_start_frame': start_frame_info['frame_index'],
                'extended_end_frame': end_frame_info['frame_index'],
                'extended_start_time': start_frame_info['timestamp'],
                'extended_end_time': end_frame_info['timestamp'],
                'extended_duration': end_frame_info['timestamp'] - start_frame_info['timestamp'],
                'roi_coords': roi_coords
            }
            
            extended_operations.append(extended_operation)
            
        except Exception as e:
            print(f"❌ 边界检测失败: {e}")
            original_timestamps = [f.get('timestamp', 0) for f in group['frames']]
            extended_operation = {
                'group_id': group['group_id'],
                'app_name': group['app_name'],
                'operation_type': group['operation_type'],
                'original_frames': group['frames'],
                'extended_start_frame': min([f['frame_index'] for f in group['frames']]),
                'extended_end_frame': max([f['frame_index'] for f in group['frames']]),
                'extended_start_time': min(original_timestamps),
                'extended_end_time': max(original_timestamps),
                'extended_duration': max(original_timestamps) - min(original_timestamps),
                'roi_coords': roi_coords,
                'note': '边界检测失败，使用原始范围'
            }
            extended_operations.append(extended_operation)
    
    return extended_operations