import cv2
import os
import shutil
import numpy as np
from tqdm import tqdm
from boundary_detector import OperationBoundaryDetector

# === 配置参数 ===
SSIM_THRESHOLD = 0.97
SAMPLE_INTERVAL = 1
JPEG_QUALITY = 90
SCALE_FACTOR = 0.77

def compress_frame(frame):
    """压缩帧图片以减少大小"""
    target_size = (720, 720)
    frame_resized = cv2.resize(frame, target_size)
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]
    _, buffer = cv2.imencode('.jpg', frame_resized, encode_param)
    return buffer, frame_resized.shape[:2]

def get_video_duration(video_path):
    """获取视频总时长（秒）"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"无法打开视频: {video_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0
    
    cap.release()
    return duration

def keyframe_extract_stream_segment(video_path, output_dir, similarity_threshold=0.95):
    """流式处理：提取整个视频的关键帧"""
    segment_output_dir = os.path.join(output_dir, "keyframes")
    if os.path.exists(segment_output_dir):
        shutil.rmtree(segment_output_dir)
    os.makedirs(segment_output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"无法打开视频: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    start_frame = 0
    end_frame = total_frames
    step = int(fps * SAMPLE_INTERVAL)
    
    print(f"  提取整个视频关键帧: 帧 0 - {end_frame}")

    feature_extractor = OperationBoundaryDetector()
    prev_feature = None
    kept_count = 0
    kept_frames = []

    pbar = tqdm(total=end_frame-start_frame, desc=f"关键帧提取")
    frame_idx = start_frame

    while frame_idx <= end_frame:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret or frame_idx > end_frame:
            break

        current_feature = feature_extractor.extract_features(frame)

        if prev_feature is not None:
            similarity = feature_extractor.calculate_similarity(prev_feature, current_feature)
            if similarity >= similarity_threshold:
                frame_idx += step
                pbar.update(step)
                continue
        
        prev_feature = current_feature
        kept_count += 1
        
        frame_info = {
            'frame': frame.copy(),
            'frame_index': frame_idx,
            'timestamp': frame_idx / fps,
        }
        kept_frames.append(frame_info)
        
        save_path = os.path.join(segment_output_dir, f"keyframe_{frame_idx:06d}.jpg")
        cv2.imwrite(save_path, frame)

        frame_idx += step
        pbar.update(step)

    pbar.close()
    cap.release()
    print(f"  ✅ 保存了 {kept_count} 张去重后的关键帧")
    return kept_frames

def select_uniform_frames(frames, step=3):
    """从帧列表中按固定步长选择帧"""
    total_frames = len(frames)
    if total_frames == 0:
        return []
    
    indices = list(range(0, total_frames, step))
    selected_frames = [frames[i] for i in indices]
    
    print(f"📊 固定步长选择: 从 {total_frames} 帧中选择 {len(selected_frames)} 帧")
    print(f"  固定步长: {step}")
    print(f"  选择帧索引: {[frames[i]['frame_index'] for i in indices]}")
    
    return selected_frames

def save_scene_keyframes(extended_operations, all_frames, output_dir):
    """将每个场景的关键帧存入不同的文件夹中"""
    print(f"\n{'='*60}")
    print("🖼️  开始保存每个场景的关键帧")
    print(f"{'='*60}")
    
    scene_keyframes_dir = os.path.join(output_dir, "scene_keyframes")
    os.makedirs(scene_keyframes_dir, exist_ok=True)
    
    frame_map = {frame['frame_index']: frame for frame in all_frames}
    saved_scenes = 0
    
    for i, operation in enumerate(extended_operations):
        print(f"\n📁 处理第 {i+1} 个场景:")
        print(f"   应用: {operation['app_name']}")
        print(f"   操作: {operation['operation_type']}")
        print(f"   时间区间: {operation['extended_start_time']:.1f}s - {operation['extended_end_time']:.1f}s")
        
        app_name_clean = "".join(c for c in operation['app_name'] if c.isalnum() or c in (' ', '-', '_')).rstrip()
        op_type_clean = "".join(c for c in operation['operation_type'] if c.isalnum() or c in (' ', '-', '_')).rstrip()
        
        scene_folder_name = f"scene_{i+1:02d}_{app_name_clean}_{op_type_clean}"
        scene_folder_path = os.path.join(scene_keyframes_dir, scene_folder_name)
        
        if os.path.exists(scene_folder_path):
            shutil.rmtree(scene_folder_path)
        os.makedirs(scene_folder_path)
        
        start_frame = operation['extended_start_frame']
        end_frame = operation['extended_end_frame']
        
        scene_frames = []
        for frame_index in range(start_frame, end_frame + 1):
            if frame_index in frame_map:
                scene_frames.append(frame_map[frame_index])
        
        if not scene_frames and 'original_frames' in operation:
            for frame_info in operation['original_frames']:
                frame_index = frame_info['frame_index']
                if frame_index in frame_map:
                    scene_frames.append(frame_map[frame_index])
        
        if scene_frames:
            print(f"   📸 找到 {len(scene_frames)} 个关键帧")
            
            for j, frame_info in enumerate(scene_frames):
                frame = frame_info['frame']
                frame_index = frame_info['frame_index']
                timestamp = frame_info['timestamp']
                
                filename = f"frame_{j+1:03d}_index{frame_index}_time{timestamp:.1f}s.jpg"
                filepath = os.path.join(scene_folder_path, filename)
                cv2.imwrite(filepath, frame)
            
            info_filepath = os.path.join(scene_folder_path, "scene_info.txt")
            with open(info_filepath, 'w', encoding='utf-8') as f:
                f.write(f"场景ID: {operation['group_id']}\n")
                f.write(f"应用名称: {operation['app_name']}\n")
                f.write(f"操作类型: {operation['operation_type']}\n")
                f.write(f"开始时间: {operation['extended_start_time']:.2f}s\n")
                f.write(f"结束时间: {operation['extended_end_time']:.2f}s\n")
                f.write(f"持续时间: {operation['extended_duration']:.2f}s\n")
                f.write(f"开始帧: {operation['extended_start_frame']}\n")
                f.write(f"结束帧: {operation['extended_end_frame']}\n")
                f.write(f"关键帧数量: {len(scene_frames)}\n")
                if operation.get('roi_coords'):
                    f.write(f"ROI坐标: {operation['roi_coords']}\n")
            
            saved_scenes += 1
            print(f"   ✅ 成功保存到: {scene_folder_name}")
        else:
            print(f"   ⚠️ 未找到该场景的关键帧")
    
    print(f"\n📊 场景关键帧保存完成:")
    print(f"   总场景数: {len(extended_operations)}")
    print(f"   成功保存: {saved_scenes}")
    print(f"   保存位置: {scene_keyframes_dir}")
    
    return scene_keyframes_dir