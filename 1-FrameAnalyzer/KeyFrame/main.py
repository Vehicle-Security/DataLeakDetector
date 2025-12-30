import cv2
import os
import shutil
from frame_processor import keyframe_extract_stream_segment, select_uniform_frames, save_scene_keyframes
from vlm_analyzer import batch_analyze_frames_with_vlm, cluster_sensitive_operations
from boundary_detector import extend_operation_boundaries

def main():
    # 步骤1: 直接处理整个视频
    # 步骤1: 查找并处理视频
    video_files = [f for f in os.listdir('.') if f.lower().endswith(('.mov', '.mp4', '.avi', '.mkv'))]
    if not video_files:
        print("❌ 未找到视频文件 (支持 .mov, .mp4, .avi, .mkv)")
        print("请将视频文件放入当前目录: " + os.getcwd())
        # 提供手动输入选项
        video_path = input("\n或者输入视频文件的完整路径 (留空退出): ").strip()
        if not video_path:
             return
        if not os.path.exists(video_path):
             print(f"❌ 文件不存在: {video_path}")
             return
    else:
        print(f"找到 {len(video_files)} 个视频文件, 默认使用第一个: {video_files[0]}")
        video_path = video_files[0]

    output_dir = "./output"
    
    if os.path.exists(output_dir):
        print(f"🧹 检测到旧输出目录，正在清理: {output_dir} ...")
        try:
            shutil.rmtree(output_dir)
        except Exception as e:
            print(f"⚠️ 清理目录失败 (可能是文件被占用): {e}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("🎬 开始处理整个视频...")
    
    # 提取整个视频的关键帧
    all_frames = keyframe_extract_stream_segment(video_path, output_dir, similarity_threshold=0.97)
    
    if not all_frames:
        print("❌ 没有提取到任何关键帧")
        return
    
    # 均匀选择帧用于VLM分析
    selected_frames = select_uniform_frames(all_frames, step=3)
    
    # 批量VLM分析
    print(f"🤖 开始VLM敏感操作原子级分析...")
    batch_result = batch_analyze_frames_with_vlm(selected_frames)
    
    if batch_result is None or 'frame_details' not in batch_result:
        print("❌ VLM分析失败或结构错误")
        return
    
    # 对原子结果进行聚类
    clustered_ops = cluster_sensitive_operations(batch_result['frame_details'])
    
    # 扩展操作边界
    print(f"\n🔄 开始操作边界扩展...")
    extended_operations = extend_operation_boundaries(clustered_ops, all_frames)
    
    # 输出结果
    print(f"\n🎯 敏感操作时间区间:")
    print("=" * 80)
    
    for operation in extended_operations:
        print(f"组 {operation['group_id']}. 应用: {operation['app_name']}")
        print(f"   操作: {operation['operation_type']}")
        print(f"   时间: {operation['extended_start_time']:.1f}s - {operation['extended_end_time']:.1f}s")
        print(f"   时长: {operation['extended_duration']:.1f}s")
        if operation['roi_coords']:
            print(f"   ROI坐标: {operation['roi_coords']}")
        print()
    
    # 保存场景关键帧
    scene_keyframes_dir = save_scene_keyframes(extended_operations, all_frames, output_dir)
    
    # 输出总结
    print(f"\n{'='*60}")
    print("🎬 处理完成总结")
    print(f"{'='*60}")
    print(f"总识别敏感操作: {len(extended_operations)} 个")
    
    # 显示场景关键帧文件夹
    print("\n📁 生成的场景关键帧文件夹:")
    for folder in os.listdir(scene_keyframes_dir):
        folder_path = os.path.join(scene_keyframes_dir, folder)
        if os.path.isdir(folder_path):
            jpg_count = len([f for f in os.listdir(folder_path) if f.endswith('.jpg')])
            print(f"  📂 {folder} ({jpg_count} 张关键帧)")

if __name__ == "__main__":
    main()