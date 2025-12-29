import cv2
import os
import shutil
import time
from frame_processor import keyframe_extract_stream_segment, select_uniform_frames, save_scene_keyframes
from vlm_analyzer import batch_analyze_frames_with_vlm, cluster_sensitive_operations
from boundary_detector import extend_operation_boundaries

def process_video(video_path, output_base_dir):
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    output_dir = os.path.join(output_base_dir, video_name)
    
    print(f"\n{'='*80}")
    print(f"🎬 正在处理视频: {video_name}")
    print(f"📄 文件路径: {video_path}")
    print(f"📂 输出目录: {output_dir}")
    print(f"{'='*80}")
    
    # 清理旧输出
    if os.path.exists(output_dir):
        try:
            shutil.rmtree(output_dir)
        except Exception as e:
            print(f"⚠️ 清理目录 {output_dir} 失败: {e}")
    os.makedirs(output_dir, exist_ok=True)
    
    start_time = time.time()
    
    try:
        # 提取关键帧
        print(f"[{video_name}] 提取关键帧...")
        all_frames = keyframe_extract_stream_segment(video_path, output_dir, similarity_threshold=0.97)
        
        if not all_frames:
            print(f"❌ [{video_name}] 没有提取到任何关键帧")
            return False
            
        # 均匀选择帧用于VLM分析
        selected_frames = select_uniform_frames(all_frames, step=3)
        print(f"[{video_name}] 选择 {len(selected_frames)} 帧用于分析")
        
        # 批量VLM分析
        print(f"[{video_name}] 开始VLM分析...")
        batch_result = batch_analyze_frames_with_vlm(selected_frames)
        
        if batch_result is None or 'frame_details' not in batch_result:
            print(f"❌ [{video_name}] VLM分析失败")
            return False
            
        # 聚类和边界扩展
        clustered_ops = cluster_sensitive_operations(batch_result['frame_details'])
        extended_operations = extend_operation_boundaries(clustered_ops, all_frames)
        
        # 保存场景关键帧
        scene_keyframes_dir = save_scene_keyframes(extended_operations, all_frames, output_dir)
        
        duration = time.time() - start_time
        print(f"✅ [{video_name}] 处理完成! 耗时: {duration:.1f}s")
        print(f"   识别操作数: {len(extended_operations)}")
        
        # 保存简易报告
        report_path = os.path.join(output_dir, "report.txt")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(f"视频: {video_name}\n")
            f.write(f"处理耗时: {duration:.1f}s\n")
            f.write(f"识别操作数: {len(extended_operations)}\n\n")
            for op in extended_operations:
                f.write(f"组: {op.get('group_id')}, 应用: {op.get('app_name')}, 操作: {op.get('operation_type')}, 时间: {op.get('extended_start_time'):.1f}-{op.get('extended_end_time'):.1f}s\n")
                
        return True
        
    except Exception as e:
        print(f"❌ [{video_name}] 处理异常: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    test_dir = "./test"
    output_base_dir = "./output"
    
    if not os.path.exists(test_dir):
        print(f"❌ 测试目录不存在: {test_dir}")
        return
        
    # 查找视频文件
    video_extensions = ('.mov', '.mp4', '.avi', '.mkv')
    video_files = [
        os.path.join(test_dir, f) 
        for f in os.listdir(test_dir) 
        if f.lower().endswith(video_extensions)
    ]
    
    if not video_files:
        print(f"❌ 在 {test_dir} 中未找到视频文件")
        return
        
    print(f"🔍 找到 {len(video_files)} 个视频文件待处理")
    
    successful = 0
    failed = 0
    
    for video_path in video_files:
        if process_video(video_path, output_base_dir):
            successful += 1
        else:
            failed += 1
            
    print(f"\n{'='*80}")
    print(f"📊 批量测试完成")
    print(f"✅ 成功: {successful}")
    print(f"❌ 失败: {failed}")
    print(f"📂 总结果目录: {os.path.abspath(output_base_dir)}")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()
