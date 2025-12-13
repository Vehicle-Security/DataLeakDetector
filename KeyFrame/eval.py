import cv2
import os
import shutil
import time
import json
from typing import List, Dict

def calculate_iou(pred_start: float, pred_end: float, gt_start: float, gt_end: float) -> float:
    """
    计算IoU（交并比）
    
    公式: IoU = 交集时长 / 并集时长
    """
    # 计算交集
    intersection_start = max(pred_start, gt_start)
    intersection_end = min(pred_end, gt_end)
    
    # 如果没有交集
    if intersection_start >= intersection_end:
        return 0.0
    
    # 计算并集
    union_start = min(pred_start, gt_start)
    union_end = max(pred_end, gt_end)
    
    # 计算长度
    intersection_duration = intersection_end - intersection_start
    union_duration = union_end - union_start
    
    # 计算IoU
    iou = intersection_duration / union_duration
    
    return iou

def evaluate_single_video(video_path: str, ground_truth_path: str) -> Dict:
    """
    评估单个视频的敏感操作检测
    
    Args:
        video_path: 视频文件路径
        ground_truth_path: 真实标注文件路径
    
    Returns:
        评估结果字典
    """
    print(f"{'='*60}")
    print(f"🎬 开始评估视频: {os.path.basename(video_path)}")
    print(f"{'='*60}")
    
    # 记录开始时间
    start_time = time.time()
    
    # 1. 加载真实标注
    print("📋 加载真实标注...")
    if os.path.exists(ground_truth_path):
        with open(ground_truth_path, 'r', encoding='utf-8') as f:
            ground_truths = json.load(f)
        print(f"✅ 加载了 {len(ground_truths)} 个真实标注")
        
        # 显示真实标注
        print("\n📝 真实标注:")
        for i, gt in enumerate(ground_truths, 1):
            print(f"  {i}. 应用: {gt['app_name']}, 时间: {gt['start_time']:.1f}s - {gt['end_time']:.1f}s")
    else:
        print(f"❌ 真实标注文件不存在: {ground_truth_path}")
        return None
    
    # 2. 准备输出目录
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    output_dir = f"./{video_name}_output"
    
    if os.path.exists(output_dir):
        print(f"🧹 清理旧输出目录...")
        shutil.rmtree(output_dir)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 3. 运行检测算法
    print(f"\n🔍 开始检测敏感操作...")
    
    # 提取关键帧
    all_frames = keyframe_extract_stream_segment(video_path, output_dir, similarity_threshold=0.97)
    
    if not all_frames:
        print("❌ 没有提取到任何关键帧")
        return None
    
    print(f"✅ 提取了 {len(all_frames)} 个关键帧")
    
    # 均匀选择帧用于VLM分析
    selected_frames = select_uniform_frames(all_frames, step=3)
    print(f"✅ 选择了 {len(selected_frames)} 帧用于VLM分析")
    
    # VLM分析
    print("🤖 进行VLM分析...")
    batch_result = batch_analyze_frames_with_vlm(selected_frames)
    
    if batch_result is None or 'frame_details' not in batch_result:
        print("❌ VLM分析失败")
        return None
    
    # 聚类和边界扩展
    print("🔄 聚类和扩展边界...")
    clustered_ops = cluster_sensitive_operations(batch_result['frame_details'])
    extended_operations = extend_operation_boundaries(clustered_ops, all_frames)
    
    detection_time = time.time() - start_time
    print(f"✅ 检测完成，用时: {detection_time:.2f}秒")
    print(f"🔍 检测到 {len(extended_operations)} 个敏感操作")
    
    # 显示检测结果
    print("\n📊 检测结果:")
    for i, op in enumerate(extended_operations, 1):
        print(f"  {i}. 应用: {op['app_name']}, 时间: {op['extended_start_time']:.1f}s - {op['extended_end_time']:.1f}s")
    
    # 4. 匹配检测结果和真实标注（基于应用名称）
    print(f"\n{'='*60}")
    print("📈 开始评估匹配...")
    print(f"{'='*60}")
    
    matches = []  # 成功匹配的对
    unmatched_detections = []  # 未匹配的检测结果
    unmatched_groundtruths = []  # 未匹配的真实标注
    
    # 记录哪些真实标注已被匹配
    gt_matched = [False] * len(ground_truths)
    
    for det_idx, detection in enumerate(extended_operations):
        det_app = detection['app_name']
        det_start = detection['extended_start_time']
        det_end = detection['extended_end_time']
        
        best_match_idx = -1
        best_iou = 0
        best_gt = None
        
        # 寻找相同应用名称的真实标注
        for gt_idx, gt in enumerate(ground_truths):
            if gt_matched[gt_idx]:  # 跳过已匹配的
                continue
                
            if det_app == gt['app_name']:
                # 计算IoU
                iou = calculate_iou(det_start, det_end, gt['start_time'], gt['end_time'])
                
                if iou > best_iou:
                    best_iou = iou
                    best_match_idx = gt_idx
                    best_gt = gt
        
        # 记录匹配结果
        if best_match_idx != -1 and best_iou > 0:
            matches.append({
                'detection_idx': det_idx,
                'detection_app': det_app,
                'detection_time': [det_start, det_end],
                'gt_idx': best_match_idx,
                'gt_app': best_gt['app_name'],
                'gt_time': [best_gt['start_time'], best_gt['end_time']],
                'iou': best_iou
            })
            gt_matched[best_match_idx] = True
        else:
            unmatched_detections.append({
                'idx': det_idx,
                'app_name': det_app,
                'time': [det_start, det_end]
            })
    
    # 找出未匹配的真实标注
    for gt_idx, matched in enumerate(gt_matched):
        if not matched:
            gt = ground_truths[gt_idx]
            unmatched_groundtruths.append({
                'idx': gt_idx,
                'app_name': gt['app_name'],
                'time': [gt['start_time'], gt['end_time']]
            })
    
    # 5. 计算评估指标
    
    
    avg_iou = sum(match['iou'] for match in matches) / len(matches) if matches else 0
    
    # 6. 输出评估结果
    
    
    if matches:
        print(f"\n📊 匹配详情 (IoU > 0):")
        for i, match in enumerate(matches, 1):
            print(f"  {i}. 应用: {match['detection_app']}")
            print(f"     预测: {match['detection_time'][0]:.1f}s - {match['detection_time'][1]:.1f}s")
            print(f"     真实: {match['gt_time'][0]:.1f}s - {match['gt_time'][1]:.1f}s")
            print(f"     IoU: {match['iou']:.4f}")
    
    if unmatched_detections:
        print(f"\n⚠️ 未匹配的检测结果 ({len(unmatched_detections)} 个):")
        for det in unmatched_detections:
            print(f"  应用: {det['app_name']}, 时间: {det['time'][0]:.1f}s - {det['time'][1]:.1f}s")
    
    if unmatched_groundtruths:
        print(f"\n❌ 未检测到的真实操作 ({len(unmatched_groundtruths)} 个):")
        for gt in unmatched_groundtruths:
            print(f"  应用: {gt['app_name']}, 时间: {gt['time'][0]:.1f}s - {gt['time'][1]:.1f}s")
    
    # 7. 显示评估指标
    total_time = time.time() - start_time
    
    print(f"\n{'='*60}")
    print("📈 评估指标汇总")
    print(f"{'='*60}")
    print(f"🔢 统计数字:")
    print(f"   总检测数: {len(extended_operations)}")
    print(f"   总真实标注数: {len(ground_truths)}")

    print(f"   平均IoU: {avg_iou:.4f}")
    
    print(f"\n⏱️ 时间统计:")
    print(f"   检测用时: {detection_time:.2f}秒")
    print(f"   总运行时间: {total_time:.2f}秒")
    
    # 8. 保存结果到文件
    results = {
        'video_name': os.path.basename(video_path),
        'evaluation_time': time.strftime('%Y-%m-%d %H:%M:%S'),
        'total_time_seconds': total_time,
        'detection_time_seconds': detection_time,
        'detections': [
            {
                'app_name': op['app_name'],
                'start_time': op['extended_start_time'],
                'end_time': op['extended_end_time'],
                'operation_type': op.get('operation_type', '')
            } for op in extended_operations
        ],
        'ground_truths': ground_truths,
        'matches': matches,
        'unmatched_detections': unmatched_detections,
        'unmatched_groundtruths': unmatched_groundtruths,
        'metrics': {
            'avg_iou': avg_iou
        }
    }
    
    result_file = os.path.join(output_dir, f"{video_name}_evaluation.json")
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    # 保存易读的文本报告
    text_file = os.path.join(output_dir, f"{video_name}_evaluation_report.txt")
    with open(text_file, 'w', encoding='utf-8') as f:
        f.write(f"{'='*60}\n")
        f.write(f"视频评估报告: {video_name}\n")
        f.write(f"{'='*60}\n\n")
        
        f.write("评估指标:\n")
        f.write(f"  平均IoU: {avg_iou:.4f}\n\n")
        
        f.write("统计数字:\n")
        f.write(f"  总检测数: {len(extended_operations)}\n")
        f.write(f"  总真实标注数: {len(ground_truths)}\n")
        
        f.write("时间统计:\n")
        f.write(f"  检测用时: {detection_time:.2f}秒\n")
        f.write(f"  总运行时间: {total_time:.2f}秒\n\n")
        
        if matches:
            f.write("成功匹配的操作:\n")
            for i, match in enumerate(matches, 1):
                f.write(f"\n  {i}. 应用: {match['detection_app']}\n")
                f.write(f"      预测时间: {match['detection_time'][0]:.1f}s - {match['detection_time'][1]:.1f}s\n")
                f.write(f"      真实时间: {match['gt_time'][0]:.1f}s - {match['gt_time'][1]:.1f}s\n")
                f.write(f"      IoU: {match['iou']:.4f}\n")
        
        if unmatched_detections:
            f.write(f"\n未匹配的检测结果 ({len(unmatched_detections)} 个):\n")
            for det in unmatched_detections:
                f.write(f"  应用: {det['app_name']}, 时间: {det['time'][0]:.1f}s - {det['time'][1]:.1f}s\n")
        
        if unmatched_groundtruths:
            f.write(f"\n未检测到的真实操作 ({len(unmatched_groundtruths)} 个):\n")
            for gt in unmatched_groundtruths:
                f.write(f"  应用: {gt['app_name']}, 时间: {gt['time'][0]:.1f}s - {gt['time'][1]:.1f}s\n")
    
    print(f"\n💾 评估结果已保存到:")
    print(f"   JSON文件: {result_file}")
    print(f"   文本报告: {text_file}")
    print(f"{'='*60}")
    
    return results

def main():
    """主函数"""
    # 配置路径
    video_path = "/home/tjl/projects/video/1.mov"
    ground_truth_path = "/home/tjl/projects/video/1_gt.json"
    
    print("🎯 单视频评估配置:")
    print(f"   视频文件: {video_path}")
    print(f"   真实标注: {ground_truth_path}")
    
    # 检查文件是否存在
    if not os.path.exists(video_path):
        print(f"❌ 视频文件不存在: {video_path}")
        return
    
    if not os.path.exists(ground_truth_path):
        print(f"⚠️ 真实标注文件不存在，创建示例文件...")
        
        
    # 运行评估
    results = evaluate_single_video(video_path, ground_truth_path)
    
    return results

if __name__ == "__main__":
    # 导入原有模块
    from frame_processor import keyframe_extract_stream_segment, select_uniform_frames
    from vlm_analyzer import batch_analyze_frames_with_vlm, cluster_sensitive_operations
    from boundary_detector import extend_operation_boundaries
    
    main()