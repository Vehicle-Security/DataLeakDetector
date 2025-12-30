"""
核心处理引擎
整合所有模块，实现两阶段筛选架构的完整处理流程
"""
import os
import json
import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import faiss
import imagehash
from PIL import Image

from global_feature_extractor import GlobalFeatureExtractor
from local_feature_extractor import LocalFeatureExtractor
from geometric_verifier import GeometricVerifier
from video_processor import VideoProcessor
from file_manager import FileManager
from visualizer import Visualizer


class ProcessingEngine:
    """核心处理引擎，实现变形图片搜索的完整流程"""
    
    def __init__(self, base_dir: str = None, 
                 top_k: int = 20,
                 extraction_fps: int = 10,
                 processing_fps: int = 1,
                 phash_threshold: int = 5,
                 enable_phash: bool = False):
        """
        初始化处理引擎
        
        Args:
            base_dir: 基础工作目录
            top_k: 候选集大小（保留全局相似度最高的K个帧）
            extraction_fps: 视频帧提取帧率（建议10fps）
            processing_fps: 实际处理帧率（从提取的帧中每秒选取几帧处理，建议1fps）
            phash_threshold: pHash汉明距离阈值，小于此值认为帧相似（建议5-10）
            enable_phash: 是否启用pHash去重优化（默认False，因为可能影响准确率）
        """
        self.top_k = top_k
        self.extraction_fps = extraction_fps
        self.processing_fps = processing_fps
        self.phash_threshold = phash_threshold
        self.enable_phash = enable_phash
        
        # 初始化各个组件
        print("正在初始化全局特征提取器...")
        self.global_extractor = GlobalFeatureExtractor()
        
        print("正在初始化局部特征提取器...")
        self.local_extractor = LocalFeatureExtractor(use_root_sift=True)
        
        print("正在初始化几何校验器...")
        self.geometric_verifier = GeometricVerifier()
        
        print("正在初始化视频处理器...")
        self.video_processor = VideoProcessor(fps=extraction_fps)
        
        print("正在初始化文件管理器...")
        self.file_manager = FileManager(base_dir)
        
        print("正在初始化可视化器...")
        self.visualizer = Visualizer()
        
        print("处理引擎初始化完成！")
    
    def process_query(self, query_image_path: str, video_path: str, 
                     threshold: float = 50.0,
                     progress_callback=None) -> Dict:
        """
        处理查询请求的主函数
        
        Args:
            query_image_path: 查询图像路径
            video_path: 视频文件路径
            threshold: 匹配率阈值 [0, 100]
            progress_callback: 进度回调函数
            
        Returns:
            处理结果字典
        """
        # 报告进度
        def report_progress(message: str, percentage: float = None):
            if progress_callback:
                progress_callback(message, percentage)
            print(f"[{percentage}%] {message}" if percentage else message)
        
        report_progress("开始处理查询请求...", 0)
        
        # 1. 保存输入图像 (0-3%)
        report_progress("保存输入图像...", 1)
        query_filename = os.path.basename(query_image_path)
        saved_query_path = self.file_manager.save_input_image(
            query_image_path, query_filename
        )
        
        # 2. 提取查询图特征 (3-8%)
        report_progress("提取查询图特征...", 3)
        query_global_features = self.global_extractor.extract_features(saved_query_path)
        query_local_kp, query_local_desc = self.local_extractor.extract_features(
            saved_query_path
        )
        
        if query_local_desc is None:
            return {
                "status": "error",
                "message": "查询图像中未检测到足够的特征点"
            }
        
        report_progress(f"查询图特征：全局维度={len(query_global_features)}, "
                       f"局部关键点={len(query_local_kp)}", 8)
        
        # 3. 保存输入视频并提取帧 (8-15%)
        report_progress("处理视频...", 8)
        video_name = os.path.basename(video_path)
        
        # 保存视频到inputs目录
        saved_video_path = self.file_manager.save_input_video(video_path, video_name)
        report_progress(f"视频已保存: {video_name}", 10)
        
        # 获取帧目录，检查是否已存在
        frames_dir, frames_exist = self.file_manager.get_frames_dir_for_video(video_name, check_existing=True)
        
        if frames_exist:
            # 使用已存在的帧
            report_progress("发现已缓存的视频帧，跳过提取...", 12)
            # 读取已存在的帧文件
            import glob
            frame_files = sorted(glob.glob(os.path.join(frames_dir, 'frame_*.jpg')))
            if len(frame_files) == 0:
                frame_files = sorted(glob.glob(os.path.join(frames_dir, 'frame_*.png')))
            
            # 构建frame_list
            frame_list = []
            for idx, frame_path in enumerate(frame_files):
                timestamp = idx / self.video_processor.fps
                frame_list.append((timestamp, frame_path))
            
            report_progress(f"使用缓存的 {len(frame_list)} 帧", 15)
        else:
            # 需要提取帧
            report_progress("提取视频帧...", 10)
            try:
                frame_list = self.video_processor.extract_frames(
                    saved_video_path, frames_dir
                )
            except Exception as e:
                return {
                    "status": "error",
                    "message": f"视频帧提取失败: {str(e)}"
                }
            
            report_progress(f"成功提取 {len(frame_list)} 帧", 15)
        
        if len(frame_list) == 0:
            return {
                "status": "error",
                "message": "未能从视频中提取任何帧"
            }
        
        # 4. 第一阶段：全局特征快速筛选 (15-60%)
        report_progress("阶段1: 全局特征筛选...", 15)
        
        # 动态调整 top_k：根据视频长度自动增加候选集
        # 基础值 + 每100帧增加5个候选
        dynamic_top_k = min(self.top_k + len(frame_list) // 100 * 5, len(frame_list))
        
        """ # 如果视频较短（少于500帧），跳过全局筛选，直接对所有帧进行匹配
        if len(frame_list) <= 500:
            print(f"视频较短({len(frame_list)}帧)，跳过全局筛选，对所有帧进行几何匹配")
            dynamic_top_k = len(frame_list) """
        
        report_progress(f"视频总帧数: {len(frame_list)}, 候选集大小: {dynamic_top_k}", 15)
        

        candidates = self._global_screening(
            query_global_features, frame_list, progress_callback, top_k=dynamic_top_k
        )
        
        report_progress(f"筛选出 {len(candidates)} 个候选帧", 60)
        
        # 5. 第二阶段：几何校验和重排序 (60-92%)
        report_progress("阶段2: 几何校验...", 60)
        final_results = self._geometric_verification(
            saved_query_path, query_local_kp, query_local_desc,
            candidates, progress_callback
        )
        
        report_progress(f"几何校验完成，得到 {len(final_results)} 个结果", 92)
        
        # 6. 阈值过滤 (92-93%)
        filtered_results = [r for r in final_results if r['score'] >= threshold]
        
        # 确保过滤后的结果仍按分数降序排列
        filtered_results.sort(key=lambda x: x['score'], reverse=True)
        
        report_progress(f"阈值过滤后保留 {len(filtered_results)} 个匹配", 93)
        
        # 打印调试信息
        print(f"\n======== 调试信息 ========")
        print(f"阈值设置: {threshold}")
        print(f"几何校验结果数: {len(final_results)}")
        print(f"过滤后结果数: {len(filtered_results)}")
        if len(final_results) > 0:
            print(f"\n所有结果的分数:")
            for idx, r in enumerate(final_results):  # [:10]):  # 只显示前10个
                print(f"  {idx+1}. 分数={r['score']:.2f}, 内点={r['inliers']}, 匹配点={r['total_matches']}, 全局相似度={r['global_similarity']:.3f}")
        print(f"========================\n")
        
        # 7. 生成可视化结果 (93-98%)
        report_progress("生成可视化结果...", 93)
        output_results = self._generate_visualizations(
            saved_query_path, filtered_results, query_filename
        )
        
        # 8. 保存结果JSON (98-100%)
        report_progress("保存结果...", 98)
        
        # 构建所有结果的分数列表
        all_scores = []
        for idx, r in enumerate(filtered_results):
            all_scores.append({
                'rank': idx + 1,
                'timestamp_sec': round(r['timestamp'], 2),
                'score': round(r['score'], 2),
                'global_similarity': round(r['global_similarity'], 3),
                'inliers': r['inliers'],
                'total_matches': r['total_matches']
            })
        
        result_json = {
            "status": "success",
            "query_image": query_filename,
            "video_file": video_name,
            "threshold": threshold,
            "total_frames": len(frame_list),
            "candidates_screened": len(candidates),
            "matches_found": len(filtered_results),
            "all_scores": all_scores,
            "results": output_results
        }
        
        json_path = self.file_manager.create_result_json_path(query_filename)
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(result_json, f, indent=2, ensure_ascii=False)
        
        report_progress("处理完成！", 100)
        
        return result_json
    
    def _global_screening(self, query_features: np.ndarray, 
                         frame_list: List[Tuple[float, str]],
                         progress_callback=None,
                         top_k: int = None) -> List[Dict]:
        """
        第一阶段：使用全局特征进行快速筛选（优化版本）
        优化点1：使用pHash动态跳帧，避免对重复帧提取特征
        优化点2：使用Faiss向量搜索，替代Python循环的余弦相似度计算
        
        Args:
            query_features: 查询图的全局特征 (2048维)
            frame_list: 帧列表 [(timestamp, frame_path), ...]
            progress_callback: 进度回调
            top_k: 候选集大小（如果为None则使用self.top_k）
            
        Returns:
            候选帧列表，按相似度排序
        """
        if top_k is None:
            top_k = self.top_k
        
        import time
        start_time = time.time()
        
        # 获取全局特征维度
        feature_dim = len(query_features)
        
        # 1. 初始化Faiss索引（使用内积IndexFlatIP，适合归一化后的余弦相似度）
        index = faiss.IndexFlatIP(feature_dim)
        
        # 2. 初始化映射表和状态变量
        timestamp_map = []  # 索引ID -> 时间戳的映射
        frame_path_map = []  # 索引ID -> 帧路径的映射
        last_processed_hash = None  # 上一个处理的帧的pHash
        unique_frame_count = 0  # 实际处理的帧数
        skipped_frame_count = 0  # 跳过的重复帧数
        sampling_skipped_count = 0  # 采样跳过的帧数
        
        total_frames = len(frame_list)
        
        # 计算采样间隔
        sample_interval = self.extraction_fps // self.processing_fps if self.processing_fps > 0 else 1
        
        print(f"\n======== 优化版全局筛选开始 ========")
        print(f"总帧数: {total_frames}")
        print(f"提取帧率: {self.extraction_fps} fps")
        print(f"处理帧率: {self.processing_fps} fps (每 {sample_interval} 帧采样 1 帧)")
        print(f"pHash去重: {'启用' if self.enable_phash else '禁用（保证准确率）'}")
        if self.enable_phash:
            print(f"pHash阈值: {self.phash_threshold} (汉明距离)")
        print(f"全局特征维度: {feature_dim}")
        
        # 3. 对查询向量进行L2归一化（为了使用内积计算余弦相似度）
        query_vec_normalized = query_features.copy().reshape(1, -1).astype('float32')
        faiss.normalize_L2(query_vec_normalized)
        
        # 4. 主循环：遍历所有帧，采样 + 动态跳帧 + 即时建库
        for idx, (timestamp, frame_path) in enumerate(frame_list):
            try:
                # 4.1 【新增】帧采样逻辑：每隔 sample_interval 帧处理一次
                # 采样策略：在每个采样窗口中选择中间的帧（更稳定）
                frame_in_second = idx % sample_interval
                should_sample = (frame_in_second == sample_interval // 2)  # 选择每秒的中间帧
                
                if not should_sample:
                    sampling_skipped_count += 1
                    continue
                
                # 4.2 读取帧图像（支持中文路径）
                with open(frame_path, 'rb') as f:
                    img_array = np.frombuffer(f.read(), dtype=np.uint8)
                    frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                    if frame is None:
                        print(f"警告: 无法解码图像 {frame_path}")
                        continue
                
                # 4.3 判断是否需要pHash去重
                should_process = True
                if self.enable_phash:
                    # 【优化点1】计算当前帧的pHash
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    pil_image = Image.fromarray(frame_rgb)
                    current_hash = imagehash.phash(pil_image)
                    
                    # 判断是否跳过（与上一帧相似）
                    if last_processed_hash is None:
                        # 第一帧，必须处理
                        should_process = True
                    else:
                        # 计算汉明距离
                        hamming_distance = current_hash - last_processed_hash
                        if hamming_distance > self.phash_threshold:
                            # 内容变化显著，需要处理
                            should_process = True
                        else:
                            # 内容几乎一样，跳过
                            skipped_frame_count += 1
                            should_process = False
                
                # 4.3 如果需要处理这一帧
                if should_process:
                    # 4.4.1 【昂贵操作】提取ResNet-50全局特征
                    frame_features = self.global_extractor.extract_features(frame)
                    
                    # 4.4.2 【优化点2】对特征向量进行L2归一化
                    frame_vec_normalized = frame_features.copy().reshape(1, -1).astype('float32')
                    faiss.normalize_L2(frame_vec_normalized)
                    
                    # 4.4.3 【优化点2】添加到Faiss索引
                    index.add(frame_vec_normalized)
                    
                    # 4.4.4 记录映射关系
                    timestamp_map.append(timestamp)
                    frame_path_map.append(frame_path)
                    unique_frame_count += 1
                    
                    # 4.4.5 更新last_processed_hash（仅在启用pHash时）
                    if self.enable_phash:
                        last_processed_hash = current_hash
                
                # 4.5 报告进度 (15-60%之间动态更新，每5帧或每处理一帧更新一次)
                if progress_callback:
                    # 更频繁的进度更新：每5帧更新，或者每次实际处理帧时更新
                    if (idx + 1) % 5 == 0 or should_process:
                        # 根据实际遍历进度计算（15-58%）
                        progress = 15 + int(43 * (idx + 1) / total_frames)
                        progress_callback(
                            f"阶段1筛选: {idx + 1}/{total_frames} (已处理{unique_frame_count}帧)", 
                            progress
                        )
            
            except Exception as e:
                print(f"处理帧 {frame_path} 失败: {e}")
                continue
        
        extraction_time = time.time() - start_time
        
        print(f"\n--- 特征提取统计 ---")
        print(f"视频总帧数: {total_frames}")
        print(f"采样跳过帧数: {sampling_skipped_count} ({100*sampling_skipped_count/total_frames:.1f}%)")
        print(f"采样后帧数: {total_frames - sampling_skipped_count}")
        print(f"实际处理帧数: {unique_frame_count} ({100*unique_frame_count/total_frames:.1f}%)")
        if self.enable_phash and skipped_frame_count > 0:
            print(f"pHash跳过帧数: {skipped_frame_count} ({100*skipped_frame_count/(total_frames - sampling_skipped_count):.1f}%)")
        print(f"特征提取耗时: {extraction_time:.2f}秒")
        if unique_frame_count > 0:
            print(f"平均处理速度: {unique_frame_count/extraction_time:.1f} 帧/秒")
        
        # 5. 【优化点2】执行Faiss高效搜索
        if unique_frame_count == 0:
            print("警告: 没有处理任何帧！")
            return []
        
        # 5.1 报告Faiss搜索开始（58%）
        if progress_callback:
            progress_callback(f"阶段1筛选: Faiss向量搜索中...", 58)
        
        search_start = time.time()
        
        # 确保 top_k 不超过实际处理的帧数
        actual_top_k = min(top_k, unique_frame_count)
        
        # Faiss搜索：一次性计算所有余弦相似度并返回Top-K
        # scores: (1, K) 相似度分数（因为已归一化，内积=余弦相似度）
        # ids: (1, K) 索引ID
        scores, ids = index.search(query_vec_normalized, actual_top_k)
        
        search_time = time.time() - search_start
        
        # 5.2 报告Faiss搜索完成（60%）
        if progress_callback:
            progress_callback(f"阶段1筛选: 完成 (筛选出{actual_top_k}个候选)", 60)
        
        print(f"\n--- Faiss向量搜索统计 ---")
        print(f"搜索帧数: {unique_frame_count}")
        print(f"Top-K: {actual_top_k}")
        print(f"搜索耗时: {search_time*1000:.2f}毫秒")
        print(f"平均速度: {unique_frame_count/search_time:.0f} 次对比/秒")
        
        # 6. 整理结果
        candidates = []
        for rank, (score, idx_in_map) in enumerate(zip(scores[0], ids[0])):
            candidates.append({
                'timestamp': timestamp_map[idx_in_map],
                'frame_path': frame_path_map[idx_in_map],
                'global_similarity': float(score)  # 转换为Python float
            })
        
        # 7. 打印结果统计
        print(f"\n======== 全局筛选结果 ========")
        print(f"候选帧数: {len(candidates)}")
        if len(candidates) > 0:
            print(f"最高相似度: {candidates[0]['global_similarity']:.4f} (时间{candidates[0]['timestamp']:.2f}s)")
            print(f"最低相似度: {candidates[-1]['global_similarity']:.4f} (时间{candidates[-1]['timestamp']:.2f}s)")
            print(f"\n前20个候选帧:")
            for i, c in enumerate(candidates[:20]):
                print(f"  {i+1}. 时间{c['timestamp']:7.2f}s, 相似度={c['global_similarity']:.4f}")
            
            # 显示相似度分布统计
            similarities = [c['global_similarity'] for c in candidates]
            print(f"\n相似度分布:")
            print(f"  > 0.9: {sum(1 for s in similarities if s > 0.9)} 帧")
            print(f"  0.8-0.9: {sum(1 for s in similarities if 0.8 <= s <= 0.9)} 帧")
            print(f"  0.7-0.8: {sum(1 for s in similarities if 0.7 <= s < 0.8)} 帧")
            print(f"  0.6-0.7: {sum(1 for s in similarities if 0.6 <= s < 0.7)} 帧")
            print(f"  < 0.6: {sum(1 for s in similarities if s < 0.6)} 帧")

            # 检测全局特征质量
            max_sim = candidates[0]['global_similarity']
            if max_sim < 0.75:
                print(f"\n⚠️  警告: 最高全局相似度仅 {max_sim:.4f}，可能存在误判风险")
                print(f"    建议：降低匹配阈值或增加候选集大小")
        
        total_time = time.time() - start_time
        print(f"\n总耗时: {total_time:.2f}秒")
        if self.enable_phash and total_frames > 0:
            print(f"加速比: 原方案需处理{total_frames}帧，现在只处理{unique_frame_count}帧")
            print(f"        节省了 {100*(1-unique_frame_count/total_frames):.1f}% 的计算量")
        print(f"=============================\n")
        
        return candidates
    
    def _geometric_verification(self, query_image_path: str,
                               query_kp, query_desc,
                               candidates: List[Dict],
                               progress_callback=None) -> List[Dict]:
        """
        第二阶段：对候选帧进行几何校验
        
        Args:
            query_image_path: 查询图像路径
            query_kp: 查询图关键点
            query_desc: 查询图描述符
            candidates: 候选帧列表
            progress_callback: 进度回调
            
        Returns:
            验证后的结果列表，按最终分数排序
        """
        results = []
        total_candidates = len(candidates)
        
        # 读取查询图用于获取尺寸
        query_img = cv2.imdecode(np.frombuffer(open(query_image_path, 'rb').read(), np.uint8), cv2.IMREAD_COLOR)
        if query_img is None:
            raise ValueError(f"无法读取查询图像: {query_image_path}")
        query_shape = query_img.shape
        
        for idx, candidate in enumerate(candidates):
            try:
                # 提取候选帧的局部特征
                frame_kp, frame_desc = self.local_extractor.extract_features(
                    candidate['frame_path']
                )
                
                if frame_desc is None:
                    # 如果没有特征点，使用全局相似度作为分数
                    results.append({
                        'timestamp': candidate['timestamp'],
                        'frame_path': candidate['frame_path'],
                        'global_similarity': candidate['global_similarity'],
                        'inliers': 0,
                        'total_matches': 0,
                        'score': candidate['global_similarity'] * 50,
                        'bounding_box': None
                    })
                    continue
                
                # 特征匹配
                matches = self.local_extractor.match_features(
                    query_desc, frame_desc
                )
                
                if len(matches) < 4:
                    # 匹配点太少，无法进行几何校验
                    results.append({
                        'timestamp': candidate['timestamp'],
                        'frame_path': candidate['frame_path'],
                        'global_similarity': candidate['global_similarity'],
                        'inliers': 0,
                        'total_matches': len(matches),
                        'score': candidate['global_similarity'] * 50,
                        'bounding_box': None
                    })
                    continue
                
                # RANSAC几何校验
                inlier_count, homography, mask = self.geometric_verifier.verify_matches(
                    query_kp, frame_kp, matches
                )
                
                # 计算边界框
                bounding_box = None
                if homography is not None and self.geometric_verifier.is_valid_homography(homography):
                    bounding_box = self.geometric_verifier.compute_bounding_box(
                        query_shape, homography
                    )
                
                # 计算混合分数
                final_score = self.geometric_verifier.calculate_hybrid_score(
                    candidate['global_similarity'], inlier_count, len(matches)
                )
                
                results.append({
                    'timestamp': candidate['timestamp'],
                    'frame_path': candidate['frame_path'],
                    'global_similarity': candidate['global_similarity'],
                    'inliers': inlier_count,
                    'total_matches': len(matches),
                    'score': final_score,
                    'bounding_box': bounding_box
                })
                
                # 报告进度 (60-92%之间动态更新)
                if progress_callback:
                    progress = 60 + int(32 * (idx + 1) / total_candidates)
                    progress_callback(
                        f"阶段2验证: {idx + 1}/{total_candidates}",
                        progress
                    )
            
            except Exception as e:
                print(f"几何校验失败 {candidate['frame_path']}: {e}")
                continue
        
        # 按最终分数降序排序
        results.sort(key=lambda x: x['score'], reverse=True)
        
        return results
    
    def _generate_visualizations(self, query_image_path: str,
                                results: List[Dict],
                                query_name: str) -> List[Dict]:
        """
        生成可视化结果图像
        
        Args:
            query_image_path: 查询图像路径
            results: 匹配结果列表
            query_name: 查询图像名称
            
        Returns:
            包含图像URL的结果列表
        """
        print(f"🎨 开始生成可视化: 共 {len(results)} 个结果")
        print(f"   查询图像: {query_image_path}")
        print(f"   查询名称: {query_name}")
        
        # 清空visualizations和raw_frames文件夹
        import shutil
        from pathlib import Path
        
        output_dir = Path(self.file_manager.get_output_dir_for_query(query_name))
        vis_dir = output_dir / 'visualizations'
        raw_dir = output_dir / 'raw_frames'
        
        if vis_dir.exists():
            print(f"   🗑️  清空可视化文件夹: {vis_dir}")
            shutil.rmtree(vis_dir)
        vis_dir.mkdir(parents=True, exist_ok=True)
        
        if raw_dir.exists():
            print(f"   🗑️  清空原始帧文件夹: {raw_dir}")
            shutil.rmtree(raw_dir)
        raw_dir.mkdir(parents=True, exist_ok=True)
        
        output_results = []
        
        # 使用成功计数器来命名文件,确保按分数排名
        success_count = 0
        
        for idx, result in enumerate(results):
            try:
                # print(f"  处理匹配 {idx+1}/{len(results)}...")
                # 读取帧图像 - 使用支持中文路径的方法
                frame_path = result['frame_path']
                # print(f"    帧路径: {frame_path}")
                with open(frame_path, 'rb') as f:
                    img_array = np.frombuffer(f.read(), dtype=np.uint8)
                    frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                    if frame is None:
                        print(f"    ❌ 无法解码帧图像")
                        continue
                    # print(f"    ✅ 帧图像读取成功: {frame.shape}")
                
                # 绘制可视化
                # print(f"    绘制匹配结果...")
                vis_image = self.visualizer.draw_match_result(
                    frame,
                    result['bounding_box'],
                    result['score'],
                    result['timestamp'],
                    query_image_path,
                    show_query=True
                )
                
                # 添加详细信息
                vis_image = self.visualizer.add_match_info_overlay(
                    vis_image,
                    result['total_matches'],
                    result['inliers'],
                    result['global_similarity']
                )
                
                # 使用成功计数器命名,确保文件名连续且按分数排序
                success_count += 1
                
                # 保存可视化结果
                result_filename = f"match_{success_count:03d}_score{result['score']:.1f}_t{result['timestamp']:.2f}s.jpg"
                # print(f"    保存可视化图像: {result_filename}")
                image_path = self.file_manager.save_result_image(
                    vis_image, query_name, result_filename
                )
                # print(f"    ✅ 可视化图像已保存: {image_path}")
                
                # 保存原始帧到raw_frames子目录
                raw_frame_filename = f"frame_{success_count:03d}_score{result['score']:.1f}_t{result['timestamp']:.2f}s.jpg"
                # print(f"    保存原始帧: {raw_frame_filename}")
                raw_frame_path = self.file_manager.save_raw_frame(
                    frame, query_name, raw_frame_filename
                )
                # print(f"    ✅ 原始帧已保存: {raw_frame_path}")
                
                # 构建结果条目
                output_results.append({
                    'timestamp_sec': round(result['timestamp'], 2),
                    'score': round(result['score'], 2),
                    'image_url': image_path,
                    'raw_frame_url': raw_frame_path,
                    'global_similarity': round(result['global_similarity'], 3),
                    'inliers': result['inliers'],
                    'total_matches': result['total_matches']
                })
            
            except Exception as e:
                import traceback
                print(f"❌ 生成可视化失败 (匹配 {idx+1}): {e}")
                print(traceback.format_exc())
                continue
        
        return output_results
