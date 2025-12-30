# -*- coding: utf-8 -*-
"""
关键帧识别模块 - 测试运行脚本

使用说明:
1. 确保已安装所有依赖
2. 将视频文件路径替换为你的实际路径
3. 运行: python test_keyframe.py
"""

import os
import sys

# 修复Windows控制台编码问题
if sys.platform == 'win32':
    import locale
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except:
        pass

def check_dependencies():
    """检查必要的依赖是否已安装"""
    required_packages = {
        'cv2': 'opencv-python',
        'torch': 'torch',
        'torchvision': 'torchvision',
        'PIL': 'Pillow',
        'tqdm': 'tqdm',
        'langchain_core': 'langchain-core',
        'langchain_openai': 'langchain-openai'
    }
    
    missing_packages = []
    
    for package, install_name in required_packages.items():
        try:
            __import__(package)
            print(f"✅ {install_name}")
        except ImportError:
            print(f"❌ {install_name} - 未安装")
            missing_packages.append(install_name)
    
    if missing_packages:
        print(f"\n⚠️ 缺少以下依赖包:")
        print(f"pip install {' '.join(missing_packages)}")
        return False
    
    return True

def find_video_files():
    """在当前目录及上级目录查找视频文件"""
    video_extensions = ['.mov', '.mp4', '.avi', '.mkv']
    video_files = []
    
    # 搜索当前目录
    for root, dirs, files in os.walk('.'):
        for file in files:
            if any(file.lower().endswith(ext) for ext in video_extensions):
                video_files.append(os.path.join(root, file))
        
        # 只搜索1层深度
        if root.count(os.sep) > 1:
            break
    
    return video_files

def main():
    print("="*60)
    print("🎬 关键帧识别模块 - 测试运行")
    print("="*60)
    
    # 1. 检查依赖
    print("\n📦 检查Python依赖...")
    if not check_dependencies():
        return
    
    # 2. 查找视频文件
    print("\n🔍 查找视频文件...")
    video_files = find_video_files()
    
    if video_files:
        print(f"找到 {len(video_files)} 个视频文件:")
        for i, video in enumerate(video_files, 1):
            print(f"  {i}. {video}")
    else:
        print("❌ 未找到视频文件")
        print("\n💡 提示: 请将视频文件放在当前目录或子目录中")
        print("   支持格式: .mov, .mp4, .avi, .mkv")
        
        # 提供手动输入选项
        video_path = input("\n请输入视频文件的完整路径 (留空退出): ").strip()
        
        if not video_path:
            print("退出程序")
            return
        
        if not os.path.exists(video_path):
            print(f"❌ 文件不存在: {video_path}")
            return
        
        video_files = [video_path]
    
    # 3. 选择视频
    if len(video_files) > 1:
        choice = input(f"\n请选择要处理的视频 (1-{len(video_files)}): ").strip()
        try:
            video_path = video_files[int(choice) - 1]
        except (ValueError, IndexError):
            print("无效的选择")
            return
    else:
        video_path = video_files[0]
    
    print(f"\n✅ 选择的视频: {video_path}")
    
    # 4. 运行主程序
    print("\n" + "="*60)
    print("🚀 开始处理...")
    print("="*60)
    
    try:
        # 导入主模块
        from frame_processor import keyframe_extract_stream_segment, select_uniform_frames, save_scene_keyframes
        from vlm_analyzer import batch_analyze_frames_with_vlm, cluster_sensitive_operations
        from boundary_detector import extend_operation_boundaries
        
        output_dir = "./output"
        
        # 清理旧输出
        import shutil
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        os.makedirs(output_dir, exist_ok=True)
        
        # 步骤1: 提取关键帧
        print("\n🎬 步骤1: 提取关键帧...")
        all_frames = keyframe_extract_stream_segment(video_path, output_dir, similarity_threshold=0.97)
        
        if not all_frames:
            print("❌ 没有提取到任何关键帧")
            return
        
        print(f"✅ 提取了 {len(all_frames)} 个关键帧")
        
        # 步骤2: 均匀采样
        print("\n📊 步骤2: 均匀采样...")
        selected_frames = select_uniform_frames(all_frames, step=3)
        print(f"✅ 选择了 {len(selected_frames)} 帧用于VLM分析")
        
        # 步骤3: VLM分析
        print("\n🤖 步骤3: VLM敏感操作分析...")
        print("⚠️ 这一步可能需要较长时间，取决于帧数和网络速度")
        
        batch_result = batch_analyze_frames_with_vlm(selected_frames)
        
        if batch_result is None or 'frame_details' not in batch_result:
            print("❌ VLM分析失败")
            return
        
        print(f"✅ VLM分析完成")
        
        # 步骤4: 聚类
        print("\n🔄 步骤4: 敏感操作聚类...")
        clustered_ops = cluster_sensitive_operations(batch_result['frame_details'])
        print(f"✅ 识别出 {len(clustered_ops)} 个敏感操作组")
        
        # 步骤5: 边界扩展
        print("\n🔄 步骤5: 操作边界扩展...")
        extended_operations = extend_operation_boundaries(clustered_ops, all_frames)
        print(f"✅ 边界扩展完成")
        
        # 输出结果
        print("\n" + "="*60)
        print("🎯 最终结果: 敏感操作时间区间")
        print("="*60)
        
        if not extended_operations:
            print("未检测到任何敏感操作")
        else:
            for operation in extended_operations:
                print(f"\n组 {operation['group_id']}. 应用: {operation['app_name']}")
                print(f"   操作: {operation['operation_type']}")
                print(f"   时间: {operation['extended_start_time']:.1f}s - {operation['extended_end_time']:.1f}s")
                print(f"   时长: {operation['extended_duration']:.1f}s")
                if operation['roi_coords']:
                    print(f"   ROI坐标: {operation['roi_coords']}")
        
        # 保存场景关键帧
        print("\n📁 步骤6: 保存场景关键帧...")
        scene_keyframes_dir = save_scene_keyframes(extended_operations, all_frames, output_dir)
        
        # 总结
        print("\n" + "="*60)
        print("🎬 处理完成总结")
        print("="*60)
        print(f"总识别敏感操作: {len(extended_operations)} 个")
        print(f"输出目录: {os.path.abspath(output_dir)}")
        
        # 显示场景关键帧文件夹
        print("\n📁 生成的场景关键帧文件夹:")
        for folder in os.listdir(scene_keyframes_dir):
            folder_path = os.path.join(scene_keyframes_dir, folder)
            if os.path.isdir(folder_path):
                jpg_count = len([f for f in os.listdir(folder_path) if f.endswith('.jpg')])
                print(f"  📂 {folder} ({jpg_count} 张关键帧)")
        
    except Exception as e:
        print(f"\n❌ 处理过程中出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
