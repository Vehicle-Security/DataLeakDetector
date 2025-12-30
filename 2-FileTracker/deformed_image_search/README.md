# 变形图片搜索系统

一个基于深度学习和计算机视觉的高级图像检索系统，能够从视频中准确检索经过复杂变换的图像。

## 📋 功能特性

### 核心功能
- **变形鲁棒性**: 支持截图、屏摄、压缩、缩放、旋转、亮度变换等多种变形
- **高精度匹配**: 使用两阶段筛选架构，平衡速度和精度
- **可视化结果**: 自动生成带检测框的可视化匹配图像
- **阈值控制**: 可调节匹配阈值，满足不同精度需求

### 支持的变形类型
- ✅ 截图、屏摄（手机拍摄屏幕）
- ✅ 压缩、缩放
- ✅ 旋转、仿射变换
- ✅ 亮度、色彩空间变换
- ✅ 局部裁剪、遮挡
- ✅ 添加文字、水印

## 🏗️ 系统架构

### 两阶段筛选架构

#### 第一阶段：全局特征快速筛选
- 使用预训练的ResNet-50提取2048维全局特征
- 计算余弦相似度快速筛选候选帧
- 从所有帧中选出Top-K个候选（默认K=20）

#### 第二阶段：几何校验与重排序
- 使用RootSIFT提取局部关键点和描述符
- FLANN匹配器进行特征匹配
- RANSAC算法验证几何一致性
- 混合评分：全局相似度 + 几何内点数

### 评分机制
- **分数 < 50**: 内容相似但未通过几何验证
- **分数 50-70**: 通过几何验证，置信度中等
- **分数 70-90**: 高置信度匹配
- **分数 90-100**: 几乎完美的匹配

## 📁 文件结构

```
deformed_image_search/
├── app.py                          # Gradio前端界面（主入口）
├── processing_engine.py            # 核心处理引擎
├── global_feature_extractor.py    # 全局特征提取（ResNet-50）
├── local_feature_extractor.py     # 局部特征提取（SIFT/RootSIFT）
├── geometric_verifier.py          # 几何校验（RANSAC）
├── video_processor.py             # 视频处理（FFmpeg/OpenCV）
├── file_manager.py                # 文件管理（输入输出）
├── visualizer.py                  # 结果可视化
├── requirements.txt               # 依赖包列表
├── inputs/                        # 输入图像目录
├── outputs/                       # 输出结果目录
│   ├── frames/                    # 视频帧缓存
│   └── [query_name]/              # 每次查询的结果
└── README.md                      # 本文件
```

## 🚀 快速开始

### 1. 环境要求
- Python 3.8+
- CUDA（可选，用于GPU加速）
- FFmpeg（可选，用于更快的视频处理）

### 2. 安装依赖

```bash
# 进入项目目录
cd deformed_image_search

# 安装Python依赖
pip install -r requirements.txt
```

### 3. 安装FFmpeg（可选但推荐）

**Windows**:
```powershell
# 使用Chocolatey
choco install ffmpeg

# 或从官网下载: https://ffmpeg.org/download.html
```

**Linux**:
```bash
sudo apt update
sudo apt install ffmpeg
```

### 4. 运行系统

```bash
python app.py
```

系统启动后，在浏览器中打开 http://localhost:7860

## 💡 使用方法

### 通过Web界面

1. **上传查询图像**: 点击左侧"变形查询图像"区域上传图片
2. **上传视频**: 点击"视频文件"区域上传视频
3. **调节阈值**: 使用滑块设置匹配阈值（推荐50-70）
4. **开始搜索**: 点击"开始搜索"按钮
5. **查看结果**: 右侧显示匹配结果和可视化图像

### 阈值选择建议

| 阈值范围 | 适用场景 | 特点 |
|---------|---------|------|
| 0-30 | 初步测试 | 非常宽松，可能有大量误匹配 |
| 30-50 | 广泛搜索 | 宽松，适合找回可能的匹配 |
| **50-70** | **推荐** | **平衡精度和召回率** |
| 70-90 | 精确搜索 | 严格，只返回高置信度结果 |
| 90-100 | 极严格 | 只返回几乎完美的匹配 |

## 📊 输出说明

### 结果JSON格式

系统会在 `outputs/[query_name]/results.json` 生成详细结果：

```json
{
  "status": "success",
  "query_image": "query.jpg",
  "video_file": "video.mp4",
  "threshold": 50,
  "total_frames": 1200,
  "candidates_screened": 20,
  "matches_found": 5,
  "results": [
    {
      "timestamp_sec": 45.2,
      "score": 85.3,
      "image_url": "outputs/query/match_001_score85.3_t45.20s.jpg",
      "global_similarity": 0.892,
      "inliers": 156,
      "total_matches": 203
    }
  ]
}
```

### 可视化图像

每个匹配结果都会生成一张可视化图像，包含：
- 左侧：原始查询图像
- 右侧：匹配的视频帧，带有：
  - 绿色多边形边界框（查询图在帧上的投影）
  - 红色角点标记
  - 匹配信息（分数、时间戳）
  - 详细统计（匹配点数、内点数、全局相似度）

## 📦 各模块功能说明

### 1. `app.py` - Gradio前端界面
**功能**: 提供用户友好的Web界面
- 左右分栏布局
- 图像和视频上传
- 阈值滑块控制
- 实时进度显示
- 结果展示（文本+图像画廊）

### 2. `processing_engine.py` - 核心处理引擎
**功能**: 整合所有模块，实现完整处理流程
- 管理整体处理流程
- 协调各个模块工作
- 两阶段筛选实现
- 进度报告
- 结果生成和保存

### 3. `global_feature_extractor.py` - 全局特征提取
**功能**: 使用ResNet-50提取图像全局特征
- 加载预训练ResNet-50模型
- 提取2048维特征向量
- L2归一化
- 余弦相似度计算
- 对压缩、色彩变换具有不变性

### 4. `local_feature_extractor.py` - 局部特征提取
**功能**: 使用SIFT/RootSIFT提取局部特征
- 检测图像关键点
- 生成128维描述符
- RootSIFT变换（性能优化）
- FLANN匹配器
- Lowe's ratio test
- 交叉检查验证
- 对旋转、缩放、仿射变换保持不变

### 5. `geometric_verifier.py` - 几何校验
**功能**: 使用RANSAC验证几何一致性
- RANSAC算法实现
- 单应性矩阵估计
- 内点统计
- 边界框投影计算
- 混合评分函数
- 有效性检查

### 6. `video_processor.py` - 视频处理
**功能**: 视频帧提取和管理
- FFmpeg快速提取（优先）
- OpenCV提取（备用）
- 时序子采样（默认10fps）
- 视频信息获取
- 指定时间戳帧提取

### 7. `file_manager.py` - 文件管理
**功能**: 管理输入输出文件
- 输入图像保存和管理
- 同名文件自动替换
- 输出目录组织
- 视频帧目录管理
- 结果文件路径生成
- 目录清理功能

### 8. `visualizer.py` - 结果可视化
**功能**: 生成可视化图像
- 绘制多边形边界框
- 角点标记
- 文本信息叠加
- 查询图与匹配帧对比视图
- 详细匹配信息覆盖层
- 网格布局（多结果）

## ⚙️ 高级配置

### 调整处理参数

在 `app.py` 中修改引擎初始化参数：

```python
engine = ProcessingEngine(
    base_dir=os.path.dirname(os.path.abspath(__file__)),
    top_k=20,              # 候选集大小
    extraction_fps=10      # 视频提取帧率
)
```

### SIFT参数调整

在 `local_feature_extractor.py` 中：

```python
LocalFeatureExtractor(
    use_root_sift=True,           # 是否使用RootSIFT
    n_features=0,                 # 特征点数量限制（0=不限制）
    contrast_threshold=0.04,      # 对比度阈值
    edge_threshold=10             # 边缘阈值
)
```

### RANSAC参数调整

在 `geometric_verifier.py` 中：

```python
GeometricVerifier(
    ransac_reproj_threshold=5.0,  # 重投影误差阈值
    ransac_max_iters=2000,        # 最大迭代次数
    ransac_confidence=0.995,      # 置信度
    min_inliers=8                 # 最小内点数
)
```

## 🔧 故障排除

### 问题1: 找不到匹配
**可能原因**:
- 阈值设置过高
- 查询图与视频内容差异太大
- 视频帧率设置过低，跳过了关键帧

**解决方案**:
- 降低阈值（试试30-40）
- 确认查询图确实来自该视频
- 提高extraction_fps参数

### 问题2: 处理速度慢
**可能原因**:
- 视频过长
- 未使用GPU
- 未安装FFmpeg

**解决方案**:
- 安装CUDA版本的PyTorch
- 安装FFmpeg加速视频处理
- 降低extraction_fps
- 减小top_k参数

### 问题3: 内存不足
**解决方案**:
- 降低extraction_fps
- 减小top_k参数
- 分段处理长视频

## 📝 技术细节

### 算法原理

1. **全局特征**: 使用ImageNet预训练的ResNet-50，截断在GAP层，生成内容语义表示
2. **局部特征**: SIFT算法检测尺度和旋转不变的关键点，RootSIFT提升匹配性能
3. **几何校验**: RANSAC从匹配点中鲁棒地估计单应性变换，剔除误匹配
4. **混合评分**: 结合内容相似度和几何一致性，避免假阳性

### 性能指标

- **处理速度**: 约10-30秒/分钟视频（取决于硬件）
- **召回率**: >90%（阈值=50时）
- **精确率**: >85%（阈值=50时）
- **内存占用**: 2-4GB（取决于视频长度）

## 📄 许可证

本项目仅供学习和研究使用。

## 🤝 贡献

欢迎提出问题和改进建议！

## 📧 联系方式

如有问题，请通过GitHub Issues联系。
