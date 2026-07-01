# 01-FrameAnalyzer/file_tracker 模块

## 概述

FileTracker 是 DataLeakDetector 系统的第二个模块，负责追踪敏感文件的流转和隐藏操作。

## 核心功能

### 1. Worklist 管理

动态维护敏感文件事件的工作列表，供模块3循环处理。

**主要特性：**
- ✅ 维护敏感文件列表
- ✅ 查询文件是否敏感
- ✅ 扫描日志构建 worklist
- ✅ 动态更新 worklist
- ✅ 追踪文件映射关系（原始 ↔ 派生）

**使用示例：**
```python
from worklist_manager import WorklistManager, load_log_from_json

# 初始化管理器
manager = WorklistManager(sensitive_files=[
    "/path/to/secret1.pdf",
    "/path/to/secret2.docx"
])

# 扫描日志构建 worklist
log_events = load_log_from_json("monitor_log.json")
manager.scan_and_build_worklist(log_events)

# 循环处理
while not manager.is_empty():
    event = manager.get_next_event()
    # 处理事件...
```

### 2. 隐藏行为分析

基于 LangGraph 的智能分析工作流，检测和追踪敏感文件的隐藏操作。

**识别的隐藏行为：**
- 🔹 重命名：文件名被改变
- 🔹 压缩：打包成压缩包（zip/rar）
- 🔹 格式转换：格式改变（docx → pdf）
- 🔹 目录移动：移动到隐蔽位置
- 🔹 复制粘贴：复制到其他位置

**工作原理：**
1. 调用模块1分析视频帧
2. 提取操作行为信息
3. 识别文件名/格式变化
4. 创建新的敏感事件
5. 更新 worklist 和文件映射

**使用示例：**
```python
from behavior_analysis_graph import analyze_sensitive_event_behavior

# 分析单个事件
result = analyze_sensitive_event_behavior(
    event=sensitive_event,
    index_path="INDEX.md",
    video_path="recording.mp4",
    worklist_manager=manager
)

if result.get("has_hidden_behavior"):
    print("发现隐藏行为！")
    for op in result["hidden_operations"]:
        print(f"{op['operation_type']}: {op['original_file']} → {op['new_file']}")
```

## 模块结构

```
01-FrameAnalyzer/file_tracker/
├── worklist_manager.py              # Worklist 管理器（核心）
├── behavior_analysis_graph.py       # 隐藏行为分析工作流（LangGraph）
├── behavior_analysis_state.py       # 状态定义
├── behavior_analysis_tools.py       # 分析工具
├── behavior_analysis_prompts.py     # Prompt 模板
├── run_behavior_analysis.py     # 完整使用示例
├── BEHAVIOR_ANALYSIS.md             # 详细文档
└── README.md                        # 本文件
```

## 与其他模块的集成

### 与模块1（FrameAnalyzer）集成

模块2调用模块1的 `analyze_video_behavior` 函数分析视频帧：

```python
from relavance_frame import analyze_video_behavior

result = analyze_video_behavior(
    rec_start_time_str='2026-01-05 17:48:33',
    search_start_time_str='2026-01-05 17:48:50',
    search_end_time_str='2026-01-05 17:49:20',
    target_keywords=['敏感文件.pdf'],
    video_path='recording.mp4'
)
```

### 与模块3（RiskHunter）集成

模块3作为主调用方：
1. 提供敏感文件列表
2. 提供日志、视频等资源
3. 循环调用 worklist 处理事件
4. 接收分析结果

## 快速开始

### 安装依赖

```bash
pip install langgraph langchain-openai python-dotenv
```

### 配置环境变量

创建 `.env` 文件：

```bash
MODEL_NAME=qwen2-vl-72b-instruct
OPENAI_BASE_URL=https://token-plan.cn-beijing.maas.aliyuncs.com/compatible-mode/v1
OPENAI_API_KEY=your_api_key_here
TEMPERATURE=0.01
```

### 运行示例

```bash
cd 01-FrameAnalyzer/file_tracker
python run_behavior_analysis.py
```

## 数据格式

### 输入：日志事件格式

```json
[
    {
        "timestamp": "2026-01-05 17:48:50",
        "event_type": "opened",
        "file_path": "/Users/dxy/Documents/secret.pdf",
        "process_info": {
            "app_name": "Preview",
            "pid": 12345
        }
    }
]
```

**支持的 event_type：**
- `created` - 文件创建
- `opened` - 文件打开
- `modified` - 文件修改
- `deleted` - 文件删除
- `moved` - 文件移动
- `renamed` - 文件重命名
- `upload_detected` - 检测到上传
- `file_selected` - 文件选择
- `app_switch` - 应用切换
- `website_visit` - 网站访问

### 输入：INDEX.md 格式

```markdown
**Recording Time**: 2026-01-05 17:48:33
```

### 输出：分析结果格式

```json
{
    "has_hidden_behavior": true,
    "hidden_operations": [
        {
            "operation_type": "格式转换",
            "original_file": "项目文档.docx",
            "new_file": "项目文档.pdf",
            "app_name": "iLovePDF",
            "time_range": "2026-01-05 17:49:00 - 17:49:20",
            "description": "用户将文件转换为PDF格式"
        }
    ],
    "file_mappings": [
        {
            "original": "项目文档.docx",
            "derived": "项目文档.pdf",
            "relationship": "格式转换"
        }
    ],
    "new_events": [...]
}
```

## API 文档

### WorklistManager

#### 构造函数
```python
manager = WorklistManager(sensitive_files: Optional[List[str]] = None)
```

#### 主要方法

**添加敏感文件：**
```python
manager.add_sensitive_file(file_path: str)
manager.add_sensitive_files(file_paths: List[str])
```

**查询敏感文件：**
```python
is_sensitive = manager.is_sensitive_file(file_path: str) -> bool
original = manager.get_original_file(file_path: str) -> Optional[str]
```

**构建和处理 worklist：**
```python
added_count = manager.scan_and_build_worklist(log_events: List[Dict]) -> int
event = manager.get_next_event() -> Optional[SensitiveFileEvent]
is_empty = manager.is_empty() -> bool
size = manager.size() -> int
```

**更新映射关系：**
```python
manager.update_file_mapping(original_file: str, new_file: str)
```

**统计信息：**
```python
stats = manager.get_statistics() -> Dict[str, Any]
```

### BehaviorAnalysisGraph

#### 分析事件
```python
from behavior_analysis_graph import analyze_sensitive_event_behavior

result = analyze_sensitive_event_behavior(
    event: SensitiveFileEvent,
    index_path: str,
    video_path: str,
    worklist_manager: WorklistManager
) -> Dict[str, Any]
```

## 典型工作流

```
1. 模块3提供敏感文件列表和日志
   ↓
2. WorklistManager 扫描日志构建 worklist
   ↓
3. 循环处理 worklist 中的事件
   ↓
4. 对每个事件调用 BehaviorAnalysisGraph
   ↓
5. Graph 调用模块1分析视频帧
   ↓
6. 识别隐藏行为并提取新文件
   ↓
7. 创建新事件并更新 worklist（动态）
   ↓
8. 更新文件映射关系
   ↓
9. 返回处理下一个事件（循环）
   ↓
10. worklist 为空，处理完成
```

## 注意事项

1. **文件路径格式**: 日志中的文件路径需要与敏感文件列表格式一致
2. **视频分析性能**: 调用模块1需要分析视频，可能耗时较长
3. **递归追踪**: 系统会递归追踪派生文件链（A→B→C）
4. **去重机制**: 已处理的事件不会重复处理

## 详细文档

更多信息请参阅：
- [BEHAVIOR_ANALYSIS.md](./BEHAVIOR_ANALYSIS.md) - 隐藏行为分析详细文档
- [run_behavior_analysis.py](./run_behavior_analysis.py) - 完整使用示例

## 开发者信息

- 模块版本: 2.0.0
- 依赖模块: 01-FrameAnalyzer
- 被依赖: 01-FrameAnalyzer/risk_hunter
- 技术栈: LangGraph, LangChain, OpenAI API
