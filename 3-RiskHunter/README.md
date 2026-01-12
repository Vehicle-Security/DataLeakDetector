# 模块3 - RiskHunter (上传检测系统)

## 概述

模块3实现了基于LangGraph的智能上传检测Agent，用于分析文件上传行为，识别黑名单应用的敏感文件外发行为并触发报警。

## 功能特性

- ✅ **自动识别文件上传行为**：分析日志和视频，检测所有文件上传/外发操作
- ⚠️ **黑名单应用报警**：对黑名单应用的敏感文件上传触发严重报警
- ℹ️ **非黑名单应用记录**：检测但不报警非黑名单应用的上传行为
- 🔄 **动态worklist更新**：通过模块2自动发现隐藏行为并更新敏感文件列表
- 📊 **详细分析报告**：输出JSON格式和文本格式的完整分析报告

## 系统架构

```
模块3 (RiskHunter)
├── 配置层
│   └── upload_detection_config.py    # 敏感资源和黑白名单配置
├── 状态层
│   └── upload_detector_state.py      # LangGraph状态定义
├── 节点层
│   └── upload_detector_nodes.py      # 分析流程各节点实现
├── 图层
│   └── upload_detector_graph.py      # LangGraph图定义
└── 应用层
    └── example_upload_detection.py   # 使用示例
```

## 模块间调用关系

```
模块3 (RiskHunter)
    ↓ 调用
模块2 (FileTracker)
    ↓ 调用
模块1 (FrameAnalyzer)
```

**重要**：模块3不直接调用模块1，而是通过模块2获取模块1的分析结果，避免重复调用造成成本浪费。

## 快速开始

### 1. 配置敏感资源和黑白名单

编辑 `upload_detection_config.py`：

```python
class UploadDetectionConfig:
    def __init__(self):
        # 敏感文件列表
        self.sensitive_files = [
            "/Users/xxx/Documents/项目1详细规划.docx",
            "/Users/xxx/Documents/项目2需求分析.docx",
            # 添加更多敏感文件...
        ]
        
        # 黑名单应用（检测到上传会报警）
        self.blacklist_apps = [
            "微信", "WeChat",
            "钉钉", "DingTalk",
            "QQ",
            "个人邮箱", "Gmail",
            "百度网盘", "OneDrive",
            # 添加更多黑名单应用...
        ]
        
        # 白名单应用（上传不报警）
        self.whitelist_apps = [
            "企业微信", "WeCom",
            "Slack",
            "企业邮箱", "Outlook",
            # 添加更多白名单应用...
        ]
```

### 2. 准备输入文件

确保以下文件存在：

```
records/{record_id}/
├── key_events/
│   └── key_events_*.json          # 日志文件
├── video/
│   └── *.mp4                      # 录屏视频
└── INDEX.md                       # 录屏开始时间
```

### 3. 运行分析

```bash
cd /home/dxy/Projects/DataLeakDetector/3-RiskHunter
python example_upload_detection.py
```

### 4. 查看结果

结果保存在 `records/{record_id}/upload_detection_results/`：

```
upload_detection_results/
├── full_state_20260109_123456.json     # 完整状态
├── alerts_20260109_123456.json         # 报警事件（黑名单）
├── info_events_20260109_123456.json    # 信息事件（非黑名单）
└── report_20260109_123456.txt          # 简要报告
```

## 输入输出

### 输入

| 参数 | 说明 | 来源 |
|------|------|------|
| log_file | 日志文件（JSON格式） | 用户提供路径 |
| video_path | 录屏视频文件 | 用户提供路径 |
| index_path | INDEX.md文件（包含录屏开始时间） | 用户提供路径 |
| sensitive_files | 敏感资源列表 | upload_detection_config.py |
| blacklist_apps | 黑名单应用列表 | upload_detection_config.py |
| whitelist_apps | 白名单应用列表 | upload_detection_config.py |

### 输出

#### 1. 报警事件 (alerts_*.json)

黑名单应用的敏感文件上传行为：

```json
{
  "record_id": 42,
  "timestamp": "20260109_123456",
  "total_alerts": 2,
  "alerts": [
    {
      "event_id": "evt_001",
      "timestamp": "2025-12-28T18:42:10",
      "file_name": "项目2需求分析.pdf",
      "original_file": "/Users/xxx/Documents/项目2需求分析.docx",
      "app_name": "QQ",
      "app_category": "blacklist",
      "operation_type": "聊天转发",
      "should_alert": true,
      "alert_level": "critical",
      "alert_reason": "检测到黑名单应用 'QQ' 的文件外发行为",
      "description": "用户在QQ中将文件发送给联系人..."
    }
  ]
}
```

#### 2. 信息事件 (info_events_*.json)

非黑名单应用的上传行为（仅记录，不报警）：

```json
{
  "record_id": 42,
  "timestamp": "20260109_123456",
  "total_events": 1,
  "events": [
    {
      "app_name": "Chrome浏览器",
      "app_category": "unknown",
      "operation_type": "网页上传",
      "should_alert": false,
      "alert_level": "info",
      "alert_reason": "非黑名单应用 'Chrome浏览器' 的文件外发（仅记录）"
    }
  ]
}
```

## 工作流程

### LangGraph流程图

```
[initialize]
     ↓
[process_event] ← ─ ─ ─ ─ ─ ─ ┐
     ↓                        │
[analyze_upload]              │
     ↓                        │
  判断worklist是否为空         │
     ├─ 不为空 ─ ─ ─ ─ ─ ─ ─ ┘
     └─ 为空
         ↓
    [finalize]
         ↓
       [END]
```

### 详细步骤

1. **initialize节点**
   - 初始化WorklistManager
   - 扫描日志构建worklist
   - 发现所有敏感文件的相关事件

2. **process_event节点**
   - 从worklist获取下一个事件
   - 调用模块2分析事件（模块2会调用模块1）
   - 获取模块1的视频帧分析结果
   - 如果发现隐藏行为，动态更新worklist

3. **analyze_upload节点**
   - 分析是否为上传/外发行为
   - 判断应用类别（黑名单/白名单/未知）
   - 决定是否报警
   - 创建UploadEvent并分类存储

4. **finalize节点**
   - 生成统计报告
   - 保存结果到JSON和文本文件
   - 显示报警和信息事件

## 报警规则

| 应用类别 | 行为类别 | 是否报警 | 报警级别 | 说明 |
|---------|---------|---------|---------|------|
| 黑名单 | 直接外发 | ✅ 是 | critical | 严重威胁，需立即处理 |
| 黑名单 | 其他可疑行为 | ✅ 是 | warning | 可疑行为，需关注 |
| 白名单 | 任何行为 | ❌ 否 | info | 正常操作 |
| 未知 | 直接外发 | ❌ 否 | info | 仅记录，不报警 |

## API参考

### 配置类

#### `UploadDetectionConfig`

配置类，管理敏感资源和黑白名单。

**方法**：

- `is_sensitive_file(file_path: str) -> bool`: 判断是否为敏感文件
- `get_app_category(app_name: str) -> str`: 获取应用类别
- `should_alert(app_category: str, behavior_category: str) -> tuple[bool, str]`: 判断是否报警

### 状态类

#### `UploadDetectorState`

LangGraph状态定义，包含：

- `worklist_size`: worklist大小
- `processed_count`: 已处理事件数
- `upload_events`: 检测到的所有上传事件
- `alert_events`: 报警事件列表
- `info_events`: 信息事件列表
- `statistics`: 统计信息

#### `UploadEvent`

上传事件数据类：

```python
@dataclass
class UploadEvent:
    event_id: str
    timestamp: str
    file_path: str
    file_name: str
    original_file: str
    app_name: str
    app_category: str
    behavior_category: str
    operation_type: str
    time_range: str
    involved_timestamps: List[str]
    description: str
    should_alert: bool
    alert_level: str
    alert_reason: str
```

### 图构建

#### `create_upload_detector_graph()`

创建LangGraph上传检测图。

**返回**：编译后的LangGraph应用

## 测试用例

### 测试场景1：黑名单应用上传（应报警）

```python
# 场景：用户通过QQ发送敏感文档
- 敏感文件：项目2需求分析.docx
- 应用：QQ（黑名单）
- 操作：聊天转发
- 预期：触发critical报警
```

### 测试场景2：白名单应用上传（不报警）

```python
# 场景：用户通过企业微信分享文档
- 敏感文件：项目1详细规划.docx
- 应用：企业微信（白名单）
- 操作：文件分享
- 预期：不报警，记录为info级别
```

### 测试场景3：非黑名单应用上传（检测但不报警）

```python
# 场景：用户通过浏览器上传文件
- 敏感文件：项目3prd设计.docx
- 应用：Chrome浏览器（未知）
- 操作：网页上传
- 预期：检测到但不报警，记录为info级别
```

## 常见问题

### Q1: 如何添加新的敏感文件？

编辑 `upload_detection_config.py`，在 `self.sensitive_files` 列表中添加文件路径。

### Q2: 如何自定义黑白名单？

编辑 `upload_detection_config.py`，修改 `self.blacklist_apps` 和 `self.whitelist_apps` 列表。

### Q3: 为什么某些上传行为没有被检测到？

可能的原因：
1. 文件不在敏感文件列表中
2. 视频帧分析未能识别该操作
3. 检测关键词不匹配

可以调整 `detection_rules` 中的 `upload_keywords` 和 `upload_operations`。

### Q4: 如何避免误报？

1. 准确配置白名单应用
2. 调整报警规则的阈值
3. 分析info_events中的非黑名单上传行为，判断是否需要加入黑名单

### Q5: 模块3会重复调用模块1吗？

不会。模块3通过模块2调用模块1，每个事件只会被模块1分析一次。模块3从模块2的返回结果中获取模块1的分析结果，避免重复调用。

## 性能优化

1. **避免重复分析**：模块3不直接调用模块1，通过模块2获取结果
2. **批量处理**：worklist机制确保每个敏感文件事件只处理一次
3. **动态更新**：发现隐藏行为时才重新扫描日志

## 依赖项

```
langgraph>=0.0.1
```

其他依赖继承自模块1和模块2。

## 目录结构

```
3-RiskHunter/
├── upload_detection_config.py      # 配置文件
├── upload_detector_state.py        # 状态定义
├── upload_detector_nodes.py        # 节点实现
├── upload_detector_graph.py        # 图定义
├── example_upload_detection.py     # 使用示例
├── README.md                        # 本文档
└── records/                         # 分析记录
    └── {record_id}/
        └── upload_detection_results/  # 输出结果
```

