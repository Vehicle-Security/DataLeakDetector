# Video Retrieval & Behavior Analysis Agent

这是一个基于多模态大模型（VLM）和计算机视觉技术的智能 Agent，旨在从海量录屏或监控视频中，根据关键词自动化检索特定操作行为，并生成结构化的行为分析报告。

## 项目架构

该 Agent 采用Pipeline设计，包含以下四个核心节点：


1. **视觉特征预处理 (Vision Preprocessing)**：利用 ResNet50 提取画面特征，通过余弦相似度过滤掉冗余静止画面。
2. **关键词 OCR 过滤 (Keyword Filtering)**：使用 EasyOCR 对关键帧进行文字识别，锁定包含目标关键词的画面。
3. **上下文扩展 (Context Extension)**：在命中关键词的时间点后，自动提取后续 3s、8s、15s 的画面以捕获完整的行为序列。
4. **VLM 行为意图分析 (Behavior Analysis)**：将所有关键画面送入 Qwen2.5-VL 模型，分析具体的操作应用、行为类别、文件名变更等深层信息。


---

## 快速开始

### 1. 安装依赖

Bash

```
pip install opencv-python torch torchvision pillow easyocr langchain-openai python-dotenv
```

### 2. 环境配置

在项目根目录创建 `.env` 文件，并配置你的 API 密钥：

Code snippet

```
DASHSCOPE_API_KEY=your_api_key_here
```

### 3. 文件结构

- `schema.py`: 定义 Agent 运行时的全局状态数据结构。
- `agent.py`: 核心逻辑实现，包含视觉筛选、OCR 匹配和 VLM 交互。
- `main.py`: 程序入口，负责配置参数并启动任务。
- `prompts.py`: 存放 VLM 所需的系统提示词模板。


---

## 使用示例

在 `main.py` 中配置视频路径、录制开始时间及搜索关键词，然后运行：

Python

```
from agent import VideoFileOperationAgent

agent = VideoFileOperationAgent()
result = agent.run({
    "video_path": "path/to/your/video.mp4",
    "keywords": ["项目1详细规划"],
    "rec_start": "2025-12-28 18:55:36",
    "search_start": "2025-12-28 18:55:46",
    "search_end": "2025-12-28 18:56:05"
})

print(result)
```


---

## 输出示例

```text
{
    "search_range": {
        "start": "2025-12-28 18:41:53",
        "end": "2025-12-28 18:42:10"
    },
    "total_events": 1,
    "events": [
        {
            "time_range": "2025-12-28 18:41:54 - 2025-12-28 18:42:06",
            "involved_timestamps": [
                "2025-12-28 18:41:54",
                "2025-12-28 18:41:58",
                "2025-12-28 18:42:01",
                "2025-12-28 18:42:06"
            ],
            "app_name": "iLovePDF",
            "behavior_category": "潜在隐藏行为",
            "operation_type": "格式转换",
            "original_filename": "项目2需求分析.docx",
            "modified_filename": "项目2需求分析.pdf",
            "description": "用户在 iLovePDF 网站上将 '项目2需求分析.docx' 文件转换为 PDF 格式。从选择文件到开始转换的过程被记录在连续的帧中。"
        }
    ],
    "status": "success"
}
```


---

## 输出字段说明

Agent 最终会返回一个包含 `events` 列表的 JSON 报告，主要字段包括：

- `app_name`: 操作涉及的应用名称。
- `behavior_category`: 行为分类（如：直接外发、潜在隐藏行为、正常操作）。
- `description`: 具体的动作描述。
- `original_filename`: 变更前的文件名。
- `modified_filename`: 变更后的文件名。


