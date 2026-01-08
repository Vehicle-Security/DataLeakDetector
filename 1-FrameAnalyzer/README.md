# 风险操作分析模块

风险操作分析模块。从海量录屏帧中精准定位并分析特定行为的功能。实现了2个Tool API给Agent调用。
Tool-1: 检索前后相关帧，并描述，分析其行为。
Tool-2: 将某些帧分析得更细致一些。利用 MLLM+OCR，分析某一帧信息+行为。


## 🚀 核心功能

1.  **检索与keyword相关的帧并描述行为 (`relavance_frame.py`)**：keyword可能是文件名、应用名、一大段文字等。
    
    -   **三阶段过滤策略**：画面特征去重（ResNet50）-> 关键词文本过滤（OCR）-> 行为意图分析（Qwen2.5-VL）。
        
    -   **高效处理**：通过采样和特征比对大幅降低 VLM 的调用成本。
        
    -   **结构化报告**：自动生成包含时间轴和事件描述的 JSON 报告。
        
2.  **精细化场景溯源 (`analyze_scene_deep_dive`)**
    
    -   **深度推理**：针对特定关键帧序列进行文分析。
        
    -   **语义增强**：在发送给大模型前，先通过 OCR 提取环境文本作为辅助信息。
        
    -   **高精度意图识别**：利用 Qwen2.5-VL 的强大推理能力，还原操作细节。


## 🛠️ 环境准备

### 1. 依赖安装

确保您的环境已安装 Python 3.9+ 及 CUDA 驱动（推荐）。

Bash

```
pip install opencv-python torch torchvision pillow tqdm easyocr langchain-openai langchain-core

```

### 2. 模型与 API 配置

-   **Local Models**: 脚本会自动下载 ResNet50 权重。
    
-   **VLM API**: 本工具使用通义千问 `qwen2.5-vl-72b-instruct`。请确保拥有阿里云 DashScope 的 API Key。


## 💻 使用示例

### 场景 A：从视频中检索与文件名‘项目2需求分析’有关的帧并描述

Python

```
analysis_results = analyze_video_behavior(
        rec_start_time_str='2025-12-28 18:41:28',    
        search_start_time_str='2025-12-28 18:41:53', 
        search_end_time_str='2025-12-28 18:42:10',   
        target_keywords=['项目2需求分析'],
        video_path="../video/42.mp4"
    )
    events = analysis_results.get("events", [])
    first_event = events[0]
    logger.info(f"第一个事件的应用名称是: {first_event.get('app_name', '未知')}")
    logger.info(f"第一个事件的操作类型是: {first_event.get('operation_type', '未知')}")
    logger.info(f"第一个事件的行为类别是: {first_event.get('behavior_category', '未知')}")
    logger.info(f"第一个事件的变更前文件名是: {first_event.get('original_filename', '未知')}")
    logger.info(f"第一个事件的变更后文件名是: {first_event.get('modified_filename', '未知')}")
    logger.info(f"第一个事件的描述是: {first_event.get('description', '无')}")


```

### 场景 B：对某些帧进行精细描述

Python

```
test_data = [
        {'frame': './output/frame_002773.jpg', 'frame_index': 2773, 'timestamp': 46.2},
        {'frame': './output/frame_002832.jpg', 'frame_index': 2832, 'timestamp': 47.2},
        {'frame': './output/frame_002891.jpg', 'frame_index': 2891, 'timestamp': 48.2},
        {'frame': './output/frame_003422.jpg', 'frame_index': 3422, 'timestamp': 57.0},
    ]
    
final_report = analyze_scene_deep_dive(
    filtered_frames=test_data,
    output_path="./output/security_deep_dive.json"
)
```


## 📊 输出结果说明

系统将生成一个 JSON 格式的报告：

### 1. 行为检索结果 (Scenario 1: `relavance_frame.py` Output)

该结果由第一个函数生成，主要用于在长视频中快速检索与文件名或应用名称或文本有关的事件片段，并进行操作描述。

JSON

```
{
    "search_range": {
        "start": "2025-12-28 18:41:53",
        "end": "2025-12-28 18:42:10"
    },
    "total_events": 1,
    "events": [
        {
            "time_range": "2025-12-28 18:41:54 - 2025-12-28 18:42:16",
            "involved_timestamps": [
                "2025-12-28 18:41:54",
                "2025-12-28 18:41:58",
                "2025-12-28 18:42:01",
                "2025-12-28 18:42:06",
                "2025-12-28 18:42:11",
                "2025-12-28 18:42:16"
            ],
            "app_name": "iLovePDF",
            "behavior_category": "潜在隐藏行为",
            "operation_type": "格式转换",
            "original_filename": "项目2需求分析.docx",
            "modified_filename": "项目2需求分析.pdf",
            "description": "用户在 iLovePDF 网站上将 '项目2需求分析.docx' 文件转换为 PDF 格式。从选择文件到完成转换的整个过程被记录下来。"
        }
    ]
}

```

----------

### 2. 深度场景分析 (Scenario 2: `analyze_scene_deep_dive` Output)
对于某些帧再次精细化分析。

JSON

```
{
    "overall_summary": "用户在QQ聊天窗口中发送了一份名为'AAA公司员工守则.docx'的文件。",
    "environment": "QQ即时通讯软件，具体页面为与联系人'青山撞入怀'的聊天窗口",
    "action_chain": [
        {
            "timestamp": 46.2,
            "description": "用户在QQ聊天窗口中选择并上传了名为'AAA公司员工守则.docx'的文件。"
        },
        {
            "timestamp": 47.2,
            "description": "用户确认发送文件，点击了发送按钮。"
        },
        {
            "timestamp": 48.2,
            "description": "文件成功发送至联系人'青山撞入怀'。"
        },
        {
            "timestamp": 57.0,
            "description": "用户在聊天窗口中输入了一些文本内容。"
        }
    ],
    "risk_assessment": {
        "level": "高",
        "reasoning": "1. 关键OCR文字：'AAA公司员工守则.docx'，这可能是一份敏感的内部文件；2. 关键视觉动作：用户将文件拖拽至QQ聊天窗口并点击发送按钮；3. 意图推导逻辑：用户有明确的数据外发行为，将可能包含敏感信息的文件发送给外部联系人。",
        "hit_criteria": [
            "存在数据外发行为"
        ]
    },
    "final_intent": "用户意图将一份可能包含敏感信息的公司内部文件通过QQ发送给外部联系人'青山撞入怀'。"
}

```

----------

### 💡 字段说明

-   **time_range**: 事件在原始视频中发生的时间区间。
    
-   **action_chain**: 深度溯源模式下，模型对每一帧画面所代表的具体原子动作的推导。
    
-   **risk_assessment**: 结合 OCR 文本和视觉动作，由大模型生成的风险判定标准，可直接用于审计告警。

