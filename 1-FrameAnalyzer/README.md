# 🛡️ 帧检索与分析模块

本模块旨在从海量录屏数据中，利用多模态大模型（MLLM）与计算机视觉技术，精准定位并审计特定违规行为。模块为 Agent 提供了 4 个核心 Tool API，覆盖了从关键词检索到深度场景分析的全流程。

## 🚀 核心功能

### 1. 目标文件名检索 (`Tool-1`)

* ​**功能**​：检索与特定文件名、应用名相关的视频片段。
* ​**策略**​：ResNet50 画面去重 -> OCR 关键词匹配 -> Qwen2.5-VL 意图识别。

### 2. 大段文字/复制粘贴检索 (`Tool-2`)

* ​**功能**​：检索视频中出现的大段特定文本（如敏感代码、核心内刊）的复制、粘贴或阅读行为。
* ​**技术点**​：利用 `OCR` + `RapidFuzz` 进行模糊文本相似度匹配，即使 OCR 有微小识别错误也能精准定位。

### 3. 疑似黑名单应用/AI 套壳检测 (`Tool-3`)

* ​**功能**​：识别未授权的 AI 套壳工具（如 Chatbox, LobeChat）或浏览器插件（Monica, Sider）。
* ​**特征工程**​：基于 UI 布局特征（聊天气泡、模型下拉框、API 配置界面）而非单纯依赖进程名。

### 4. 精细化场景溯源 (`Tool-4: vlm_analysis`)

* ​**功能**​： 对锁定的关键帧进行多帧关联推理，生成更精细化的描述。

---

## 🛠️ 环境准备

### 依赖安装

Bash

```
pip install opencv-python torch torchvision pillow tqdm easyocr langchain-openai langchain-core rapidfuzz numpy
```

### API 配置

在项目根目录下创建 `.env` 文件，用于安全存储阿里云 DashScope 密钥：

Bash

```
echo "DASHSCOPE_API_KEY=你的_API_KEY_在此" > .env
```

---

## 🏗️ 项目架构

本项目采用**面向对象 (OOP)** 的分层设计，以实现最高程度的代码复用：

| **文件名**            | **职能描述**                                                                                 |
| ----------------------------- | ---------------------------------------------------------------------------------------------------- |
| `core_analyzer.py`      | ​**基类**​：负责加载 ResNet、初始化 LLM、OCR 延迟加载、Base64 编码及统一 JSON 保存逻辑。 |
| `unified_retriever.py`  | ​**内容检索**​：处理关键词及长文本匹配逻辑。支持根据匹配类型自动控制是否延伸上下文帧。          |
| `blacklist_analyzer.py` | ​**黑名单扫描**​：针对长时间视频进行均匀采样与分批分析，专注于违规应用识别。                    |
| `main.py`               | ​**入口脚本**​：统一调度不同场景的分析任务。                                                    |
| `prompts.py`            | ​**提示词库**​：存储不同业务场景下的多模态指令。                                                |

## 💻 使用示例

可直接运行`main.py`文件，每个场景的结果都会保存为json文件

### 场景 A：从视频中检索与文件名有关的帧并描述 (Tool-1)

Python

```
from unified_retriever import ContentRetriever
retriever = ContentRetriever()
kw_results = retriever.analyze(
        rec_start_str='2025-12-28 18:55:36',
        s_start_str='2025-12-28 18:55:46',
        s_end_str='2025-12-28 18:56:05',
        target_keywords=['项目1详细规划'],
        video_path="../video/43.mp4"
    )
```

### 场景 B：检索大段敏感文字的传播路径 (Tool-2)

适用于审计：员工是否将敏感代码片段粘贴到了外部编辑器或聊天工具中。

Python

```
from unified_retriever import ContentRetriever
retriever = ContentRetriever()
analysis_results = retriever.analyze(
        rec_start_time_str='2025-12-28 10:28:00',    
        search_start_time_str='2025-12-28 10:28:00', 
        search_end_time_str='2025-12-28 10:28:40',   
        target_text='''我总结了当前四类主流防护工具的局限性：
        EPR/IPS：只能记录系统调用，粒度过粗
        DLP：依赖内容匹配，能识别的场景少。
        UAM/UEBA：侧重行为元数据分析，但无法理解用户操作的具体内容和意图。
        总的来说，现有方法对非结构化视觉内容（如图片、视频）和跨应用场景的覆盖严重不足。

        因此，本研究提出三个核心研究问题：
        如何从视频流中高效捕捉关键操作事件？
        如何识别用户是否在执行敏感操作？
        如何追踪隐私数据的传播路径？''',
        video_path="../video/paste.mp4"
    )
```

### 场景 C：检测疑似黑名单应用 (Tool-3)

Python

```
from blacklist_analyzer import BlacklistAnalyzer
bl_analyzer = BlacklistAnalyzer()
results =bl_analyzer.analyze_blacklist(
        rec_start_time_str='2026-01-05 10:00:00',
        search_start_time_str='2026-01-05 10:00:00',
        search_end_time_str='2026-01-05 10:01:20',
        video_path="../video/wrapped_app.mp4",
        batch_size=6
    )
```

### 场景 D：对某些帧进行精细描述 (Tool-4)

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

---

## 📊 输出结果说明

### 文件相关帧行为检索结果 (Tool-1 Output)

JSON

```
{
  "root": {
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
        "app_name": "ilovePDF",
        "behavior_category": "潜在隐藏行为",
        "operation_type": "格式转换",
        "original_filename": "项目2需求分析.docx",
        "modified_filename": "项目2需求分析.pdf",
        "description": "用户在 ilovePDF 网站上将 '项目2需求分析.docx' 文件转换为 PDF 格式。从选择文件到完成转换的整个过程被记录下来。"
      }
    ]
  }
}
```

### 复制粘贴行为分析报告 (Tool-2 Output)

JSON

```
{
{
  "root": {
    "search_range": {
      "start": "2025-12-28 10:28:00",
      "end": "2025-12-28 10:28:40"
    },
    "total_events": 2,
    "events": [
      {
        "event_id": 1,
        "behavior_category": "文本复制",
        "app_name": "记事本",
        "source_app": "未知",
        "start_time": "2025-12-28 10:28:19",
        "end_time": "2025-12-28 10:28:20",
        "original_filename": "未知",
        "modified_filename": "未知",
        "description": "用户在记事本中选中并复制了目标文本。",
        "confidence": 0.95
      },
      {
        "event_id": 2,
        "behavior_category": "文本粘贴",
        "app_name": "QQ",
        "source_app": "记事本",
        "start_time": "2025-12-28 10:28:23",
        "end_time": "2025-12-28 10:28:28",
        "original_filename": "未知",
        "modified_filename": "未知",
        "description": "用户将复制的文本粘贴到了QQ聊天窗口中。",
        "confidence": 0.95
      }
    ]
  }
}
```

---

### 黑名单检测报告 (Tool-3 Output)

JSON

```
{
  "root": {
    "search_range": [
      "2026-01-05 10:00:00",
      "2026-01-05 10:01:20"
    ],
    "events": [
      {
        "event_id": 1,
        "app_name": "Cherry Studio",
        "behavior_category": "第三方AI套壳应用",
        "visual_evidence": "在图片中看到了Cherry Studio的图标和界面，这是一个已知的独立客户端。此外，界面显示了与AI助手的对话框，符合典型的Chatbox UI布局。",
        "detected_content": "用户正在与AI助手进行对话，内容包括问候和一些基本交流。",
        "start_time": "2026-01-05 10:00:00",
        "end_time": "2026-01-05 10:00:18",
        "confidence": 0.95
      }
    ]
  }
}
```

### 深度场景分析 (Tool-4 Output)

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

---
