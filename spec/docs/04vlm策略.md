有，而且不少。提示词和合帧只是 VLM 模块最显眼的两块，真正要让它可评测、可复盘、可控成本，还要补一圈工程能力。

我会把 VLM 模块拆成这些事：

**1. 候选帧选择**
现在 `choose_vlm_frames(...)` 主要靠 OCR 可疑词和低置信度选帧。还可以继续做：

- anchor 帧优先。
- 日志窗口 strong 优先。
- OCR 空但日志强提示的帧也能进 VLM。
- 相邻重复帧不要都送。
- 同一个窗口至少保留 1 张，避免只看某个窗口。
- 支持 `vlm_all_candidate_frames` 做纯 VLM 评估。
- 支持 `vlm_grid_size=2` 做合帧评估。

**2. VLM 输入构造**
除了 prompt 和合帧，还要组织上下文：

- 每张图/子图的 `frame_id`、`timestamp_ms`、`reason`、`window_id`。
- 对应 OCR 文本和 confidence。
- 相关日志 anchor：`file_selected`、`app_switch`、`upload`。
- `active_apps`，比如日志窗口附近出现过哪些前台应用。
- 敏感源文件名、basename、stem。
- 输出 schema 的严格说明。

最好导出：

```text
vlm_request.json
keyframes_vlm_grid/
```

这样你能复盘“模型到底看了什么”。

**3. 输出 schema 和解析**
现在 parser 可以吃简单 JSON，但还不够硬。需要：

- 强制模型输出 `events[]`。
- 每个 event 必须有 `evidence_frame_ids`。
- 支持 `frame_id` / `timestamp_ms` / `time_range` 三种定位。
- 支持模型字段别名，比如 `file_name`、`original_file`、`resource`。
- 对无效 JSON 做修复或记录错误。
- 对缺字段事件降级，而不是整批失败。
- 输出 `parse_errors` 和 `dropped_events`。

理想 schema 类似：

```json
{
  "events": [
    {
      "evidence_frame_ids": ["frame_0_0"],
      "app_name": "Cherry Studio",
      "behavior_category": "direct_leak",
      "operation_type": "ai_chat_upload",
      "original_filename": "公司合作合同.docx",
      "modified_filename": "",
      "sink_type": "ai_chat",
      "description": "...",
      "confidence": 0.86
    }
  ]
}
```

**4. 证据回指**
这个非常关键。VLM 不能只说“我觉得泄露了”，它必须能回指证据：

```text
VLM event -> frame_id -> image_path -> OCR text -> log window -> correlated event -> leak path
```

否则后面你评估准确性时，会发现没法回答：

- 它看的是哪张图？
- 它为什么认为是外发？
- 它是不是把桌面图标误当成前台 app？
- 它有没有引用错误文件？

**5. 原始响应落盘**
现在成功响应没有完整落盘，不利于调试。应该导出：

```text
vlm_request.json
vlm_response.json
vlm_parse_result.json
```

其中包括：

- provider/model/base_url。
- 发送了哪些图片。
- 每张图对应哪些原始 frame。
- 原始文本响应。
- 解析出的事件。
- 被丢弃事件和原因。
- 错误信息。

**6. 成本控制**
VLM 贵，所以模块要有预算机制：

- `max_vlm_frames`。
- `vlm_grid_size`。
- `vlm_max_image_side`。
- `vlm_jpeg_quality`。
- 每个窗口最多几张 VLM 图。
- 每个 case 最多几次请求。
- 可选 dry-run，只导出 request，不调用 API。
- 记录估算图像数量/像素/请求数。

**7. 错误处理**
要处理这些情况：

- 没有 API key。
- HTTP 429/超时/模型错误。
- 返回非 JSON。
- 模型返回空 events。
- 某张图读取失败。
- base64 太大。
- provider 字段格式不兼容。

现在 `vlm_failed` 会进 `errors`，但还可以更细。

**8. Provider 抽象**
现在是 OpenAI-compatible client，够用但比较薄。后面可以加：

- DashScope/Qwen 的专门配置。
- OpenAI 视觉模型。
- 本地 VLM/Ollama/LM Studio。
- mock provider，用于测试和离线评估。
- replay provider，从旧 `vlm_response.json` 复现解析结果。

**9. 评测模式**
你后面要测 VLM 准确性，这块很重要：

- `--no-non-vlm` 已经有了。
- 还需要 `--vlm-dry-run`：只生成 VLM 输入，不调用。
- `--vlm-replay-response`：用保存的响应复跑 parser。
- 输出 per-event 评测字段：TP/FP/FN 以后才好算。
- groundtruth 对齐时要知道 VLM 事件对应哪个 operation。

**10. 安全边界**
VLM 模块要避免把 groundtruth 当提示词证据。可以给敏感源文件名（注意原来的标注不规范，实际上只有第一个打开的是敏感文件，后面不一定是，可能是派生文件），但不要给“这个样例标注是泄露”。否则评测会污染。

**优先级建议**
如果只按性价比排，我建议这样做：

1. `vlm_request.json` / `vlm_response.json` / `vlm_parse_result.json` 落盘。
2. VLM 输出强制带 `evidence_frame_ids`，parser 支持并保留。
3. `vlm_grid_size=2` 合帧。
4. OCR 空但强日志 anchor 的帧也能进 VLM。
5. `--vlm-dry-run` 和 replay。
6. 更硬的 schema 校验和 parse error 统计。
7. provider mock，用测试锁住 VLM-only 模式。

所以一句话：**VLM 模块不只是“给模型看图”**。它应该负责“选什么图、给什么上下文、怎样省钱、如何约束输出、怎么回指证据、如何复盘评测”。现在代码已经有骨架，下一步要把它从“能调用”升级成“能实验”。


当前 **ocr_all** 和 **ocr_triage** 两种策略都会先跑 OCR，然后把 OCR 内容放进 VLM 请求里：

* **vlm.py** 的 **VlmRequestFrame** 里有 **ocr_text**、**ocr_confidence**
* prompt 里每帧会写入：
  frame_id、**timestamp_ms**、**window_id**、**reason**、**selection_reason**、**ocr_confidence**、**ocr_text**
* artifact 里的 **vlm_request.json** 也会保存每帧的 **ocr_text** 和 **ocr_confidence**

只有 **--vlm-frame-strategy direct_keyframes** 不经过 OCR，所以传给 VLM 的 **ocr_text=""**、**ocr_confidence=0.0**，相当于纯看图。
