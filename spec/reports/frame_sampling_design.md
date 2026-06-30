# VLM 抽帧与实时评测日志设计

本文记录 `tools/benchmark_nas_samples.py` 在重构后的 live VLM 评测路径中如何抽帧、如何记录每个样本的中间结果，以及这些设计为什么更适合 DataLeakDetector 的泄露检测任务。

## 背景

全量 NAS live VLM 评测的耗时主要来自两件事：

1. 需要对大量 triage-only case 调用 VLM；
2. 每次 VLM 请求都要上传多张截图，并等待多模态模型判断“是否已经完成泄露”。

因此抽帧目标不是“完整理解整段视频”，而是在日志已经给出可疑时间窗的前提下，把最有可能包含完成态证据的少量画面交给 VLM。完成态证据包括发送成功、上传完成、远端列表出现文件、屏幕共享中暴露敏感内容、VM/远程桌面复制完成等。

## 当前评测流水线

![总体架构](figures/system_architecture.svg)

NAS benchmark 中每个 case 按如下顺序处理：

1. `LogFirstDetector` 先基于日志做 deterministic 判断。
2. 如果日志已经能连出敏感文件外发链路，则直接进入 final positive。
3. 如果日志不能直接确认，但出现 AI、上传、剪贴板、会议、VM、远程桌面等可疑上下文，则进入 VLM fallback。
4. 可选的 feature-based VLM gate 会先判断是否已有足够强的本地特征可以直接给出本地 positive verdict。
5. 仍然不确定的 case 才根据 review window 抽取代表帧，调用 live VLM。
6. VLM verdict 或本地 gate verdict 进入 EventCorrelator，转成 upload candidate，再形成 final bucket。

![VLM 触发门控与最终判定](figures/vlm_gate_flow.svg)

## 抽帧策略

重构前的 live VLM 抽帧采用简单均匀采样：在 review window 内按时间平均取 `max_vlm_frames` 个时间点。这个方法实现简单，但有两个问题：

- 可能错过短暂的完成态画面，例如“发送成功”提示、上传完成后的远端列表刷新；
- 对长窗口不够敏感，长视频中几帧均匀点可能都落在静止页面或无关等待阶段。

重构后的抽帧策略改为“候选帧 + 轻量场景变化评分 + 上下文保底”：

![代表帧抽样策略](figures/frame_sampling_strategy.svg)

步骤如下：

1. 从 fallback meta 得到 VLM review windows。
2. 为每个窗口分配候选帧预算，默认候选数量为 `max(24, max_vlm_frames * 6)`。
3. 只 seek/decode 候选时间点，不全量解码视频。
4. 将候选帧缩放为 `96x54` 灰度缩略图。
5. 计算当前缩略图与上一候选缩略图的平均绝对差，归一化为 `scene_score`。
6. 选择代表帧：
   - `window_start`：保留窗口开头上下文；
   - `window_mid`：保留中段状态；
   - `window_end`：保留完成态或取消态；
   - `scene_change`：补充画面变化最大的候选帧。
7. 对最终入选帧做 resize、JPEG 编码和 base64 封装。
8. VLM prompt 中附带每帧的 `source_frame`、`selection_reason` 和 `scene_score`。

这种策略参考了视频 shot boundary / scene detection 的常见思想：先找画面变化，再从变化附近取代表帧。TransNetV2 使用深度网络做 shot transition detection；PySceneDetect 和 FFmpeg scene detection 也都围绕画面内容变化来定位镜头/场景边界。我们的实现不引入额外深度模型或 FFmpeg 依赖，而是用 OpenCV 缩略图差异作为轻量近似，适合 benchmark 中“日志已经限定可疑窗口”的场景。

## VLM 请求门控

`--vlm-gate-mode` 用于减少不必要的远端 VLM 请求。这个门控只基于运行时可观测特征，不基于样本名、目录名、groundtruth、case 顺序或固定数量上限。

会使用的特征包括：

- 日志事件类型，例如 `file_send`、`data_upload`、`file_share`、`screen_share_start`、`screenshot_capture`、`clipboard_image`。
- 日志文本中的应用、窗口、文件名和内容摘要。
- `LogFirstDetector` 产生的 `operation_records`。
- 上下文类别，例如截图、导出、VM、远程桌面、Git、压缩/转换。
- 抽帧后本地 OCR 的 `ocr_flags`，例如 `completion_keyword` 和 `sensitive_name_visible`。

门控模式：

| 模式 | 含义 |
| --- | --- |
| `all` | 保持旧行为：所有 triage-only case 都进入远端 VLM 队列。 |
| `strict` | 只对强本地特征做本地 positive，例如显式发送/上传/共享事件、截图上下文、导出上下文。 |
| `adaptive` | 在 `strict` 基础上加入较高置信的 VM 场景，但仍排除邮件/云盘等容易误判的外部上下文。推荐用于半小时预算评测。 |
| `aggressive` | 进一步允许 Git、压缩/转换等带完成态文本的场景本地 positive。速度更快，但需要单独验证误报风险。 |

当 gate 命中时，case 不会发送给 qwen，而是得到 `live_vlm_verdict.status=local_positive`。如果已经完成抽帧/OCR，并且本地 OCR 同时看到完成态关键词和敏感文件名，则得到 `live_vlm_verdict.status=local_ocr_positive`。

## 关键参数

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| `--vlm-gate-mode` | `all` | 远端 VLM 前的特征门控模式：`all`、`strict`、`adaptive`、`aggressive` |
| `--vlm-workers` | `DLD_VLM_WORKERS` 或 `1` | 并发 live VLM worker 数 |
| `--max-vlm-frames` | `6` | 每个 live VLM case 最多发送多少张图 |
| `DLD_VLM_REVIEW_CACHE` | `1` | 是否启用 live VLM verdict 磁盘缓存；设为 `0` 可关闭 |
| `DLD_VLM_REVIEW_CACHE_DIR` | `spec/output/cache/vlm_reviews` | live VLM verdict 缓存目录 |
| `DLD_VLM_LOCAL_OCR_GATE` | `1` | 是否启用抽帧后本地 OCR positive 短路；设为 `0` 可关闭 |
| `DLD_VLM_REVIEW_CANDIDATE_FRAMES` | `max(24, max_vlm_frames * 6)` | 每个 case 最多扫描多少个候选时间点 |
| `DLD_VLM_REVIEW_BASE_FRAMES` | `6` | 自适应帧预算的基础帧数 |
| `DLD_VLM_REVIEW_MIN_FRAMES` | `min(4, cap)` | 自适应帧预算的最小帧数 |
| `DLD_VLM_REVIEW_MAX_IMAGE_FRAMES` | OCR 开启时 `min(max_frames, 5)` | 每个 case 最多发送多少张真实图片给 VLM |
| `DLD_VLM_REVIEW_MIN_IMAGE_FRAMES` | `min(4, max_image_frames)` | 每个 case 至少保留多少张真实图片 |
| `DLD_VLM_REVIEW_MAX_OCR_FRAMES` | `min(selected_frames, 4)` | 每个 case 最多对多少张入选帧跑本地 OCR |
| `DLD_VLM_REVIEW_MIN_FRAME_GAP` | `12` | 优先挑选 scene-change 帧时的最小帧号间隔 |
| `DLD_VLM_REVIEW_IMAGE_SCENE_THRESHOLD` | `0.08` | 判定候选帧有明显场景变化的阈值 |
| `DLD_VLM_REVIEW_IMAGE_MAX_EDGE` | `960` | 发送给 VLM 的图片最长边 |
| `DLD_VLM_REVIEW_JPEG_QUALITY` | `65` | JPEG 质量 |

如果 VLM 调用很慢，可以先降低帧数：

```powershell
python tools/benchmark_nas_samples.py --use-vlm --max-vlm-cases 5 --max-vlm-frames 3
```

如果担心漏掉短暂 UI 状态，可以增加候选帧数：

```powershell
$env:DLD_VLM_REVIEW_CANDIDATE_FRAMES = "48"
```

半小时预算下推荐使用 `adaptive` gate，并限制 OCR/图片 payload：

```powershell
$ts = Get-Date -Format "yyyyMMdd_HHmmss"
$log = "spec\output\nas_vlm_adaptive_$ts.log"
$json = "spec\output\nas_vlm_adaptive_$ts.json"

$env:DLD_VLM_REVIEW_MAX_OCR_FRAMES = "2"
$env:DLD_VLM_REVIEW_MAX_IMAGE_FRAMES = "4"
$env:DLD_VLM_REVIEW_CACHE = "1"
$env:DLD_VLM_LOCAL_OCR_GATE = "1"

python tools\benchmark_nas_samples.py `
  --use-vlm `
  --vlm-gate-mode adaptive `
  --vlm-workers 4 `
  --max-vlm-frames 10 `
  --json-output $json `
  2>&1 | Tee-Object -FilePath $log
```

## 实时日志

重构后，benchmark 会把每个 case 的中间状态打印到 `stderr`。原来的 PowerShell `2>&1 | Tee-Object` 命令会把这些日志写入 `.log` 文件。

示例：

```text
[VLM 12/all QUEUED] case=stage1\0-normal-meeting-feishu-1 progress=35/211 expected=0 gate=adaptive reasons=ambiguous_exfil_context_near_sensitive_log frames=6/10 complexity=external_transfer_context
[VLM LOCAL] case=stage1\3-Messaging-QQ-1 progress=72/211 gate=strict reason=explicit_transfer_event
[CASE 35/211] FP case=stage1\0-normal-meeting-feishu-1 expected=0 final=1 det=0 triage=1 vlm=success frames=6 confidence=0.72 action=screen_share reasons=ambiguous_exfil_context_near_sensitive_log
```

字段解释：

| 字段 | 含义 |
| --- | --- |
| `CASE x/y` | 当前 case 进度 |
| `TP/FP/TN/FN` | 当前 case 的 final bucket |
| `expected` | groundtruth 是否为正例 |
| `final` | 最终判定是否为 positive |
| `det` | deterministic 阶段是否命中 |
| `triage` | triage 阶段是否进入 positive/复核 |
| `vlm` | VLM/本地 gate 状态，如 `success`、`failed`、`triage_only`、`local_positive`、`local_ocr_positive` |
| `frames` | 实际发送给 VLM 的帧数 |
| `confidence` | VLM verdict 置信度 |
| `action` | VLM 判定的完成动作 |
| `reasons` | fallback gate 触发原因 |
| `remote_vlm_requests` | summary 中实际远端 VLM/qwen 请求数 |
| `local_vlm_resolutions` | summary 中由本地 feature/OCR gate 解析的 case 数 |
| `vlm_cache_hits` | summary 中复用历史 live VLM verdict 缓存的 case 数 |

## JSON 审计字段

当 live VLM 成功返回时，case 的 `live_vlm_verdict` 中会包含：

```json
{
  "frames_sent": 6,
  "frame_selection": [
    {
      "index": 1,
      "timestamp": "2026-06-03 00:41:51",
      "frame_index": 12,
      "scene_score": 1.0,
      "selection_reason": "window_start"
    }
  ]
}
```

这让后续排查误报/漏报时可以直接定位：VLM 看到的是哪几帧，为什么选中这些帧，是否错过了关键完成态。

启用 `--vlm-gate-mode strict/adaptive/aggressive` 后，每个 case 还会包含 `vlm_gate`：

```json
{
  "vlm_gate": {
    "mode": "adaptive",
    "action": "local_positive",
    "reason": "explicit_transfer_event",
    "features": {
      "explicit_transfer_event": true,
      "screenshot_context": false,
      "export_context": false
    }
  },
  "live_vlm_verdict": {
    "status": "local_positive",
    "model": "local_vlm_gate",
    "frames_sent": 0
  }
}
```

`summary.vlm_reviews` 表示进入 VLM fallback 的总数；`summary.vlm_remote_requests` 表示实际请求 qwen 的数量；`summary.vlm_local_resolutions` 表示被本地 gate 或 OCR 短路解析的数量。

live VLM verdict 会按 case、视频文件指纹、review window、敏感文件、帧/OCR 参数、模型名和 prompt/cache 版本写入 `DLD_VLM_REVIEW_CACHE_DIR`。重跑同一配置时，命中缓存的 case 会直接返回已有 verdict，不再抽帧或请求 qwen；对应 verdict 会带有 `cache_hit=true` 和 `cache_path`。`summary.vlm_cache_hits` 记录缓存命中数量，且缓存命中不会计入 `summary.vlm_remote_requests`。

## 证据流

![轻量证据图模型](figures/evidence_graph.svg)

重构后的证据流强调两点：

1. 日志规则和 VLM 不是相互替代，而是分工：日志负责缩小时间窗和建立文件/应用线索，VLM 负责确认视觉完成态。
2. 抽帧结果进入 JSON 审计链，避免 live VLM 变成不可解释的黑箱判断。

## 参考资料

- TransNetV2: An effective deep network architecture for fast shot transition detection: https://arxiv.org/abs/2008.04838
- PySceneDetect detector documentation: https://www.scenedetect.com/docs/latest/api/detectors.html
- PySceneDetect CLI documentation: https://www.scenedetect.com/docs/latest/cli.html
- FFmpeg filters documentation, scene/change filters: https://ffmpeg.org/ffmpeg-filters.html
