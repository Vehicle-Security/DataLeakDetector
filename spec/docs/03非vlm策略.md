# 03 非 VLM 策略：不用 AI 怎么确定性推理泄露

本文解释一个核心问题：**不借助 VLM/LLM，系统怎么把杂乱日志、OCR 文本、文件路径和窗口信息合成“是否泄露”的结论**。

关键点不是让代码“理解自然语言”，而是把所有材料压成几类可计算事实，再用确定性规则做 join 和 taint 传播。每一步都应该能被单元测试复现：给定输入字段，输出固定事实；没有“模型觉得像”的黑箱判断。

![非 VLM 确定性推理流程](non_vlm_deterministic_pipeline.svg)

整条链路是：

```text
原始日志 + OCR 文本
  -> 归一化
  -> 字段/文件名/动作抽取
  -> 时间邻近 join
  -> 同文件 join
  -> sink / 外发动作分类
  -> CorrelatedEvent
  -> UploadCandidate
  -> DatalogFact
  -> taint 传播
  -> LeakPath / final conclusion
```

下面每一步都按“输入是什么、机械规则怎么做、输出是什么”写清楚。

## 0. 原则

非 VLM 策略只做确定性判断：

- 可以用字符串匹配、正则、时间窗口、文件路径归一化、进程名白名单/黑名单、Datalog 规则。
- 不让模型解释画面语义。
- 不把 groundtruth 当证据；groundtruth 只用于最后评估。
- 不直接让 OCR 文本决定“泄露了”，OCR 只是提供可 join 的文件名、按钮词、页面词、状态词。

一句话：

```text
不理解全文，只匹配事实。
```

## 1. 日志归一化

### 输入

原始日志可能长这样：

```json
{
  "timestamp": "2026-03-10T00:15:56.517",
  "event_type": "file_selected",
  "file_path": "C:\\Users\\46521\\Desktop\\数据采集测试\\公司合作合同.docx",
  "process_info": {
    "process_name": "Cherry Studio"
  },
  "window_info": {
    "window_title": "打开"
  }
}
```

### 机械规则

归一化不做推理，只整理字段：

1. `timestamp` 转成 `timestamp_ms`。
2. 用 `recording_start_ms` 或 `extra.relative_timestamp` 得到 `video_time_ms`。
3. 路径统一成 `/`。
4. 从嵌套字段里取 `process_name`、`app_name`、`window_title`。
5. 把 `description`、`extra`、`upload_detection` 等文本拼成 searchable text。

伪代码：

```python
event = LogEvent(
    event_id="log_123",
    timestamp_ms=parse_timestamp_ms(raw["timestamp"]),
    video_time_ms=timestamp_ms - recording_start_ms,
    event_type=raw["event_type"].lower(),
    file_path=normalize_path(raw["file_path"]),
    process_name=raw["process_info"]["process_name"],
    window_title=raw["window_info"]["window_title"],
    raw=raw,
)
```

### 输出

```text
LogEvent(
  event_id="log_123",
  video_time_ms=19517,
  event_type="file_selected",
  file_path="C:/Users/46521/Desktop/数据采集测试/公司合作合同.docx",
  process_name="Cherry Studio",
  window_title="打开"
)
```

这个阶段的要点是：**后续所有 join 都靠 `video_time_ms` 和标准化路径**。

## 2. OCR 结果归一化

### 输入

`ocr_results.json` 里的一条结果：

```json
{
  "frame_id": "frame_0_0",
  "timestamp_ms": 19517,
  "reason": "strong:anchor",
  "text": "... 默认助手 gpt-3.5-turbo ... 公司合作合同.docx ...",
  "confidence": 0.92,
  "provider": "paddleocr_gpu"
}
```

### 机械规则

OCR 文本不直接推理，只抽取事实：

1. 用正则抽文件名。
2. 用词表抽动作词。
3. 用词表抽 sink/app 线索。
4. 原始 OCR 文本保留在 `description`。

文件名正则可以很土：

```python
FILE_RE = re.compile(
    r"[\w\u4e00-\u9fff ._\-()（）]+"
    r"\.(docx|doc|pdf|txt|png|jpg|jpeg|xlsx|xls|sql|zip|7z|rar)",
    re.IGNORECASE,
)

files = [match.group(0).strip() for match in FILE_RE.finditer(ocr_text)]
```

动作词：

```python
SEND_WORDS = {"发送", "上传", "附件", "分享", "传输", "文件已成功传输", "正在发送"}
AI_CHAT_WORDS = {"gpt", "deepseek", "cherry studio", "默认助手", "聊天"}
```

### 输出

```text
OcrFact(
  frame_id="frame_0_0",
  timestamp_ms=19517,
  mentioned_files=["公司合作合同.docx"],
  action_words=["gpt", "默认助手"],
  text="... 默认助手 gpt-3.5-turbo ... 公司合作合同.docx ..."
)
```

如果不想引入新模型，也可以把这些字段折进 `FrameObservation`：

```text
FrameObservation(
  observation_id="ocr_189",
  start_ms=19517,
  operation_type="visual_text_observed",
  description="OCR text: ... 公司合作合同.docx ..."
)
```

但更推荐显式保留 `mentioned_files`，后面 join 会清楚很多。

## 3. 敏感源识别

### 输入

检测运行时只使用 `spec/config/sensitive_files..json` 中维护的原始敏感源：

```text
C:/Users/46521/Desktop/数据采集测试/公司合作合同.docx
```

日志用于核实和补充这个配置：当日志能明确证明某个完整路径是原始敏感源时，
将它人工写入 JSON 后再参与检测。groundtruth 只核对行为，不提供敏感源。
复制、重命名、压缩、转换、截图及内容摘录等派生文件只进入谱系，不写入该 JSON。

### 机械规则

把完整路径、basename、stem 都建索引：

```python
SensitiveIndex:
  full_path_lower:
    "c:/users/46521/desktop/数据采集测试/公司合作合同.docx"
  basename_lower:
    "公司合作合同.docx"
  stem_lower:
    "公司合作合同"
```

匹配函数：

```python
def resolve_sensitive(path_or_text: str, sensitive_files: list[str]) -> str:
    normalized = normalize_path(path_or_text).lower()
    for sensitive in sensitive_files:
        s = normalize_path(sensitive).lower()
        name = basename(s)
        stem = Path(name).stem
        if normalized == s:
            return sensitive
        if name and name in normalized:
            return sensitive
        if len(stem) >= 4 and stem in normalized:
            return sensitive
    return ""
```

### 输出

```text
resolve_sensitive("公司合作合同.docx") 
=> "C:/Users/46521/Desktop/数据采集测试/公司合作合同.docx"
```

这一步也没有 AI：只是路径、文件名、stem 匹配。

## 4. 日志动作分类

### 输入

```text
event_type="file_selected"
process_name="Cherry Studio"
window_title="打开"
```

### 机械规则

将杂事件归入少量动作类：

```python
OPEN_EVENTS = {"open", "opened", "read", "file_read", "access", "file_access"}
TRANSFER_EVENTS = {"copy", "copied", "rename", "renamed", "save_as", "print_to_pdf"}
SINK_EVENTS = {"file_selected", "file_upload", "upload", "uploaded", "upload_complete", "send_click"}

def classify_log_action(event):
    if event.event_type in OPEN_EVENTS:
        return "open_file"
    if event.event_type in TRANSFER_EVENTS:
        return "derive_or_transfer"
    if event.event_type in SINK_EVENTS:
        return "sink_action"
    if event.process_name.lower() == "fsquirt.exe":
        return "sink_action"
    return "other"
```

### 输出

```text
file_selected -> sink_action
fsquirt.exe   -> sink_action
opened        -> open_file
save_as       -> derive_or_transfer
```

注意：动作分类只回答“这类事件像什么动作”，不回答“是否泄露”。

## 5. sink / 外发通道分类

### 输入

```text
process_name="Cherry Studio"
window_title="默认助手"
ocr_text="... gpt-3.5-turbo ... 公司合作合同.docx ..."
```

### 机械规则

固定映射 + 关键词：

```python
SINK_PROCESS = {
    "fsquirt.exe": "bluetooth",
    "qq.exe": "chat",
    "wechat.exe": "chat",
    "weixin.exe": "chat",
    "tim.exe": "chat",
    "dingtalk.exe": "chat",
    "feishu.exe": "chat",
    "lark.exe": "chat",
}

SINK_TEXT = {
    "gpt": "ai_chat",
    "deepseek": "ai_chat",
    "cherry": "ai_chat",
    "上传": "network_upload",
    "附件": "network_upload",
    "发送": "chat",
    "蓝牙文件传送": "bluetooth",
    "文件已成功传输": "bluetooth",
}

def classify_sink(process_name, window_title, ocr_text):
    proc = process_name.lower()
    if proc in SINK_PROCESS:
        return SINK_PROCESS[proc]
    text = f"{window_title} {ocr_text}".lower()
    for token, sink_type in SINK_TEXT.items():
        if token.lower() in text:
            return sink_type
    return ""
```

### 输出

```text
"gpt-3.5-turbo" -> ai_chat
"蓝牙文件传送"  -> bluetooth
"上传文件"      -> network_upload
```

这一步仍然只是词表命中，不做语义理解。

## 6. 时间邻近 join：日志和 OCR 为什么能绑在一起

这是你刚才卡住的关键步骤。

### 输入

日志：

```text
log:
  event_id=log_123
  video_time_ms=19517
  event_type=file_selected
  file_path=C:/Users/46521/Desktop/数据采集测试/公司合作合同.docx
```

OCR：

```text
ocr:
  observation_id=ocr_189
  timestamp_ms=19517
  text="... 公司合作合同.docx ... gpt-3.5-turbo ..."
```

### 机械规则 1：时间差

```python
def nearest_ocr(log, ocr_items, tolerance_ms=15000):
    candidates = [
        ocr for ocr in ocr_items
        if abs(ocr.timestamp_ms - log.video_time_ms) <= tolerance_ms
    ]
    return min(
        candidates,
        key=lambda ocr: abs(ocr.timestamp_ms - log.video_time_ms),
        default=None,
    )
```

这个例子里：

```text
abs(19517 - 19517) = 0 <= 15000
```

所以时间上可 join。

### 机械规则 2：同文件

```python
def file_mentioned_in_ocr(file_path, ocr_text):
    name = basename(file_path).lower()
    stem = Path(name).stem
    text = normalize_path(ocr_text).lower()
    return name in text or (len(stem) >= 4 and stem in text)
```

这个例子里：

```text
basename(log.file_path) = 公司合作合同.docx
ocr.text contains 公司合作合同.docx
=> True
```

所以文件上可 join。

### 机械规则 3：日志动作是外发入口

```python
log.event_type in {"file_selected", "file_upload", "upload", "send_click"}
=> True
```

### 机械规则 4：上下文像外部通道

```python
classify_sink(log.process_name, log.window_title, ocr.text)
=> "ai_chat"
```

### 输出

四个条件都满足：

```text
同一时间
同一文件
外发动作
外发通道
```

于是生成：

```text
CorrelatedEvent(
  original_file="C:/Users/46521/Desktop/数据采集测试/公司合作合同.docx",
  current_file="C:/Users/46521/Desktop/数据采集测试/公司合作合同.docx",
  operation_type="external_sink_interaction",
  behavior_category="data_exfiltration_candidate",
  evidence_refs=["log:log_123", "frame:ocr_189"]
)
```

完整伪代码：

```python
def join_log_ocr(log, ocr_items, sensitive_files):
    original = resolve_sensitive(log.file_path, sensitive_files)
    if not original:
        return None

    if classify_log_action(log) != "sink_action":
        return None

    ocr = nearest_ocr(log, ocr_items, tolerance_ms=15000)
    if not ocr:
        return None

    if not file_mentioned_in_ocr(log.file_path, ocr.text):
        return None

    sink_type = classify_sink(log.process_name, log.window_title, ocr.text)
    if not sink_type:
        return None

    return CorrelatedEvent(
        event_id=new_id(),
        timestamp=log.timestamp,
        event_type=log.event_type,
        app_name=log.app_name or log.process_name,
        original_file=original,
        current_file=log.file_path,
        operation_type="external_sink_interaction",
        behavior_category="data_exfiltration_candidate",
        confidence=max(0.68, ocr.confidence),
        evidence_refs=(f"log:{log.event_id}", f"frame:{ocr.observation_id}"),
    )
```

这就是“不用 AI 怎么合成”的具体答案：**它不是理解出来的，是被四个布尔条件筛出来的**。

## 7. OCR 单独成证据的情况

有时候日志没有清楚写文件名，但 OCR 读到了文件名和外发页面。也可以生成视觉证据，不过置信度应该更保守。

### 输入

```text
ocr_time=19517
ocr_text="... gpt-3.5-turbo ... 公司合作合同.docx ..."
```

### 机械规则

```python
mentioned_sensitive = resolve_sensitive(ocr.text, sensitive_files)
sink_type = classify_sink("", "", ocr.text)

if mentioned_sensitive and sink_type:
    emit visual-only CorrelatedEvent
```

### 输出

```text
CorrelatedEvent(
  event_type="visual_observation",
  original_file="C:/.../公司合作合同.docx",
  current_file="C:/.../公司合作合同.docx",
  operation_type="visual_text_observed",
  behavior_category="data_exfiltration_candidate",
  evidence_refs=["frame:ocr_189"]
)
```

这类事件能提示风险，但最好没有 log+OCR join 那么强。真正严格的判断应优先用 `log:file_selected + OCR:file_name` 这种双证据。

## 8. 文件派生 lineage

泄露不一定直接外发源文件。常见是：

```text
A = 公司战略规划.docx
B = 公司战略规划.pdf
B 被发送
```

### 输入

```text
log1: opened 公司战略规划.docx
log2: print_to_pdf / save_as 公司战略规划.pdf
log3: upload 公司战略规划.pdf
```

### 机械规则

建立派生关系：

```python
if original_file_from_metadata(event.raw):
    lineage.add(target_file, original_file)

if same_process_recent_sensitive and target_stem_similar_source_stem:
    lineage.add(target_file, recent_sensitive_file)
```

stem 相似可以这样做：

```python
source_stem = "公司战略规划"
target_stem = "公司战略规划"
target_stem.startswith(source_stem) => True
```

### 输出

```text
DerivedFrom("公司战略规划.pdf", "公司战略规划.docx")
```

后面解析外发 PDF 时：

```python
root = lineage.root("公司战略规划.pdf")
# 公司战略规划.docx
```

所以外发的是 PDF，泄露源仍然能回到 docx。

## 9. 生成 UploadCandidate

### 输入

```text
CorrelatedEvent(
  original_file="公司合作合同.docx",
  current_file="公司合作合同.docx",
  operation_type="external_sink_interaction",
  behavior_category="data_exfiltration_candidate"
)
```

### 机械规则

```python
if event.behavior_category == "data_exfiltration_candidate":
    sink_type = classify_sink(event.app_name, "", "")
    create UploadCandidate
```

### 输出

```text
UploadCandidate(
  candidate_id="upload_0",
  original_file="公司合作合同.docx",
  current_file="公司合作合同.docx",
  sink_type="ai_chat",
  risk_level="in_progress",
  evidence_refs=["log:log_123", "frame:ocr_189"]
)
```

这个对象仍然不是最终结论，只是“外发候选”。

## 10. 生成 Datalog facts

### 输入

Correlated events、upload candidates、lineage。

### 机械规则

对每个 correlated event：

```text
OpenFile(corr_id:open, app/process, original_file, timestamp)
```

对每个 lineage：

```text
TransferFile(op_id, process, source, target, timestamp)
```

对每个 upload candidate：

```text
LeakFile(upload_id:leak, app/process, current_file, sink_type, timestamp)
```

### 输出

直接外发例：

```text
OpenFile("corr_0:open", "Cherry Studio", "公司合作合同.docx", 19517)
LeakFile("upload_0:leak", "Cherry Studio", "公司合作合同.docx", "ai_chat", 19517)
```

派生外发例：

```text
OpenFile("op1", "Word", "公司战略规划.docx", 5944)
TransferFile("op2", "Word", "公司战略规划.docx", "公司战略规划.pdf", 46485)
CrossProcessTransfer("op3", "Word", "Chrome", "公司战略规划.pdf", 55760)
LeakFile("op4", "Chrome", "公司战略规划.pdf", "browser", 55760)
```

## 11. Taint 传播

### 输入

```text
OpenFile
TransferFile
CrossProcessTransfer
LeakFile
```

### 机械规则

当前 reasoner 的核心规则可以写成：

```text
OpenFile(op, P, F, T)
=> Tainted(P, F)

Tainted(P, A) AND TransferFile(op, P, A, B, T)
=> Tainted(P, B)

Tainted(P1, F) AND CrossProcessTransfer(op, P1, P2, F, T)
=> Tainted(P2, F)

Tainted(P, F) AND LeakFile(op, P, F, Channel, T)
=> LeakPath(F, Channel)
```

### 输出

直接外发：

```text
Tainted("Cherry Studio", "公司合作合同.docx")
LeakFile("Cherry Studio", "公司合作合同.docx", "ai_chat")
=> LeakPath
```

派生外发：

```text
Tainted("Word", "公司战略规划.docx")
TransferFile("Word", docx -> pdf)
=> Tainted("Word", "公司战略规划.pdf")

CrossProcessTransfer("Word" -> "Chrome", pdf)
=> Tainted("Chrome", "公司战略规划.pdf")

LeakFile("Chrome", "公司战略规划.pdf", "browser")
=> LeakPath(original="公司战略规划.docx", leaked="公司战略规划.pdf")
```

## 12. 最终判定

### 输入

`DatalogEngine.query_leak()` 输出：

```text
LeakPath(...)
```

### 机械规则

```python
if leak_paths:
    conclusion = "data_leak_risk_detected"
else:
    conclusion = "no_confirmed_data_leak"
```

### 输出

```text
data_leak_risk_detected
```

注意：如果只有下面任意一条，不应该直接判泄露：

```text
只打开了敏感文件
只打开了聊天软件
OCR 只读到文件名但没有 sink
日志只有系统缓存 modified
```

必须形成至少一条：

```text
敏感文件/派生文件 -> 外发动作 -> 外部通道
```

## 13. 用实际 GPT case 串起来

当前 artifact：

```text
artifacts/neo4j_frame_selection_check/2-ai-gpt-cherystudio-2_logs_402/
```

OCR 里有：

```text
19517ms strong:anchor
"... gpt-3.5-turbo ... 公司合作合同.docx ..."
```

确定性策略会这样处理：

```text
1. OCR 正则抽出 公司合作合同.docx
2. sensitive index 解析到 C:/Users/46521/Desktop/数据采集测试/公司合作合同.docx
3. 时间 19517ms 附近找 file_selected / upload / sink log
4. 如果日志 file_path 同样是 公司合作合同.docx，则 log+OCR join
5. OCR 文本命中 gpt / 默认助手，sink_type=ai_chat
6. 生成 CorrelatedEvent(evidence_refs=["log:...", "frame:ocr_..."])
7. 生成 LeakFile(...)
8. reasoner 看到 tainted file 到 ai_chat，输出 LeakPath
```

如果第 3 步找不到外发日志，也可以生成较弱的 visual-only 风险事件；但更理想、更可解释的是 `log + OCR` 双证据 join。

## 14. 当前实现和建议补强

当前仓库已经有这些能力：

- `normalize_logs(...)`：日志归一化和时间轴映射。
- `FrameObservation`：OCR/VLM 结果变成结构化观察。
- `EventCorrelator`：敏感文件、lineage、日志/观察绑定、upload candidate、Datalog facts。
- `DatalogEngine`：taint 传播和 `LeakPath` 查询。

当前非 VLM OCR-only 基础链路已经补上：OCR 文本会抽取 `mentioned_files`、sink/transfer 上下文，并写入 `FrameObservation.resource/related_resources/description`；`EventCorrelator` 会优先把显式外发日志和同文件 OCR 观察绑定成双证据 `CorrelatedEvent`；同一原始/当前文件的上传候选会合并证据，避免重复 leak path 刷屏。

后续要把非 VLM 策略做得更硬，还应继续补强：

1. **更细的 OCR fact extraction**  
   目前已抽 `mentioned_files` 和 sink/transfer 上下文；后续可以把 `sink_words`、`action_words` 做成更明确的结构化字段，而不是只放在 description 前缀里。

2. **更可解释的 log-OCR join 输出**  
   当前 join 逻辑已经在 `EventCorrelator` 内按时间、同文件、外发动作和 sink context 打分选择 OCR 观察；后续可以把 join reason 写进报告，便于直接解释“四条件命中”。

3. **更完整 lineage**  
   明确支持 `save_as`、`print_to_pdf`、`rename`、`screenshot`、`base64_encode` 等派生关系。

这些补强完成后，不用 VLM 也能把更多“日志 + OCR 文字”稳定推理成泄露链路。
