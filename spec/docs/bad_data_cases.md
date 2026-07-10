# 不规范数据记录

这个文件记录数据集里需要人工修正的样例。这里不修改 `spec/data` 原始数据，只说明问题、影响和建议修法。

## stage1 / 3-Messaging-dingding-1

- 样例路径：`spec/data/nas_samples/stage1/3-Messaging-dingding-1`
- 当前输出：`artifacts/all_data_direct_keyframes_grid1/3-Messaging-dingding-1_logs_1263`
- 问题文件：`groundtruth.json`
- 问题类型：groundtruth 标注与样例不匹配，且敏感文件只写了裸文件名。

当前 `groundtruth.json` 内容里有两处明显问题：

- `record_id` 写成了 `4-Drive-baidu-2`，但目录样例是 `3-Messaging-dingding-1`。
- `sensitive_file_path` 只有 `产品设计方案.docx`，没有完整源路径。检测逻辑本来应按路径确定初始敏感源，裸文件名容易和派生文件、同名文件、VLM 识别出的文件名混在一起。

影响：

- `extract_sensitive_sources(...)` 无法从该 groundtruth 提取可靠初始敏感源。
- VLM 已经识别出 DingTalk/钉钉文件上传行为和 `产品设计方案.docx`，但 EventCorrelator 没有可信敏感源表可关联，最终 `leak_paths.json` 为空。

建议修法：

- 将 `record_id` 改为 `3-Messaging-dingding-1`。
- 将 `sensitive_file_path` 改为该样例日志/真实文件操作中的完整源路径，而不是只写 `产品设计方案.docx`。
- 如存在多个 `产品设计方案(1).docx`、`产品设计方案(2).docx` 等派生/副本，groundtruth 的初始敏感源仍应只标第一个源文件；派生关系交给日志和推理链路处理。
