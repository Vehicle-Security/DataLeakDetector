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
- grid2 复核中同样表现为 `artifacts/all_data_direct_keyframes_grid2/3-Messaging-dingding-1_logs_1263` 已抽到关键帧，VLM 识别 `direct_leak / chat_upload`，但 `correlated_events=0`、`LeakFile=0`。

建议修法：

- 将 `record_id` 改为 `3-Messaging-dingding-1`。
- 将 `sensitive_file_path` 改为该样例日志/真实文件操作中的完整源路径，而不是只写 `产品设计方案.docx`。
- 如存在多个 `产品设计方案(1).docx`、`产品设计方案(2).docx` 等派生/副本，groundtruth 的初始敏感源仍应只标第一个源文件；派生关系交给日志和推理链路处理。

## grid2 批量检查新增疑似数据问题

以下条目来自 `artifacts/all_data_direct_keyframes_grid2` 的错报/漏报复核。这里先只记录更像数据标注或评测口径问题的样例；本地转换、压缩、重命名等更像检测规则误报的样例不放在本节。

### stage1 / 1-email-protonmail-1

- 样例路径：`spec/data/nas_samples/stage1/1-email-protonmail-1`
- 当前输出：`artifacts/all_data_direct_keyframes_grid2/1-email-protonmail-1_logs_1401`
- 问题文件：`groundtruth.json` / 评测期望结论
- 问题类型：operation 明确写邮箱外发，但期望结论为 no leak。
- 处理状态：已通过 `spec/config/groundtruth_policy.json` 增加 `邮箱外发` 口径覆盖；重跑后期望结论应为 leak。

当前 `groundtruth.json` 只有 1 条操作：

- `operation`: `敏感操作-邮箱外发`
- `sensitive_file_path`: `C:\Users\clhcl\Desktop\客户联系方式 .pdf`

但 `verdict_check.json` 中 `expected_conclusion` 是 `no_confirmed_data_leak`。VLM 和推理链均识别到 Proton Mail 邮件附件外发并生成 `mail_attachment` 类型 leak path。建议把该样例期望改为 `data_leak_risk_detected`，或明确说明为什么“敏感操作-邮箱外发”不算外泄。

### stage1 / 1-email-Outlook-1

- 样例路径：`spec/data/nas_samples/stage1/1-email-Outlook-1`
- 当前输出：`artifacts/all_data_direct_keyframes_grid2/1-email-Outlook-1_logs_3279`
- 问题文件：`groundtruth.json`
- 问题类型：groundtruth 标注疑似串样例，敏感源与视频/VLM 外发文件不匹配。

当前 groundtruth 中的敏感源是：

- `D:\DataLeakDetector\DataLeakDetector-main\ScreenMonitor\winows_monitor\test_files\公司机密条款.docx`
- `operation`: `潜在隐藏行为-打印-公司机密条款.docx`

但当前样例的关键帧/VLM 识别到的是 Outlook Web 邮件附件外发 `公司合作合同.docx`，且描述中包含个人邮箱收件人。由于 groundtruth 抽出的敏感源是 `公司机密条款.docx`，EventCorrelator 无法把 VLM 里的 `公司合作合同.docx` 回连到敏感源，最终 `correlated_events=0`、`LeakFile=0`、`leak_paths=[]`。

建议人工复核该 groundtruth 是否来自打印样例或其他会话；若该样例确实是 Outlook 邮件外发，应将 `record_id`、`operation` 和 `sensitive_file_path` 改为当前视频中的真实 Outlook 外发文件。

### stage1 / 2-ai-Gemini-3

- 样例路径：`spec/data/nas_samples/stage1/2-ai-Gemini-3`
- 当前输出：`artifacts/all_data_direct_keyframes_grid2/2-ai-Gemini-3_logs_2650`
- 问题文件：`groundtruth.json`
- 问题类型：groundtruth 标识串样例，且敏感源只有裸文件名，无法提取可靠完整路径。

当前 groundtruth 存在以下问题：

- `record_id` 写成 `3-Messaging-dingding-1`，与目录 `2-ai-Gemini-3` 不匹配。
- JSON 使用重复引号形式，虽然评估器能宽松解析一部分，但不是规范 JSON。
- `sensitive_file_path` 只有 `产品设计方案.docx`，没有完整源路径。
- `operation` 写的是 `直接外发-通讯工具发送`，但当前样例关键帧/VLM 识别到的是 Gemini AI chat 上传。

当前 VLM 已经识别到 `direct_leak / ai_chat_upload`，文件名为 `产品设计方案.docx`，但由于 groundtruth 无法提供完整敏感源，`extract_sensitive_sources(...)` 结果为空，最终无法生成 `LeakFile`。建议改为当前 Gemini 样例的正确 `record_id`、真实完整敏感源路径，以及 `直接外发-AI对话上传` 一类操作文案。

### stage1 / 4-Drive-kuake-3

- 样例路径：`spec/data/nas_samples/stage1/4-Drive-kuake-3`
- 当前输出：`artifacts/all_data_direct_keyframes_grid2/4-Drive-kuake-3_logs_2750`
- 问题文件：`groundtruth.json`
- 问题类型：groundtruth 只标了重命名/查看监控状态，疑似漏标云盘上传。

当前 groundtruth 标注为两次重命名和一次查看监控状态，因此评测期望为 no leak。但视频/关键帧中可以看到用户在夸克网盘选择并上传 `公司合作合同.docx`，推理链也生成了 `network_upload` leak path。建议人工复核该样例真实任务：如果云盘上传是有效行为，应补一条外发/云盘上传操作并将期望改为 leak；如果不是，应说明为什么上传画面不计入标注。

### stage1 / 5-meeting-feishu-1

- 样例路径：`spec/data/nas_samples/stage1/5-meeting-feishu-1`
- 当前输出：`artifacts/all_data_direct_keyframes_grid2/5-meeting-feishu-1_logs_340`
- 问题文件：`groundtruth.json` / 评测期望结论
- 问题类型：operation 写屏幕共享展示敏感文件，但期望结论为 no leak。
- 处理状态：已通过 `spec/config/groundtruth_policy.json` 增加 `屏幕共享展示敏感文件` 口径覆盖；重跑后期望结论应为 leak。

当前 groundtruth 包含：

- `潜在隐藏行为-wps打开敏感文件`
- `潜在隐藏行为-Lark会议屏幕共享展示敏感文件`

其中第二条已经描述了通过 Lark 会议屏幕共享展示敏感文件。若当前评测口径认为会议共享敏感文件属于泄露，应将该样例期望改为 `data_leak_risk_detected`；若会议共享不算外泄，需要在数据说明中单独定义该例为非泄露。

### stage1 / 5-meeting-Tencent-4

- 样例路径：`spec/data/nas_samples/stage1/5-meeting-Tencent-4`
- 当前输出：`artifacts/all_data_direct_keyframes_grid2/5-meeting-Tencent-4_logs_619`
- 问题文件：`groundtruth.json` / 评测期望结论
- 问题类型：录屏/会议共享类口径不一致。

当前 groundtruth 标注：

- `潜在隐藏行为-录屏-C:/Users/46521/Desktop/数据采集测试/公司合作合同.docx`

但关键帧/VLM 显示腾讯会议屏幕共享环境下敏感文件内容被展示，检测链生成了会议共享相关 leak path。建议人工确认该样例究竟是“本地录屏派生文件”还是“会议屏幕共享外泄”。如果是屏幕共享，应改为 leak；如果只是本地录屏，应修正视频/标注或为检测规则提供排除口径。

### stage1 / 7-Tech community-solo-1

- 样例路径：`spec/data/nas_samples/stage1/7-Tech community-solo-1`
- 当前输出：`artifacts/all_data_direct_keyframes_grid2/7-Tech-community-solo-1_logs_2573`
- 问题文件：`groundtruth.json` / 评测期望结论
- 问题类型：operation 写“复制内容外发”，但期望结论为 no leak。
- 处理状态：已通过 `spec/config/groundtruth_policy.json` 增加 `复制内容外发` / `内容外发` 口径覆盖；重跑后期望结论应为 leak。

当前 groundtruth 标注：

- `潜在隐藏行为-复制内容外发`
- `sensitive_file_path`: `D:\DataLeakTest\docx\公司合作合同.docx`

按字段语义，“复制内容外发”更像泄露行为，但 `verdict_check.json` 中期望为 `no_confirmed_data_leak`。建议改为 leak，或将 operation 文案改成非外发的本地复制/编辑行为。

### stage2 / 2-filestruct-rename-1

- 样例路径：`spec/data/nas_samples/stage2/2-filestruct-rename-1`
- 当前输出：`artifacts/all_data_direct_keyframes_grid2/2-filestruct-rename-1_logs_1891`
- 问题文件：`groundtruth.json`
- 问题类型：JSON 格式和样例标识明显错误。

当前 artifact 中复制出的 `groundtruth.json` 存在以下问题：

- `record_id` 写成 `1-email-163email-2`，与样例目录 `2-filestruct-rename-1` 不匹配。
- Windows 路径里的反斜杠未转义，例如 `D:\gdata\documents_1\...`，导致标准 JSON 解析失败。
- `operations` 数组最后一个元素后有多余逗号。
- 文案存在明显编码异常。

建议修正源数据中的 `record_id`、路径转义、尾逗号和中文编码后，再判断该样例应属于本地重命名/派生还是外泄。

### stage1 / 7-Tech community-bokeyuan-1

- 样例路径：`spec/data/nas_samples/stage1/7-Tech community-bokeyuan-1`
- 当前输出：`artifacts/all_data_direct_keyframes_grid2/7-Tech-community-bokeyuan-1_logs_297`
- 问题文件：`INDEX.md` / `video/` / `groundtruth.json`
- 问题类型：同一 case 目录内存在多个录屏，`INDEX.md` 指向的视频与 groundtruth 时间不一致。

当前目录中有两个视频：

- `video/recording_20260309_095252.mp4`，时长约 71.3 秒。
- `video/recording_20260420_222228.mp4`，时长约 35.6 秒。

`groundtruth.json` 中 `recording_start_time` 是 `2026-03-09 09:52:52`，对应 `recording_20260309_095252.mp4`；但 `INDEX.md` 的 Session ID 和 File List 指向 `recording_20260420_222228.mp4`。当前 `discover_data_case(...)` 会优先按 `INDEX.md` 选择视频，因此实际运行使用了 20260420 视频。

影响：

- `artifacts/all_data_direct_keyframes_grid2/7-Tech-community-bokeyuan-1_logs_297.json` 中 `video_file` 是 `recording_20260420_222228.mp4`，但 `recording_start_ms` 来自 20260309 groundtruth。
- 抽帧阶段出现 `no_keyframes_selected`，`keyframes_raw_files=0`、`vlm_frames=0`。
- `frame_observations.json` 中日志锚点时间是数十亿毫秒级，明显不在视频相对时间坐标内。

建议修法：

- 若该样例应使用 20260309 会话，应修正 `INDEX.md` 指向 `video/recording_20260309_095252.mp4`，或删除/移动不属于该样例的 20260420 视频与索引。
- 若该样例应使用 20260420 会话，则应同步修正 `groundtruth.json` 的 `record_id`、`recording_start_time`、日志和敏感文件标注。

## stage2_vlm correct=false 中明确的数据问题

本节只记录 `artifacts/all_data_release_matrix_neo4j/stage2_vlm.log` 中 `correct=false`，且在 `spec/docs/错误.md` 里明确标注为“数据问题”的条目。仅“没截到关键帧/没截到图”的条目属于抽帧或代码问题，不放在这里。

### stage1 / 8-git-GitCode-1

- 样例路径：`spec/data/nas_samples/stage1/8-git-GitCode-1`
- 当前输出：`artifacts/all_data_release_matrix_neo4j/vision_precompute/8-git-GitCode-1/8-git-GitCode-1_logs_1056`
- stage2_vlm：line 824，`detector=no_confirmed_data_leak`，`expected=data_leak_risk_detected`
- 问题类型：样例标识/场景与标注不匹配。

当前 `groundtruth.json` 的 `record_id` 为 `1`，operation 写有 `直接外发-上传文件发送`、`直接外发-粘贴外发`，但复核备注指出该样例未涉及 GitCode。建议人工核对该目录的视频、日志、groundtruth 是否串入了其他上传/粘贴样例；如果该视频不是 GitCode 场景，应修正目录/record_id/标注，或移出 `8-git-GitCode-1` 正例集合。

### stage2 / 4-contentchange-transfer-2

- 样例路径：`spec/data/nas_samples/stage2/4-contentchange-transfer-2`
- 当前输出：`artifacts/all_data_release_matrix_neo4j/vision_precompute/4-contentchange-transfer-2/4-contentchange-transfer-2_logs_556`
- stage2_vlm：line 944，`detector=no_confirmed_data_leak`，`expected=data_leak_risk_detected`
- 问题类型：groundtruth 标注文件上传，但样例片段中没有文件上传。

`groundtruth.json` 的 `record_id` 为 `1-clipboard-email-1`，与目录 `4-contentchange-transfer-2` 不一致；operation 中两次写 `直接外发-文件上传-WPS`。复核备注明确指出该样例没有文件上传。建议核对 groundtruth 是否串入 clipboard/email/WPS 上传样例；若当前视频只包含复制/翻译等中间行为，应改为 `suspicious_behavior_detected` 或补齐真正的上传片段。

### stage2 / 5-screen-screenshot-3

- 样例路径：`spec/data/nas_samples/stage2/5-screen-screenshot-3`
- 当前输出：`artifacts/all_data_release_matrix_neo4j/vision_precompute/5-screen-screenshot-3/5-screen-screenshot-3_logs_2553`
- stage2_vlm：line 964，`detector=no_confirmed_data_leak`，`expected=data_leak_risk_detected`
- 问题类型：groundtruth 标注截图外发，但样例未覆盖截图外发。

建议将该例从 `data_leak_risk_detected` 调整为只记录截图派生/可疑行为。

### stage2 / 5-screen-screenshot-4

- 样例路径：`spec/data/nas_samples/stage2/5-screen-screenshot-4`
- 当前输出：`artifacts/all_data_release_matrix_neo4j/vision_precompute/5-screen-screenshot-4/5-screen-screenshot-4_logs_1363`
- stage2_vlm：line 974，`detector=no_confirmed_data_leak`，`expected=data_leak_risk_detected`
- 问题类型：groundtruth 标注截图工具截图敏感文件，但样例未覆盖截图外发。
