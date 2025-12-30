## 目标
分析一组帧的 操作应用"app_name", 操作的资源名称"resource_name",操作时间"operation_time",操作类型"operation_type", 判断依据"reasoning"。

## 输入
1. 一个场景的关键帧序列和对应的时间戳
2. 系统日志文件（可选，增强分析准确性）

## 输出
类似：
{
    "app_name": "百度网盘",
    "resource_name": "公司2025战略规划.pptx",
    "operation_time": "14.0s",
    "operation_type": "上传文件",
    "reasoning": "用户在百度网盘中上传了一个敏感的公司战略规划文件"
}

## 实现过程
基于langgraph创建了3个node：
1. **reason_node**：让LLM遵循 ReAct 模式来分析用户的屏幕录像关键帧和系统日志，回答：
   - Thought: 思考我目前知道了什么，还需要知道什么
   - Action: 工具名称 (analyze_gui_frames 或 analyze_system_logs)
   - Action Input: 工具的入参
   
2. **act_node**：负责"手"部分。解析 LLM 的文本输出，并执行相应的工具（analyze_gui_frames 或 analyze_system_logs）来完成任务。

3. **observation_node**：负责"眼睛"部分。将工具的执行结果格式化，并追加到对话历史中，让 LLM 在下一次 Reason 时能看到。

## 可用工具
1. **analyze_gui_frames**: 调用视觉模型分析GUI界面，识别应用界面、用户交互等信息
   - 参数格式: `{"frame_indices": [...], "query": "..."}`
   
2. **analyze_system_logs**: 分析系统日志，获取精确的文件操作、进程信息、应用名称等结构化数据
   - 参数格式: `{"time_range": "开始时间-结束时间(可选)", "query": "查询目标"}`

## 执行过程
1. 先执行 **reason**，产生两类结果：
   - 中间指令（Thought、Action、Action_input）→ 进入act节点
   - 最终结果 Final_Answer(此时帧序列的所有信息都被获取) → END

2. 如果是中间结果：
   - 进入 **act** 节点，执行相应工具（优先使用系统日志分析）
   - 执行完毕后进入 **observation**，记录执行结果
   - 返回 **reason** 继续思考

3. 如果是最终结果：流程结束

## 运行
1. 替换关键帧路径
2. 配置api-key
3. （可选）配置系统日志文件路径

## 控制台输出案例
🚀 启动数据泄露敏感行为分析 Agent...


🧠 [Reasoning] Qwen 正在思考...

🧠 [Reasoning] 
Thought: 首先，我需要了解在关键帧序列中用户进行了哪些敏感操作。这可以通过分析系统日志来获取精确的文件操作、进程和应用信息。因此，我将首先调用analyze_system_logs工具，并指定时间范围为92.7s到114.0s，以覆盖所有关键帧的时间段。

Action: analyze_system_logs
Action Input: {"time_range": "92.7s-114.0s", "query": "文件操作信息"}
Observation:

--- [Act Node] 解析指令 ---
>> 执行系统日志分析，Time Range: 92.7s-114.0s
   [LogAnalyzer] 正在分析系统日志... Query: 文件操作信息, Time Range: 92.7s-114.0s

--- [Observation Node] ---

🧠 [Reasoning] Qwen 正在基于 Observation 思考...

🧠 [Reasoning] 
Thought: 通过系统日志，我了解到在指定的时间范围内发生了多个文件操作。其中，记录5显示了一个敏感操作：用户在D:\DouBao\app\Doubao.exe进程中修改了名为"公司核心技术文档.pdf"的文件。这个操作发生在2025-12-07T15:09:10.111。为了确认这个操作是否涉及数据泄漏，我需要查看关键帧序列中对应时间点的画面，以了解用户的实际操作行为。

Action: analyze_gui_frames
Action Input: {"frame_indices": [6, 7, 8], "query": "用户正在执行的操作"}
Observation:

--- [Act Node] 解析指令 ---
>> 执行 VLM 关键帧分析，Indices: [6, 7, 8]
   [VLM] 正在调用 Qwen2-VL 分析 3 张图片... Query: 用户正在执行的操作
视觉大模型识别结果:  用户正在执行的操作是：

1. 在一个聊天界面中，用户上传了一个名为“公司核心技术文档.pdf”的文件。
2. 用户请求帮助提取该文档的关键要点。

--- [Observation Node] ---

🧠 [Reasoning] Qwen 正在基于 Observation 思考...

🧠 [Reasoning] 
Thought: 根据系统日志和视觉分析，用户在D:\DouBao\app\Doubao.exe进程中上传了一个名为"公司核心技术文档.pdf"的文件，并请求帮助提取该文档的关键要点。这个操作发生在2025-12-07T15:09:10.111。结合这两种数据源，可以确认这是一个敏感操作，可能涉及数据泄漏。

Final Answer: {
    "app_name": "豆包", 
    "resource_name": "公司核心技术文档.pdf", 
    "operation_time": "2025-12-07T15:09:10.111", 
    "operation_type": "上传文件并请求关键要点提取", 
    "reasoning": "根据系统日志记录和关键帧视觉分析，用户在指定时间范围内通过豆包应用上传了敏感文件，并请求关键信息提取，这表明可能存在数据泄漏风险。"
}

✅ 分析结束。最终结果:
{
    "app_name": "豆包",
    "resource_name": "公司核心技术文档.pdf",
    "operation_time": "2025-12-07T15:09:10.111",
    "operation_type": "上传文件并请求关键要点提取",
    "reasoning": "根据系统日志记录和关键帧视觉分析，用户在指定时间范围内通过豆包应用上传了敏感文件，并请求关键信息提取，这表明可能存在数据泄漏风险。"
}

