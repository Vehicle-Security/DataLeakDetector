## 目标
分析一组帧的 操作应用"app_name", 操作的资源名称"resource_name",操作时间"operation_time",操作类型"operation_type", 判断依据"reasoning"。
## 输入
一个场景的关键帧序列和对应的时间戳
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
1. reason_node：让LLM遵循 ReAct 模式来分析用户的屏幕录像关键帧，回答Thought: 思考我目前知道了什么，还需要知道什么。 Action: 工具名称 (analyze_gui_frames) Action Input(工具的入参)
2. act_node：负责“手”部分。解析 LLM 的文本输出，并执行analyze_gui_frames来完成任务。
3. observation_node：负责“眼睛”部分。将工具的执行结果格式化，并追加到对话历史中，让 LLM 在下一次 Reason 时能看到。

## 执行过程
先reason, reason会有两类结果：中间指令（Thought、Action、Action_input） 和 最终结果 Final_Answer(此时帧序列的所有信息都被获取)
如果reason生成的是中间结果，则进入act节点，执行analyze_gui_frames工具，执行完毕后observation，记录执行结果。
如果reason生成的是最终结果，则END

## 运行
1. 替换关键帧路径
2. 配置api-key
