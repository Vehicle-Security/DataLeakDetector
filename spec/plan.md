Final Design:
https://docs.google.com/presentation/d/16_dCxZe9O5YCsyrnrv52AuKX8p3fhyQtr-aT9he6M-8/edit?usp=sharing

挑战
1. 视频高效率分析

2. 文件隐藏行为识别

3. 检测复杂跨过程威胁


方案设计

模块1: 视频推断引擎

模块2: 主Agent Workflow及FileTracker


模块3: 威胁检测

1. 已知威胁：基于Datalog的证据推理引擎得到精确的结果
定义三种Datalog关系：
- 文件访问：OpenFile(OpertionID, Proc, File)
- 文件转移：TransferFile(OpertionID, Proc, SrcFile, DstFile)
- 文件外泄：LeakFile(OpertionID, Proc, File)
说明：OpertionID会关联操作时间，关联的原始log和帧，以及详细描述。后面可用于LLM进一步分析验证。

利用LLM解析Log和分析视频生成以上三种规则的明确Datalog语句。再用Datalog去推理：
SearchLeak: find a path from open to leak
/* 
 * 基础事实定义 
 * OperationID: 唯一标识，关联时间戳、原始上下文
 * Proc: 进程名或应用名 (e.g., "Excel.exe", "WeChat.exe")
 * File: 文件路径或标识符 (e.g., "C:/Confidential/salary.xlsx")
 */

// 1. 进程打开了某个文件 (作为数据源头)
.decl OpenFile(OperationID: symbol, Proc: symbol, File: symbol)

// 2. 进程内的文件/数据转移 (或跨进程转移)
// SrcFile 和 DstFile 可以是文件路径，也可以是剪贴板(Clipboard)，或者是内存对象
.decl TransferFile(OperationID: symbol, Proc: symbol, SrcFile: symbol, DstFile: symbol)
///8/99/79988/7/877787777877/776//947//789/879/7/587/9=+0
// 3. 进程执行了外泄操作 (网络发送、保存到外部介质等)
.decl LeakFile(OperationID: symbol, Proc: symbol, File: symbol)


// --- 声明类型 ---
.type OperationID <: symbol
.type Proc <: symbol
.type Data <: symbol // 文件、剪贴板或内存对象

// --- 输入关系 (由 LLM 生成) ---
.decl OpenFile(id: OperationID, p: Proc, f: Data)
.decl TransferFile(id: OperationID, p: Proc, src: Data, dst: Data)
.decl LeakFile(id: OperationID, p: Proc, f: Data)

// --- 中间推导关系：污点追踪 ---
// Tainted 表示：在执行了 path 中的操作序列后，进程 p 持有的数据 d 是敏感的
// Path 是一个字符串，拼接了 OperationID，用于回溯
.decl Tainted(p: Proc, d: Data, path: symbol)

// --- 最终输出关系：发现泄露 ---
// 记录完整的泄露链条
.decl SearchLeak(StartOp: OperationID, EndOp: OperationID, LeakingProc: Data, LeakedFile: Data, FullPath: symbol)

// --- 规则定义 ---

// 1. 污染源头 (Source)
// 如果一个进程打开了一个敏感文件，那么这个文件在该进程中被标记为 Tainted。
// 这里的 "Sensitive" 是一个假设的过滤器，或者我们假设所有 OpenFile 的都是敏感的。
Tainted(p, f, id) :- 
    OpenFile(id, p, f).

// 2. 污染传播 (Propagation)
// 如果进程 p 持有被污染的数据 src (基于之前的路径 history)，
// 并且执行了 TransferFile 将 src 转移到 dst (操作 id)，
// 那么 dst 也被污染，路径更新为 history -> id。
Tainted(p, dst, cat(history, " -> ", id)) :- 
    Tainted(p, src, history),
    TransferFile(id, p, src, dst).

// 3. 跨进程传播 (可选，如果 TransferFile 支持跨进程逻辑)
// 如果 TransferFile 表示从 p1 的 src 传到 p2 的 dst (例如通过剪贴板或共享文件)
// 需要修改 TransferFile 的定义或者增加专门的 IPC 规则。
// 假设 TransferFile 涵盖了 "Proc A 写入 SharedFile" 和 "Proc B 读取 SharedFile"：
// 这种情况下，TransferFile 通常需要拆解，或者依靠 Tainted 规则的传递性。
// 简化起见，假设 TransferFile 就在同一 Proc 内，或者 Proc 字段代表系统级视角。

// 4. 泄露检测 (Sink)
// 如果进程 p 持有被污染的数据 f (路径 history)，
// 并且执行了 LeakFile 操作将 f 发送出去 (操作 id)，
// 则触发 SearchLeak。
SearchLeak(first_op, id, p, f, cat(history, " -> ", id)) :- 
    Tainted(p, f, history),
    LeakFile(id, p, f),
    // 提取起始操作ID用于展示 (这里简化处理，实际需字符串解析或额外参数传递)
    // 假设 history 的开头就是 StartOp
    first_op = substr(history, 0, length(id)). // 仅作示意，Datalog字符串处理较弱

// --- 输出结果 ---
.output SearchLeak

2. 未知威胁：基于LLM去模糊推理
将所有log，以及MLLM分析出来的原始信息存到Neo4J里面。再利用FPGrow等算法挖掘出和敏感文件存在时空关联的进程（时间上一起打开，空间上界面并排出现在屏幕里面）。读取这些片段，再利用MLLM来分析这段视频的行为。

[图片]
Agent工作流：
[图片]
System Prompt:
你是一个高级数据安全分析专家。你的任务是分析由 Neo4j 图数据库挖掘出的可疑行为对。
你需要协调下属工具来验证是否存在“模态转换数据泄露”（例如将敏感文档转为音频、图片）。

当前输入上下文：
- 进程: python.exe (PID: 4432)
- 行为: 读取 'confidential_Q3.pdf', 写入 'test_audio.wav'
- 时间差: 200ms
- 用户: dev_user_01

请按照以下步骤思考 (Chain of Thought):
1. **分析行为合理性**: 开发人员用 Python 处理 PDF 转音频是否正常？
2. **调用工具验证**: 使用 `multimodal_content_check` 工具对比源文件和目标文件内容。
3. **判断风险**: 
   - 如果相似度 > 0.8，极大概率为泄露。
   - 如果相似度 < 0.2，可能是正常的测试数据生成。
4. **执行决策**: 如果风险极高，调用 `kill_process` 并生成报告。

请开始分析。
