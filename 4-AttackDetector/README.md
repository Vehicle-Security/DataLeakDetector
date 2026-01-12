# 4-AttackDetector: Agent Complex Attack Detection Module

基于 **Datalog (Soufflé)** 的复杂攻击识别模块，通过分析日志中的间接关联链来检测高级数据泄露攻击。

## 设计思路

### 问题背景

传统的黑名单规则匹配只能检测**直接攻击**（如访问敏感网站、打开敏感应用）。但高级攻击者会使用**间接路径**：

1. **剪贴板外泄**：在 Word 中复制敏感内容 → 切换到浏览器 → 粘贴到 AI 网站
2. **文件重命名规避**：将 `机密合同.docx` 重命名为 `notes.txt` → 上传到云盘
3. **截图泄密**：打开敏感文档 → 截图 → 通过 IM 发送图片
4. **时间窗口关联**：在 5 分钟内，先打开敏感文件，再访问上传网站

### 解决方案

使用 **Datalog** 这种声明式逻辑编程语言来表达复杂的时序关联规则：

```prolog
// 检测：敏感文件打开后，同一用户在时间窗口内访问了 AI 网站
potential_leak(File, URL, T1, T2) :-
    open_file(_, File, T1),
    sensitive_file(File),
    browser_access(_, URL, T2),
    ai_site(URL),
    T2 > T1,
    T2 - T1 < 300.  // 5分钟内
```

**为什么选择 Datalog/Soufflé？**
- ✅ 自然表达时序关系和状态关联
- ✅ 高效处理大量事实（日志条目）
- ✅ 规则可组合、可扩展
- ✅ Soufflé 是工业级 Datalog 引擎，性能优异

---

## 模块架构

```
4-AttackDetector/
├── README.md                    # 本文档
├── requirements.txt             # Python 依赖
├── log_to_facts.py              # 日志 → Datalog 事实转换器
├── detector.py                  # 主检测引擎
├── rules/
│   ├── schema.dl                # 类型声明和基础事实
│   ├── file_browser.dl          # 文件-浏览器关联规则
│   ├── clipboard.dl             # 剪贴板外泄规则
│   └── advanced.dl              # 高级组合规则
├── example/
│   ├── mock_events.json         # 模拟攻击场景日志
│   └── run_example.py           # 运行示例
└── output/                      # 生成的事实和查询结果
```

---

## 运行流程

### 1. 安装依赖

```bash
# 安装 Python 依赖
pip install -r requirements.txt

# 安装 Soufflé (Windows 使用 WSL)
# Ubuntu/Debian:
sudo apt install souffle

# macOS:
brew install souffle
```

### 2. 准备日志数据

日志格式兼容 `win_monitor` 输出的 JSON 格式：

```json
{
    "timestamp": "2026-01-09T10:30:00.000",
    "event_type": "opened",
    "file_path": "C:\\Documents\\机密合同.docx",
    "process_info": {"process_name": "WINWORD.EXE"},
    ...
}
```

### 3. 运行检测

```bash
cd 4-AttackDetector

# 使用示例数据运行
python example/run_example.py

# 或使用真实日志
python detector.py --input path/to/events.json --output output/
```

### 4. 查看结果

检测结果输出为 CSV 和人类可读的报告：

```
=== 检测结果 ===

[HIGH] 潜在数据泄露检测到 3 个事件:

1. 文件-浏览器关联
   时间: 10:30:00 - 10:32:45
   文件: C:\Documents\机密合同.docx
   目标: https://doubao.com/chat

2. 剪贴板外泄嫌疑
   时间: 10:35:00 - 10:35:30
   来源: Microsoft Word
   目标: Chrome (doubao.com)
```

---

## Datalog 规则说明

### 基础事实 (Facts)

从日志自动生成的基础事实：

| 事实 | 描述 |
|------|------|
| `open_file(process, path, time)` | 文件打开事件 |
| `create_file(process, path, time)` | 文件创建事件 |
| `modify_file(process, path, time)` | 文件修改事件 |
| `rename_file(process, src, dst, time)` | 文件重命名事件 |
| `app_switch(from, to, time)` | 应用切换事件 |
| `browser_access(browser, url, time)` | 浏览器访问事件 |

### 推导规则 (Rules)

**规则 1: 敏感文件访问后上传嫌疑**
```prolog
potential_upload(File, URL, T1, T2) :-
    open_file(P1, File, T1),
    sensitive_extension(File),
    browser_access(P2, URL, T2),
    upload_site(URL),
    T2 > T1,
    T2 - T1 < 300.
```

**规则 2: 文件重命名规避检测**
```prolog
rename_evasion(Orig, New, URL, T1, T2) :-
    rename_file(_, Orig, New, T1),
    sensitive_extension(Orig),
    not sensitive_extension(New),
    browser_access(_, URL, T2),
    T2 > T1,
    T2 - T1 < 600.
```

**规则 3: 跨应用数据流追踪**
```prolog
cross_app_leak(App1, App2, T1, T2) :-
    app_switch(App1, App2, T2),
    sensitive_app(App1),
    risky_app(App2),
    open_file(App1, _, T1),
    T2 - T1 < 120.
```

---

## 结果解读

### 风险等级

| 等级 | 描述 | 示例 |
|------|------|------|
| **CRITICAL** | 高度确认的数据泄露 | 敏感文件内容直接上传到外部网站 |
| **HIGH** | 强关联的泄露嫌疑 | 敏感文件打开后访问 AI 网站 |
| **MEDIUM** | 需要人工确认的可疑行为 | 频繁的应用切换和文件操作 |
| **LOW** | 信息性提示 | 非敏感文件的正常操作 |

### 输出文件

- `output/facts/` - 生成的 Datalog 事实文件
- `output/results/potential_upload.csv` - 潜在上传事件
- `output/results/rename_evasion.csv` - 重命名规避事件
- `output/report.md` - 人类可读的检测报告

---

## 扩展规则

可以通过添加新的 `.dl` 文件来扩展检测能力：

```prolog
// rules/custom.dl - 自定义规则示例

// 检测截图后发送
screenshot_leak(File, Target, T1, T2) :-
    screenshot(_, File, T1),
    app_switch(_, Target, T2),
    im_app(Target),
    T2 > T1,
    T2 - T1 < 180.
```

---

## 参考资料

- [Soufflé Documentation](https://souffle-lang.github.io/docs.html)
- [Datalog Tutorial](https://souffle-lang.github.io/tutorial)
- [Static Analysis with Datalog](https://yanniss.github.io/doop-oopsla09.pdf)
