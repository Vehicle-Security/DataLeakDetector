# Windows Screen Monitor 更新说明

**更新日期**: 2026-01-31  
**更新人**: 开发团队

---

## 🎯 2026-01-31 更新：项目清理与优化

### 控制台输出精简
- **移除冗余输出**：删除了大量 `[FS] modified/created` 调试日志
- **禁用 Flask 请求日志**：不再显示 `127.0.0.1 - - [date] "GET /api/status...` 
- **统一日志系统**：将 `print()` 替换为 `app_logger`，只保留关键信息

### 代码清理
- **删除冗余脚本**：
  - `cleanup_keyevents.py` - 已集成到 Engine
  - `merge_etw_to_keyevents.py` - 已集成到 Engine
- **简化生命周期管理**：文件系统监控完全由 Engine 管理

### 改进
- 控制台输出更清爽，只显示重要事件
- 项目结构更简洁

---

## 🎯 2026-01-26 更新：EtwMonitor.exe 集成

### 问题背景
浏览器访问本地文件时（如通过文件选择对话框上传），原有的 watchdog 和 pywintrace 无法捕获这些事件。

### 解决方案
集成 `log/bin/EtwMonitor.exe`（基于 Windows ETW 内核事件），专门捕获浏览器进程的文件访问。

### 新增文件

| 文件 | 说明 |
|------|------|
| `core/monitors/etw_launcher.py` | EtwMonitor.exe 启动器，负责进程管理和日志格式转换 |

### 修改文件

| 文件 | 修改内容 |
|------|----------|
| `core/monitors/engine.py` | 录制开始时启动 EtwMonitor，结束时停止并合并日志 |
| `core/monitors/engine.py` | 修复重复调用 `clipboard_monitor.start()` 的 bug |

### 启动方式

> ⚠️ **必须以管理员身份运行 `web_server.py`**，否则 EtwMonitor.exe 无法启动。

```powershell
# PowerShell (管理员)
cd path\to\win_monitor
python web_server.py
```

### 日志格式
EtwMonitor 捕获的事件会自动转换为 core 格式并合并到 `logs/logs.json`：

```json
{
  "event_type": "browser_file_access",
  "source": "etw_monitor",
  "app_name": "Chrome",
  "file_path": "C:\\Users\\xxx\\Documents\\secret.docx"
}
```

---

## 🎯 2026-01-19 更新

| Case | 问题描述 | 修复状态 |
|------|----------|----------|
| 46/47 | `INDEX.md` 录屏开始时间错误（显示结束时间） | ✅ 已修复 |
| 48 | 屏幕录制文件未被日志抓到 | ✅ 已修复 |
| 49 | 拍照识字泄露 - 文件未被捕获 | ✅ 已修复 |
| 50 | WPS 打开文件事件未被日志抓到 | ✅ 已修复 |

---

## 🆕 新增功能

### 1. ETW 文件打开监控

新增 **ETW (Event Tracing for Windows)** 监控器，类似 Mac 的 `fs_usage`：

- **精确捕获文件打开事件**（之前 watchdog 只能捕获文件变更）
- **完整进程信息**：PID、进程名、应用名
- **敏感文件检测**：自动标记包含关键字的文件（合同、机密、密码等）

**新增文件**: `core/monitors/etw_file_monitor.py`

### 2. ~~扩展监控路径~~ → **监控整台机器所有驱动器** ✅

**已调整为监控整个机器**：
- 自动检测并监控所有可用驱动器（C:\, D:\, E:\, ...）
- 不再限制于特定用户目录
- 覆盖整个文件系统

### 3. ~~放宽过滤规则~~ → **移除所有过滤规则** ✅

**已完全移除过滤**：
- ❌ 不再过滤任何文件扩展名（.tmp, .log 等也会被监控）
- ❌ 不再过滤任何路径（AppData、Temp 等也会被监控）
- ✅ 捕获整台机器的所有文件操作事件
- ⚠️ **注意**：会产生大量日志，请确保有足够的磁盘空间

---

## 📦 依赖安装

**新增依赖**（在 Windows 机器上执行）：

```bash
pip install pywintrace
```

---

## 🚀 启动方式

### 重要：需要管理员权限！

ETW 监控需要管理员权限运行（类似 Mac 需要 `sudo`）。

**方式一：使用批处理脚本**
```batch
# 右键点击 start_admin.bat -> 以管理员身份运行
```

**方式二：手动启动**
1. 右键点击 **命令提示符** 或 **PowerShell** → **以管理员身份运行**
2. 进入项目目录
3. 执行：
```bash
cd ScreenMonitor\win_monitor
python main.py
```

---

## 🔧 修改的文件清单

| 文件 | 修改内容 |
|------|----------|
| `core/monitors/engine.py` | 1. 修复 INDEX.md 录制时间；2. 集成 ETW 监控 |
| `core/monitors/file_system_monitor.py` | 1. **监控所有驱动器**；2. **移除所有过滤规则** |
| `core/monitors/etw_file_monitor.py` | **新增** - ETW 文件打开监控器；**移除所有过滤规则** |
| `requirements.txt` | 新增 `pywintrace` 依赖 |

---

## ✅ 验证方法

启动监控后，执行以下操作并检查日志：

1. **用 WPS 打开文档** → 应看到 `event_type: "opened"` 事件
2. **保存屏幕录制到 Videos 目录** → 应看到 `event_type: "created"` 事件
3. **通过 QQ 发送文件** → 应看到相关文件操作日志

日志文件位置：`output/session_xxx/logs.json`

---

## ⚠️ 注意事项

1. **必须以管理员身份运行**，否则 ETW 会报 `AccessDenied` 错误
2. 如果 `pywintrace` 安装失败，程序会自动降级为 **Recent Files 监控**（功能有限）
3. 首次运行可能需要允许防火墙/安全软件权限

---

## 📞 问题反馈

如遇问题，请提供：
1. 完整的错误日志
2. 操作系统版本
3. Python 版本（`python --version`）
4. 是否以管理员身份运行
