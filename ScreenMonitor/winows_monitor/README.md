# 数据泄露检测系统 - 监控模块 (Win Monitor)

本模块主要负责监控 Windows 系统下的文件操作行为。

## 功能特性

- 📹 点击开始后立即进入手动持续录制
- 📂 文件系统监控 (watchdog)
- 🌐 **浏览器文件访问监控** (EtwMonitor) - 捕获浏览器上传文件等操作
- 📋 剪贴板监控
- 🔍 黑名单分类与敏感文件检测

## 安装指南

确保已安装 Python 环境，然后在当前目录下运行以下命令安装依赖：

```bash
pip install -r requirements.txt
```

## 启动指南

### ⚠️ 重要：需要管理员权限

EtwMonitor 需要管理员权限才能捕获浏览器文件访问事件。

### 推荐脚本

- 普通启动：`start_win_monitor.cmd`
- 管理员启动：`start_win_monitor_admin.cmd`
- PowerShell 入口：`start_win_monitor.ps1`

**方式一：PowerShell (管理员)**
```powershell
# 右键 PowerShell -> 以管理员身份运行
cd path\to\win_monitor
python web_server.py
```

**方式二：命令提示符 (管理员)**
```batch
:: 右键 命令提示符 -> 以管理员身份运行
cd path\to\win_monitor
python web_server.py
```

服务启动后，打开浏览器访问：

[http://localhost:5000](http://localhost:5000)

## 录制行为

- 点击“开始监控”后会立即创建会话并开始录制，直到手动点击“停止监控”
- `blacklist_apps` 和 `blacklist_websites` 仅用于窗口/网站风险分类，不再决定是否开始录制
- `logs/logs.json` 保留原始监控日志，允许 `app_switch` / `website_visit` 这类窗口事件的 `file_path=""`
- `logs/keyevents.json` 是下游统一契约，窗口事件若保留则必须已经绑定到精确文件路径
- 录制结束后的 `logs/` 目录只保留 `logs.json` 和 `keyevents.json`；ETW 中间 JSON 会被合并后清理掉
- `keyevents.json` 中：
  - `timestamp` 固定表示事件发生时间
  - `file_path` 固定表示事件涉及的完整文件路径
  - `app_switch` / `website_visit` 若存在，`file_path` 必须是完整精确路径；否则该事件不会进入 `keyevents.json`
  - `process_info.process_path` 固定表示应用程序路径
  - `file_path` 不会回填文件名，也不会回填 `process_info.process_path`

## 更新日志

详见 [CHANGELOG_2026-01-19.md](./CHANGELOG_2026-01-19.md)
