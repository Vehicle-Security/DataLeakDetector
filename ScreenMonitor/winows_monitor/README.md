# 数据泄露检测系统 - 监控模块 (Win Monitor)

本模块主要负责监控 Windows 系统下的文件操作行为。

## 功能特性

- 📹 屏幕录制与行为捕获
- 📂 文件系统监控 (watchdog)
- 🌐 **浏览器文件访问监控** (EtwMonitor) - 捕获浏览器上传文件等操作
- 📋 剪贴板监控
- 🔍 敏感文件检测

## 安装指南

确保已安装 Python 环境，然后在当前目录下运行以下命令安装依赖：

```bash
pip install -r requirements.txt
```

## 启动指南

### ⚠️ 重要：需要管理员权限

EtwMonitor 需要管理员权限才能捕获浏览器文件访问事件。

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

## 更新日志

详见 [CHANGELOG_2026-01-19.md](./CHANGELOG_2026-01-19.md)

