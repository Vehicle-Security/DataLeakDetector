# 统一日志格式说明 (Unified Log Format Specification)

本文档定义了 `Mac_monitor` 和 `win_monitor` 的统一日志输出格式。

> **注**: `keyevents.json` 是最终用于数据集标注的文件，格式为 **JSON Array**。  
> 完整原始日志 `logs.json` 和 ETW 浏览器日志 `etw_session_*.json` 仅作辅助参考。

---

## 日志文件格式

`keyevents.json` 使用 **JSON Array** 格式：

```json
[
  {"timestamp":"2026-02-11T12:27:29.457","event_type":"clipboard_text",...},
  {"timestamp":"2026-02-11T12:27:30.133","event_type":"app_switch",...},
  {"timestamp":"2026-02-11T12:27:36.000","event_type":"created",...}
]
```

---

## 完整示例

### 示例 1: 剪贴板文本事件 (clipboard_text)

```json
{
  "timestamp": "2026-02-11T12:27:29.457",
  "event_type": "clipboard_text",
  "file_path": "",
  "file_name": "",
  "file_size": 0,
  "file_extension": "",
  "process_info": {
    "pid": "11488",
    "process_name": "chrome.exe",
    "process_path": "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe"
  },
  "window_info": {
    "window_handle": "",
    "window_title": "",
    "window_class": ""
  },
  "user_info": {
    "username": "zbn20",
    "hostname": "zbn"
  },
  "disk_info": {
    "drive_letter": "",
    "disk_type": ""
  },
  "app_name": "Chrome",
  "content_preview": "python web_server.py",
  "content_hash": "a82b1ab641bf8838",
  "extra": {
    "raw_operation": "clipboard_text",
    "category": "",
    "source": "clipboard_monitor"
  }
}
```

### 示例 2: 应用切换事件 (app_switch)

```json
{
  "timestamp": "2026-02-11T12:27:30.133",
  "event_type": "app_switch",
  "file_path": "",
  "file_name": "",
  "file_size": 0,
  "file_extension": "",
  "process_info": {
    "pid": "11488",
    "process_name": "chrome.exe",
    "process_path": "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe",
    "cmdline": ""
  },
  "window_info": {
    "window_handle": "656328",
    "window_title": "Win Monitor - 数据泄露行为监控系统 - Google Chrome",
    "window_class": "Chrome_WidgetWin_1"
  },
  "user_info": {
    "username": "zbn20",
    "hostname": "zbn"
  },
  "disk_info": {
    "drive_letter": "",
    "disk_type": ""
  },
  "app_name": "Chrome",
  "extra": {
    "raw_operation": "app",
    "category": "浏览器",
    "source": "window_monitor",
    "risk_level": "高",
    "relative_timestamp": 0.672
  }
}
```

### 示例 3: 文件对话框检测 (app_switch, window_class="#32770")

当浏览器弹出文件选择对话框时，会产生窗口类为 `#32770` 的 app_switch 事件：

```json
{
  "timestamp": "2026-02-11T12:27:33.148",
  "event_type": "app_switch",
  "file_path": "",
  "file_name": "",
  "file_size": 0,
  "file_extension": "",
  "process_info": {
    "pid": "21424",
    "process_name": "chrome.exe",
    "process_path": "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe",
    "cmdline": ""
  },
  "window_info": {
    "window_handle": "2098228",
    "window_title": "打开",
    "window_class": "#32770"
  },
  "user_info": {
    "username": "zbn20",
    "hostname": "zbn"
  },
  "disk_info": {
    "drive_letter": "",
    "disk_type": ""
  },
  "app_name": "Chrome",
  "extra": {
    "raw_operation": "app",
    "category": "浏览器",
    "source": "window_monitor",
    "risk_level": "高",
    "relative_timestamp": 3.687
  }
}
```

### 示例 4: 浏览器文件访问事件 (browser_file_access / ETW)

由 C++ETW 组件捕获的浏览器进程文件访问，`event_type` 为 `created`，`source` 为 `etw_monitor`：

```json
{
  "timestamp": "2026-02-11T12:27:36.000",
  "event_type": "created",
  "file_path": "D:\\code\\DLP\\win_monitor\\test\\AAA公司服务合作合同.docx",
  "file_name": "AAA公司服务合作合同.docx",
  "file_size": 0,
  "file_extension": ".docx",
  "process_info": {
    "pid": "11488",
    "process_name": "chrome.exe",
    "process_path": "",
    "cmdline": ""
  },
  "window_info": {
    "window_handle": "",
    "window_title": "",
    "window_class": ""
  },
  "user_info": {
    "username": "zbn20",
    "hostname": "ZBN"
  },
  "disk_info": {
    "drive_letter": "D:",
    "disk_type": "Fixed"
  },
  "app_name": "Chrome",
  "extra": {
    "raw_operation": "browser_file_access",
    "category": "浏览器文件访问",
    "source": "etw_monitor"
  }
}
```

---

## 字段说明

### 基础字段

| 字段 | 类型 | 必填 | 说明 |
|------|------|:----:|------|
| `timestamp` | string | ✅ | ISO8601 格式时间戳，精确到毫秒<br>格式: `YYYY-MM-DDTHH:MM:SS.mmm` |
| `event_type` | string | ✅ | 事件类型，见下方枚举值 |
| `file_path` | string | ✅ | 文件完整路径（无文件操作时为空字符串） |
| `file_name` | string | ✅ | 文件名（含扩展名） |
| `file_size` | int | ❌ | 文件大小（字节），未知则为 `0` |
| `file_extension` | string | ❌ | 文件扩展名（含点号，如 `.docx`） |
| `app_name` | string | ❌ | 规范化的应用名称（如 `Chrome`, `Edge`, `QQ`） |

### event_type 枚举值

| 值 | 说明 | 来源 (source) |
|----|------|------|
| `clipboard_text` | 剪贴板文本复制 | `clipboard_monitor` |
| `clipboard_image` | 剪贴板图片截图 | `clipboard_monitor` |
| `app_switch` | 切换到应用窗口 | `window_monitor` |
| `website_visit` | 访问黑名单网站 | `window_monitor` |
| `created` | 文件创建 / 浏览器文件访问 | `watchdog` / `etw_monitor` |
| `modified` | 文件修改 | `watchdog` |
| `deleted` | 文件删除 | `watchdog` |
| `renamed` | 文件重命名 | `watchdog` |
| `opened` | 文件被打开/读取 | `fs_usage` (Mac) |
| `file_selected` | 用户通过文件对话框选择文件 | `unified_log` (Mac) |
| `upload_detected` | 检测到上传行为（启发式） | 上传检测器 |
| `manual_note` | 人工补充标注 | `manual` |

---

### process_info 对象

| 字段 | 类型 | 说明 |
|------|------|------|
| `pid` | string | 进程 ID |
| `process_name` | string | 进程名称（如 `chrome.exe`, `QQ`） |
| `process_path` | string | 进程执行文件路径 |
| `cmdline` | string | 命令行参数（可选） |

---

### window_info 对象

| 字段 | 类型 | 说明 |
|------|------|------|
| `window_handle` | string | 窗口句柄 |
| `window_title` | string | 窗口标题（如 `"打开"` 表示文件对话框） |
| `window_class` | string | 窗口类名（`#32770` = Win32 标准对话框） |

---

### user_info 对象

| 字段 | 类型 | 说明 |
|------|------|------|
| `username` | string | 当前登录用户名 |
| `hostname` | string | 计算机主机名 |

---

### disk_info 对象

| 字段 | 类型 | 说明 |
|------|------|------|
| `drive_letter` | string | 驱动器盘符（Mac: `/`，Win: `C:`, `D:` 等） |
| `disk_type` | string | 磁盘类型（`Fixed`, `Removable`, `SSD/HDD`） |

---

### 剪贴板专用字段（clipboard_text / clipboard_image）

| 字段 | 类型 | 说明 |
|------|------|------|
| `content_preview` | string | 剪贴板文本内容预览 |
| `content_hash` | string | 内容哈希（用于去重） |

---

### extra 对象

存放扩展元数据。

| 字段 | 类型 | 说明 |
|------|------|------|
| `raw_operation` | string | 原始操作类型（如 `app`, `clipboard_text`, `browser_file_access`） |
| `category` | string | 应用分类（如 `"浏览器"`, `"即时通讯"`, `"浏览器文件访问"`） |
| `source` | string | 事件来源，见下表 |
| `risk_level` | string | 风险等级（可选）：`高`, `中`, `低` |
| `relative_timestamp` | float | 相对于会话开始的秒数（可选） |
| `note` | string | 人工标注的说明文字（仅 `manual_note` 类型） |

### source 来源枚举

| 值 | 平台 | 说明 |
|----|------|------|
| `window_monitor` | Mac/Win | 窗口监控（app_switch, website_visit） |
| `clipboard_monitor` | Mac/Win | 剪贴板监控 |
| `watchdog` | Win | Python watchdog 文件系统监控 |
| `etw_monitor` | Win | C++ETW 浏览器文件访问捕获 |
| `fs_usage` | Mac | macOS fs_usage 文件操作监控 |
| `fsevents_ipc` | Mac | macOS FSEvents 文件系统事件 |
| `unified_log` | Mac | macOS Unified Log 文件选择检测 |
| `manual` | 通用 | 人工补充标注 |

---

## 敏感文件关键字

以下关键字用于自动标记敏感文件（触发 `upload_detection`）：

```
合同, 机密, 密码, password, secret, private,
财务, 工资, 薪资, 银行, 账号, 证件,
身份证, 护照, 驾照, 简历, resume
```
