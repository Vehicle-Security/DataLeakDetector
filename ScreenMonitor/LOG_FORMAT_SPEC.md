# 统一日志格式说明 (Unified Log Format Specification)

本文档定义了 `Mac_monitor` 和 `win_monitor` 的统一日志输出格式。

---

## 日志文件格式

使用 **JSON Lines** 格式：每行一个独立的 JSON 对象，便于流式处理和追加写入。

```
{"timestamp":"2026-01-18T21:24:15.345","event_type":"opened",...}
{"timestamp":"2026-01-18T21:24:16.123","event_type":"created",...}
{"timestamp":"2026-01-18T21:24:17.456","event_type":"renamed",...}
```

---

## 完整示例

### 示例 1: 文件打开事件 (opened)

```json
{
  "timestamp": "2026-01-18T21:24:37.243",
  "event_type": "opened",
  "file_path": "/Users/qwer/Downloads/test/AA公司合同.docx",
  "file_name": "AA公司合同.docx",
  "file_size": 16011,
  "file_extension": ".docx",
  "process_info": {
    "pid": "130338",
    "process_name": "QQ",
    "process_path": "/Applications/QQ.app",
    "cmdline": ""
  },
  "window_info": {
    "window_handle": "",
    "window_title": "QQ",
    "window_class": ""
  },
  "user_info": {
    "username": "qwer",
    "hostname": "MacBook-Pro"
  },
  "disk_info": {
    "drive_letter": "/",
    "disk_type": "SSD/HDD"
  },
  "app_name": "QQ",
  "upload_detection": {
    "is_upload": true,
    "app_name": "QQ",
    "upload_type": "File Access",
    "original_file": "/Users/qwer/Downloads/test/AA公司合同.docx",
    "temp_directory": ""
  },
  "extra": {
    "raw_operation": "opened",
    "category": "",
    "source": "fs_usage"
  }
}
```

### 示例 2: 文件重命名事件 (renamed)

```json
{
  "timestamp": "2026-01-18T21:24:15.582",
  "event_type": "renamed",
  "file_path": "/Users/qwer/Documents/report_v1.pdf",
  "file_name": "report_v1.pdf",
  "file_size": 524288,
  "file_extension": ".pdf",
  "process_info": {
    "pid": "",
    "process_name": "",
    "process_path": "",
    "cmdline": ""
  },
  "window_info": {
    "window_handle": "",
    "window_title": "",
    "window_class": ""
  },
  "user_info": {
    "username": "qwer",
    "hostname": "MacBook-Pro"
  },
  "disk_info": {
    "drive_letter": "/",
    "disk_type": "SSD/HDD"
  },
  "app_name": "",
  "extra": {
    "raw_operation": "renamed",
    "category": "",
    "source": "fsevents_ipc"
  }
}
```

### 示例 3: 应用切换事件 (app_switch)

```json
{
  "timestamp": "2026-01-18T21:30:00.123",
  "event_type": "app_switch",
  "file_path": "",
  "file_name": "",
  "file_size": 0,
  "file_extension": "",
  "process_info": {
    "pid": "12345",
    "process_name": "WeChat",
    "process_path": "/Applications/WeChat.app",
    "cmdline": ""
  },
  "window_info": {
    "window_handle": "0x12345",
    "window_title": "微信",
    "window_class": "WeChatMainWnd"
  },
  "user_info": {
    "username": "qwer",
    "hostname": "MacBook-Pro"
  },
  "disk_info": {
    "drive_letter": "",
    "disk_type": ""
  },
  "app_name": "微信",
  "extra": {
    "raw_operation": "app",
    "category": "即时通讯",
    "source": "window_monitor",
    "risk_level": "高"
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
| `file_path` | string | ✅ | 文件完整路径 |
| `file_name` | string | ✅ | 文件名（含扩展名） |
| `file_size` | int | ❌ | 文件大小（字节），未知则为 `0` |
| `file_extension` | string | ❌ | 文件扩展名（含点号，如 `.docx`） |
| `app_name` | string | ❌ | 规范化的应用名称 |

### event_type 枚举值

| 值 | 说明 | 来源 |
|----|------|------|
| `opened` | 文件被打开/读取 | fs_usage (Mac), watchdog (Win) |
| `created` | 文件创建 | FSEvents (Mac), watchdog (Win) |
| `modified` | 文件修改 | FSEvents (Mac), watchdog (Win) |
| `deleted` | 文件删除 | FSEvents (Mac), watchdog (Win) |
| `renamed` | 文件重命名 | FSEvents (Mac), watchdog (Win) |
| `file_selected` | 用户通过文件对话框选择文件 | Unified Log (Mac) |
| `upload_detected` | 检测到上传行为（启发式） | 上传检测器 |
| `app_switch` | 切换到黑名单应用 | 窗口监控 |
| `website_visit` | 访问黑名单网站 | 窗口监控 |

---

### process_info 对象

| 字段 | 类型 | 说明 |
|------|------|------|
| `pid` | string | 进程 ID |
| `process_name` | string | 进程名称（如 `QQ`, `Chrome`） |
| `process_path` | string | 进程执行文件路径 |
| `cmdline` | string | 命令行参数（可选） |

---

### window_info 对象

| 字段 | 类型 | 说明 |
|------|------|------|
| `window_handle` | string | 窗口句柄 |
| `window_title` | string | 窗口标题 |
| `window_class` | string | 窗口类名 |

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
| `drive_letter` | string | 驱动器盘符（Mac: `/`，Win: `C:`） |
| `disk_type` | string | 磁盘类型（`SSD/HDD`, `Fixed`, 等） |

---

### upload_detection 对象（可选）

仅在检测到敏感文件操作时存在。

| 字段 | 类型 | 说明 |
|------|------|------|
| `is_upload` | bool | 是否检测为上传操作 |
| `app_name` | string | 操作应用名称 |
| `upload_type` | string | 上传类型：`File Access`, `File Dialog Selection`, `Sensitive Access` |
| `original_file` | string | 原始文件路径 |
| `temp_directory` | string | 临时目录路径（如有） |

---

### extra 对象

存放扩展元数据。

| 字段 | 类型 | 说明 |
|------|------|------|
| `raw_operation` | string | 原始操作类型 |
| `category` | string | 应用分类（如"即时通讯"、"云存储"） |
| `source` | string | 事件来源：`fsevents_ipc`, `fs_usage`, `watchdog_fs_monitor`, `window_monitor` |
| `risk_level` | string | 风险等级（可选）：`高`, `中`, `低` |

---

## 敏感文件关键字

以下关键字用于自动标记敏感文件（触发 `upload_detection`）：

```
合同, 机密, 密码, password, secret, private,
财务, 工资, 薪资, 银行, 账号, 证件,
身份证, 护照, 驾照, 简历, resume
```
