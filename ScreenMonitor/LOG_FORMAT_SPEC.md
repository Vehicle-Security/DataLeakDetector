# DataLeakDetector 日志格式规范 v1.0.0

## 概述

本规范定义了 Mac_monitor 和 win_monitor 的统一日志格式，确保两个平台产生的日志结构完全一致。

---

## event_type 统一定义

### 文件操作类事件 (FILE_OPS)

| event_type | 说明 | Mac | Windows |
|---|---|:---:|:---:|
| `opened` | 文件被打开/读取 | ✅ | ✅ |
| `created` | 文件创建 | ✅ | ✅ |
| `modified` | 文件修改 | ✅ | ✅ |
| `deleted` | 文件删除 | ✅ | ✅ |
| `renamed` | 文件重命名 | ✅ | ✅ |

### 用户行为类事件 (USER_ACTION)

| event_type | 说明 | Mac | Windows |
|---|---|:---:|:---:|
| `file_selected` | 用户通过文件对话框选择文件 | ✅ | ✅ |
| `upload_detected` | 检测到上传行为（启发式） | ✅ | ✅ |

### 上下文切换类事件 (CONTEXT_SWITCH)

| event_type | 说明 | Mac | Windows |
|---|---|:---:|:---:|
| `app_switch` | 切换到黑名单桌面应用 | ✅ | ✅ |
| `website_visit` | 访问黑名单网站 | ✅ | ✅ |

### 系统事件 (SYSTEM)

| event_type | 说明 | Mac | Windows |
|---|---|:---:|:---:|
| `unified_log_event` | macOS Unified Log 事件 | ✅ | N/A |

---

## 日志条目结构 (LogEntry)

```json
{
  "timestamp": "2026-01-07T12:00:00.000",
  "event_type": "opened|created|modified|deleted|renamed|file_selected|upload_detected|app_switch|website_visit",
  
  "file_path": "/Users/xxx/Documents/机密文件.docx",
  "file_name": "机密文件.docx",
  "file_size": 1024,
  "file_extension": ".docx",
  
  "process_info": {
    "pid": "12345",
    "process_name": "WeChat",
    "process_path": "/Applications/WeChat.app",
    "cmdline": ""
  },
  
  "window_info": {
    "window_handle": "",
    "window_title": "微信",
    "window_class": ""
  },
  
  "user_info": {
    "username": "admin",
    "hostname": "MacBook-Pro"
  },
  
  "disk_info": {
    "drive_letter": "/" ,
    "disk_type": "SSD"
  },
  
  "app_name": "微信",
  
  "upload_detection": {
    "is_upload": true,
    "app_name": "微信", 
    "upload_type": "File Dialog Selection|Sensitive Access|Drag and Drop",
    "original_file": "/path/to/original",
    "temp_directory": ""
  },
  
  "extra": {
    "raw_operation": "open",
    "category": "即时通讯",
    "detection_method": "file_dialog|browser_file_monitor|fs_usage"
  }
}
```

---

## 需要修改的文件

### Mac_monitor (Go)

1. **session_manager.go** - LogEntry 结构已定义，格式正确
2. **file_monitor.go** - 已实现 `opened`, `created`, `modified`, `deleted`, `renamed`, `file_selected`
3. **window_monitor.go** - 已实现 `app_switch`, `website_visit`
4. **需添加**: `upload_detected` 事件支持

### win_monitor (Python)

1. **core/browser_file_monitor.py** - 需将 `modified` 改为根据实际操作区分
2. **core/log_manager.py** - 需添加 `opened`, `created`, `deleted`, `renamed` 的处理
3. **core/logger.py** - LogEntry 结构需与 Mac 保持一致
4. **core/upload_detector.py** - `upload_detected` 已实现
5. **core/file_dialog_detector.py** - `file_selected` 已实现

---

## 字段说明

| 字段 | 类型 | 必填 | 说明 |
|---|---|:---:|---|
| timestamp | string | ✅ | ISO8601 格式，精确到毫秒 |
| event_type | string | ✅ | 事件类型，见上方定义 |
| file_path | string | ✅ | 完整文件路径 |
| file_name | string | ✅ | 文件名 |
| file_size | int64 | ❌ | 文件大小（字节），未知则为 0 |
| file_extension | string | ❌ | 文件扩展名（含点号） |
| process_info | object | ✅ | 进程信息 |
| window_info | object | ❌ | 窗口信息 |
| user_info | object | ✅ | 用户信息 |
| disk_info | object | ❌ | 磁盘信息 |
| app_name | string | ❌ | 规范化的应用名称 |
| upload_detection | object | ❌ | 上传检测信息，仅在检测到上传时存在 |
| extra | object | ❌ | 扩展字段 |

---

## event_type 枚举值

```go
const (
    // 文件操作
    EventTypeOpened   = "opened"
    EventTypeCreated  = "created"
    EventTypeModified = "modified"
    EventTypeDeleted  = "deleted"
    EventTypeRenamed  = "renamed"
    
    // 用户行为
    EventTypeFileSelected   = "file_selected"
    EventTypeUploadDetected = "upload_detected"
    
    // 上下文切换
    EventTypeAppSwitch    = "app_switch"
    EventTypeWebsiteVisit = "website_visit"
    
    // 系统
    EventTypeUnifiedLog = "unified_log_event"
)
```

```python
# Python 枚举
class EventType:
    # 文件操作
    OPENED = "opened"
    CREATED = "created"
    MODIFIED = "modified"
    DELETED = "deleted"
    RENAMED = "renamed"
    
    # 用户行为
    FILE_SELECTED = "file_selected"
    UPLOAD_DETECTED = "upload_detected"
    
    # 上下文切换
    APP_SWITCH = "app_switch"
    WEBSITE_VISIT = "website_visit"
```

---

## 版本历史

| 版本 | 日期 | 变更 |
|---|---|---|
| 1.0.0 | 2026-01-07 | 初始版本，统一 Mac 和 Windows 日志格式 |
