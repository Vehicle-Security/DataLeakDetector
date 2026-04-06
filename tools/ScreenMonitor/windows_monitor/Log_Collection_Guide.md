# Windows Monitor 数据泄露日志采集指南

本文档旨在指导如何使用 `winows_monitor` (ScreenMonitor) 进行数据泄露行为的日志采集，特别针对构建数据集时的操作流程。文档详细说明了哪些泄露行为可以被自动捕获，哪些需要人工补充说明。

---

## 1. 安装与启动 (Installation & Startup)

### 1.1 环境要求
| 依赖项 | 是否必须 | 说明 |
| :--- | :---: | :--- |
| **Python 3.8+** | ✅ 必须 | 运行主程序 |
| **pip 依赖** | ✅ 必须 | 见 `requirements.txt`，安装命令见下方 |
| **管理员权限** | ✅ 必须 | ETW 文件监控需要管理员权限 |
| **ffmpeg** | ⚠️ 推荐 | 屏幕录制所需，不安装则无法录制视频（其他功能正常） |
| **Visual Studio** | ❌ 不需要 | C++ 组件已预编译为 `.exe`，无需编译 |

### 1.2 安装步骤

**Step 1: 克隆仓库**
```bash
git clone <仓库地址>
cd DataLeakDetector/ScreenMonitor/winows_monitor
```

**Step 2: 安装 Python 依赖**
```bash
pip install -r requirements.txt
```

**Step 3: 安装 ffmpeg (用于屏幕录制)**
> 如果不需要屏幕录制功能，可跳过此步骤。

1. 下载 ffmpeg: https://www.gyan.dev/ffmpeg/builds/ (选择 `ffmpeg-release-essentials.zip`)
2. 解压后将 `bin` 目录添加到系统 `PATH` 环境变量
3. 验证安装: 打开 CMD 输入 `ffmpeg -version`

**Step 4: 验证 C++ 组件**
确认以下文件存在（已随代码仓库提供，无需编译）：
- `core/C++ETW/bin/EtwMonitorV2.exe` (或 `EtwMonitor.exe`)

如果 `core/C++ETW/bin/` 目录为空，说明 Git 提交时未包含二进制文件，请联系项目负责人获取。

### 1.3 启动监控服务
> **⚠️ 重要：必须以管理员权限运行！**

```powershell
# 右键 PowerShell -> 以管理员身份运行
cd <你的项目路径>\DataLeakDetector\ScreenMonitor\winows_monitor
python web_server.py
```

### 1.4 控制界面
启动后访问 Web UI 进行操作：
- 地址: `http://localhost:5000`
- 点击 **"开始监控"** 按钮后会立即创建会话并开始持续录制
- 完成操作后，点击 **"停止监控"**

---

## 2. 配置说明 (Configuration)

在采集前，请确保 `config.yaml` 已包含目标测试应用或网站。
- **blacklist_apps**: 添加需要测试的黑名单应用（如 QQ, 微信等），用于窗口/应用风险分类，不再决定是否开始录制
- **blacklist_websites**: 添加需要测试的黑名单网站（如 GitHub, 网盘等），用于网站风险分类，不再决定是否开始录制
- **sensitive_keywords**: 确保测试用的文件名包含敏感关键词（如"合同", "机密"），以便触发高危告警

---

## 3. 数据集采集指南 (Dataset Collection)

**图例**:
- ✅ **自动采集**: 系统会自动记录日志 (Video, Window Title, File Event, Upload Alert)
- ⚠️ **需人工补充**: 系统记录部分信息，但缺乏上下文或特定动作的标记
- ❌ **无法采集**: 系统无法感知，**必须**人工记录

### 3.1 不同黑白名单应用
| # | 场景描述 | 采集状态 | 说明 |
| :---: | :--- | :---: | :--- |
| 1 | **单黑名单访问** | ✅ | 窗口标题、进程名、屏幕录制均会自动记录 |
| 2 | **多黑名单串联访问** | ✅ | 窗口切换事件 (`app_switch`) 自动记录切换轨迹 |
| 3 | **黑名单混入白名单** | ✅ | 完整记录操作流 |
| 4 | **后台运行黑名单 (隐蔽传输)** | ⚠️ | 无界面交互时仅靠网络传输无法被捕获。**人工记录**：`"xx:xx 后台启动传输"` |
| 5 | **伪装成合法应用 (跳转)** | ✅ | 屏幕录制捕获跳转过程，窗口标题变化被记录 |
| 6 | **系统分享接口** | ✅ | 文件读取事件 (`opened`) 被捕获 |

### 3.2 不同泄露模式 (文件混淆/上传)
| 场景类型 | 场景描述 | 采集状态 | 说明 |
| :--- | :--- | :---: | :--- |
| **文件混淆** | 重命名文件 | ✅ | `renamed` 事件被文件监控捕获 |
| | 压缩文件 (Zip/加密) | ✅ | `created`/`modified` 事件捕获压缩包生成 |
| | 导出文件 (PDF/Excel) | ✅ | `created` 事件捕获新文件生成 |
| | 文件拆分 (分段/切图) | ✅ | `created` 事件捕获生成的多个小文件 |
| **文件上传** | 本地分享 (Chat软件) | ✅ | ETW/Watchdog 捕获文件读取 |
| | Web 上传 (浏览器) | ✅ | ETW 捕获浏览器文件访问 (`Browser File Access`) |
| | 云端上传 | ✅ | 客户端读取文件同步也会产生文件事件 |

### 3.3 未知泄露 (物理/外设/其他)
| # | 场景描述 | 采集状态 | 补全操作建议 |
| :---: | :--- | :---: | :--- |
| 1 | **剪贴板泄露** (复制粘贴) | ✅ | `ClipboardMonitor` 记录文本内容和图片截图 |
| 2 | **蓝牙传输** | ⚠️ | 文件读取可被记录，但"蓝牙连接"无记录。**人工记录**：`"xx:xx 连接蓝牙设备"` |
| 3 | **VPN / 代理泄露** | ⚠️ | 访问黑名单网站会被记录，但 VPN 状态不会。**人工记录**：`"此时开启了VPN"` |
| 4 | **U盘 / 外设导出** | ✅ | 向 U 盘写入文件产生 `create`/`write` 事件 (路径如 `E:\...`) |
| 5 | **像素点编码 / 隐写术** | ✅* | 文件创建被记录，但语义无法识别。*日志中只体现为普通文件操作* |
| 6a | 屏幕录制 (软件) | ✅ | 录屏软件进程被记录 |
| 6b | **截屏/拍照 (手机/相机)** | ❌ | 系统无法感知物理拍照。**必须人工记录**：`"xx:xx 使用手机拍照"` |
| 7 | **输入法 / 语音输入** | ❌ | 无键盘/语音录制。**必须人工记录输入的敏感内容** |
| 8 | **OCR / 扫描仪** | ⚠️ | 外接设备生成文件→✅。手机拍照识别→❌ 人工 |
| 9 | **Base64 编码** | ✅ | 剪贴板操作和文件保存均会记录 |
| 10 | **临时文件 / 草稿箱** | ✅ | 浏览器缓存写入或剪贴板操作会被记录 |

---

## 4. 补充记录模板 (Manual Log Template)

对于标记为 **⚠️** 或 **❌** 的场景，建议采集人员维护 `session_notes.txt`：

```text
Session ID: 20260210_143000
----------------------------------------
[时间戳] [操作类型] [详情]
14:32:15 [VPN] 开启 VPN 代理连接到 US 节点
14:35:00 [物理] 使用手机拍摄屏幕上的合同内容
14:38:20 [输入] 通过语音输入法读出身份证号
14:40:00 [后台] 此时黑名单 App 在后台静默运行
```

---

## 5. 部署与分发说明 (Deployment)

### 5.1 C++ 组件 (EtwMonitor)
- `EtwMonitor.exe` 是预编译的，**使用者不需要 Visual Studio，也不需要重新编译**
- 确保 `core/C++ETW/bin/` 目录已随代码提交到 GitHub（检查 `.gitignore` 不要排除此目录）
- 如果目标电脑运行时提示 `MSVCP140.dll` 缺失，安装 [VC++ Redistributable](https://learn.microsoft.com/cpp/windows/latest-supported-vc-redist)

### 5.2 常见问题
| 问题 | 解答 |
| :--- | :--- |
| 没有 VS 能用吗？ | **能用**。C++ 已编译为 `.exe`，放在 `core/C++ETW/bin/` 目录下 |
| 没有 ffmpeg 能用吗？ | **能用**，但没有屏幕录制视频。日志采集、窗口监控、剪贴板监控不受影响 |
| 浏览器上传检测不到？ | 检查是否以管理员权限运行；检查 `core/C++ETW/bin/EtwMonitorV2.exe` 是否存在 |
| `pip install` 报错？ | 如果是 `pywintrace` 安装失败可忽略（可选依赖），不影响核心功能 |

---

## 6. 会话输出说明 (Session Output)

每次点击 **"停止监控"** 后，系统会在 `recordings/` 下生成一个会话目录：

```
recordings/session_20260211_122729/
├── INDEX.md                                    # 会话索引 (自动生成)
├── logs/
│   ├── logs.json                               # ① 完整原始日志 (所有事件，通常 1~2MB)
│   ├── etw_session_20260211_122730.json         # ② ETW 浏览器文件访问日志 (C++ETW 捕获)
│   └── keyevents.json                          # ③ ★ 关键事件 (最终使用的数据集)
└── video/
    └── recording_20260211_122729.mp4            # ④ 屏幕录像 (需 ffmpeg)
```

### 各文件详细说明

| 文件 | 用途 | 说明 |
| :--- | :--- | :--- |
| **① `logs.json`** | 完整原始日志 | 包含窗口切换、剪贴板、文件创建/修改等所有事件。数据量大（通常上千条），包含大量系统噪声，**一般不直接使用**。该文件保留原始窗口事件，允许 `app_switch` / `website_visit` 的 `file_path=""` |
| **② `etw_session_*.json`** | ETW 浏览器文件访问 | 由 C++ETW 组件 (`EtwMonitorV2.exe`) 捕获的浏览器进程文件访问记录。该文件只在录制结束前作为中间产物存在，后续会被合并进 `logs.json` / `keyevents.json` 并清理掉 |
| **③ `keyevents.json`** | ★ **最终关键事件** | 从 `logs.json` 和 ETW 中间日志中提取、过滤并归一化后的关键事件。**这是最终用于数据集标注的文件**。包含：窗口切换 (`app_switch`)、剪贴板操作 (`clipboard_text`)、浏览器文件访问（`event_type=created` 且 `extra.raw_operation=browser_file_access`）等。窗口事件只有在能绑定到精确文件路径时才会保留 |
| **④ `recording_*.mp4`** | 屏幕录像 | 监控期间的屏幕录制视频，需要安装 ffmpeg |

### `keyevents.json` 字段约定

- `INDEX.md` 中的 `**Recording Time**:` 字段名和格式固定不变
- `timestamp` 固定表示事件实际发生时间
- `file_path` 固定表示事件涉及的完整文件路径
- `app_switch` / `website_visit` 若存在，`file_path` 必须是完整精确路径；无法精确绑定时该事件不会出现在 `keyevents.json`
- `clipboard_*`、`manual_note` 等真正非文件事件允许 `file_path=""`
- `process_info.process_path` 固定表示应用程序路径，不会写入 `file_path`
- `file_path` 不会回填文件名，也不会回填 `process_info.process_path`
- 会话结束后的 `logs/` 目录只保留 `logs.json` 和 `keyevents.json`

### 人工补充标注

**如果需要补充系统无法自动捕获的事件（如手机拍照、VPN 状态等），直接在 `keyevents.json` 中追加条目：**

```json
{
  "timestamp": "2026-02-11T12:30:00.000",
  "event_type": "manual_note",
  "file_path": "",
  "file_name": "",
  "file_size": 0,
  "file_extension": "",
  "process_info": { "pid": "", "process_name": "", "process_path": "" },
  "window_info": { "window_handle": "", "window_title": "", "window_class": "" },
  "user_info": { "username": "zbn20", "hostname": "zbn" },
  "disk_info": { "drive_letter": "", "disk_type": "" },
  "app_name": "",
  "extra": {
    "raw_operation": "manual",
    "category": "人工标注",
    "source": "manual",
    "note": "使用手机拍摄屏幕上的合同内容"
  }
}
```

> **提示**: 补充时保持 `timestamp` 时间戳与实际操作时间一致，方便与视频对照。
