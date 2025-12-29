# macOS 数据泄露行为监控系统

<div align="center">

**实时监控和记录 macOS 系统的屏幕活动与文件操作行为**

[![License](https://img.shields.io/badge/license-MIT-blue.svg)]()
[![macOS](https://img.shields.io/badge/platform-macOS-lightgrey.svg)]()
[![Go](https://img.shields.io/badge/Go-1.19+-00ADD8.svg)]()
[![React](https://img.shields.io/badge/React-18+-61DAFB.svg)]()

</div>

## 📖 简介

macOS 数据泄露行为监控系统是一个综合性的行为审计平台,提供:

- 🎥 **屏幕录制**: 实时录制屏幕活动,捕获用户操作全过程
- 📁 **文件操作监控**: 追踪文件创建、修改、删除和访问操作
- 🔍 **关键事件检测**: 自动识别和标记潜在的敏感数据泄露行为
- 📊 **可视化审计**: 直观的 Web 界面查看录制视频和操作日志
- 🏷️ **智能摘要**: 自动生成会话摘要和风险事件统计

## ✨ 主要功能

### 监控能力

- **屏幕录制**: 使用 FFmpeg 进行高质量屏幕捕获
- **文件系统监控**: 基于 `fs_usage` 的实时文件操作追踪
- **系统日志监控**: 基于 macOS Unified Logs 的高级事件捕获
  - 文件选择对话框监控（浏览器上传意图检测）
  - AirDrop/系统分享行为检测
  - Rust 原生高性能实现
- **进程识别**: 精确定位执行操作的应用程序和进程
- **事件分类**: 自动分类文件操作类型 (创建/修改/删除/重命名)

### Web 界面

- **会话管理**: 浏览所有录制会话,支持日期筛选和分页
- **同步回放**: 视频播放与操作日志时间轴同步显示
- **日志搜索**: 实时搜索过滤文件路径、进程名称和操作类型
- **风险标注**: 高亮显示潜在的敏感文件访问操作

## 🚀 快速开始

### 系统要求

- macOS 10.15 或更高版本
- Go 1.19+
- Rust 1.70+ (用于 Unified Logs 监控)
- Node.js 16+
- FFmpeg

### 安装

```bash
# 1. 安装 FFmpeg
brew install ffmpeg

# 2. 进入项目目录
cd Mac_monitor

# 3. 编译 Rust 监控代理
cd macos-UnifiedLogs/examples/monitor
cargo build --release
cd ../../..

# 4. 安装前端依赖
cd frontend && npm install && cd ..

# 5. 配置系统权限
# 在 系统设置 > 隐私与安全性 中授予以下权限:
# - 屏幕录制
# - 辅助功能
# - 完全磁盘访问
```

### 启动

```bash
# 使用启动脚本 (推荐)
chmod +x start.sh
./start.sh

# 或手动启动
# 终端 1: 启动后端
cd server && go build -o ../monitor_server . && cd ..
sudo ./monitor_server

# 终端 2: 启动前端
cd frontend && npm run dev
```

### 访问

打开浏览器访问: **http://localhost:5173**

## 📸 界面预览

### 会话列表
- 查看所有录制会话
- 筛选和搜索功能
- 会话状态和风险事件统计

### 会话详情
- 左侧: 录制视频播放
- 右侧: 时间轴同步的操作日志
- 支持搜索和过滤

## 🛠️ 技术架构

### 后端
- **语言**: Go 1.19+
- **框架**: 标准库 HTTP 服务器
- **监控**: fs_usage、FFmpeg
- **数据存储**: JSON 文件

### 前端
- **框架**: React 18 + Vite
- **路由**: React Router
- **样式**: Tailwind CSS
- **图标**: Lucide React
- **时间处理**: date-fns

## 📁 项目结构

```
Mac_monitor/
├── server/                 # Go 后端服务
│   ├── main.go            # 主服务器和 API 路由
│   ├── recorder.go        # 屏幕录制控制器
│   ├── file_monitor.go    # 文件系统监控
│   ├── session_manager.go # 会话管理
│   └── config.go          # 配置和数据结构
├── frontend/              # React 前端
│   ├── src/
│   │   ├── pages/        # 页面组件
│   │   │   ├── Home.jsx          # 主控制面板
│   │   │   ├── SessionList.jsx   # 会话列表
│   │   │   └── SessionDetail.jsx # 会话详情
│   │   ├── App.jsx       # 应用主组件
│   │   └── index.css     # 全局样式
│   └── package.json
├── recordings/            # 录制数据存储
│   └── session_*/        # 会话目录
│       ├── video/        # 录制视频
│       ├── logs/         # 操作日志
│       ├── key_events/   # 关键事件
│       └── INDEX.md      # 会话索引
├── start.sh              # 启动脚本
├── DEPLOYMENT.md         # 部署文档
└── README.md            # 项目说明
```

## 📖 使用指南

### 开始录制

1. 访问 Web 界面
2. 点击"开始监控"按钮
3. 执行需要记录的操作
4. 点击"停止监控"

### 查看录制

1. 在会话列表中选择会话
2. 查看录制视频和操作日志
3. 使用搜索功能过滤日志
4. 下载视频或导出日志

## 🔧 配置

### 端口配置

- 后端 API: `server/main.go` 中修改 `port := ":8081"`
- 前端服务: `frontend/vite.config.js` 中修改 server 配置

### 录制质量

在前端界面调整 FPS 设置:
- 低: 10 FPS
- 中: 15 FPS (默认)
- 高: 30 FPS

## 📚 API 文档

### 录制控制

- `GET /api/recording/status` - 获取录制状态
- `POST /api/recording/start` - 开始录制
- `POST /api/recording/stop` - 停止录制

### 会话管理

- `GET /api/sessions` - 获取所有会话
- `GET /api/sessions/{id}` - 获取会话详情
- `GET /api/key-events/{id}` - 获取关键事件
- `GET /api/logs/{id}` - 获取操作日志

## 🔒 安全建议

1. **限制访问**: 仅在可信网络环境中使用
2. **数据加密**: 对敏感录制数据进行加密存储
3. **定期清理**: 及时删除旧的录制数据
4. **权限管理**: 严格控制系统权限授予
5. **合规使用**: 遵守当地数据隐私法规

## 🐛 故障排除

### 屏幕录制失败
- 检查屏幕录制权限
- 验证 FFmpeg 安装: `ffmpeg -version`
- 查看错误日志: `recordings/session_*/video/*.error.log`

### 文件监控无数据
- 确认完全磁盘访问权限
- 使用 sudo 运行后端服务
- 检查日志文件权限

详细故障排除请参阅 [DEPLOYMENT.md](DEPLOYMENT.md)

## 📄 许可证

本项目仅供学习和研究使用。

## 🤝 贡献

欢迎提交问题和改进建议。

---

<div align="center">

**⚠️ 重要提示**: 本系统涉及屏幕录制和文件监控,请确保您的使用符合当地法律法规和公司政策。

</div>
