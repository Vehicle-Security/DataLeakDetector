#!/bin/bash

# macOS 录屏和 Log 监控系统 - 启动脚本

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🚨 macOS 数据泄露行为监控系统"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📊 监控功能: 录屏 | 文件操作 | 窗口切换 | 剪贴板"

# 检查 ffmpeg 是否安装
if ! command -v ffmpeg &> /dev/null; then
    echo "❌ 错误: ffmpeg 未安装"
    echo "💡 请运行: brew install ffmpeg"
    exit 1
fi

echo "✅ ffmpeg 已安装"

# 检查是否使用 sudo 运行
if [ "$EUID" -ne 0 ]; then
   echo "🔐 需要 root 权限以启用完整的行为监控功能..."
   exec sudo "$0" "$@"
   exit 0
fi

# 获取脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 编译后端（如果需要）
# 总是重新编译以确保最新更改生效
echo "📦 正在编译后端服务..."
cd server
go build -o ../mac_monitor_server .
cd ..

# 检查前端依赖
if [ ! -d "./frontend/node_modules" ]; then
    echo "📦 正在安装前端依赖..."
    cd frontend
    npm install
    cd ..
fi

# 创建录制目录
mkdir -p ./recordings
chmod 777 ./recordings

echo ""
echo "🧹 清理被占用的端口..."
# 清理后端端口 8081
PID_8081=$(lsof -ti:8081)
if [ ! -z "$PID_8081" ]; then
    echo "Killing process on port 8081: $PID_8081"
    kill -9 $PID_8081 2>/dev/null
fi

# 清理前端端口 3000
PID_3000=$(lsof -ti:3000)
if [ ! -z "$PID_3000" ]; then
    echo "Killing process on port 3000: $PID_3000"
    kill -9 $PID_3000 2>/dev/null
fi

# 清理遗留的监控进程
pkill -f "mac_monitor_server" 2>/dev/null
pkill -f "fs_usage" 2>/dev/null

echo "✅ 端口清理完成"
echo ""

echo ""
echo "🚀 启动服务..."
echo ""

# 启动后端服务
echo "📡 启动后端服务 (端口 8081)..."
./mac_monitor_server &
BACKEND_PID=$!

# 等待后端启动
sleep 2

# 启动前端开发服务器
echo "🌐 启动前端服务..."
cd frontend
# 使用 silent 模式减少噪音
npm run dev -- --port 3000 &
FRONTEND_PID=$!
cd ..

# 等待前端启动
sleep 3

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ 系统已启动！"
echo ""
echo "🌐 前端界面: http://localhost:3000"
echo "📡 后端 API: http://localhost:8081"
echo "📁 录制文件: $SCRIPT_DIR/recordings"
echo ""
echo "💡 按 Ctrl+C 停止所有服务"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# 捕获退出信号
cleanup() {
    echo ""
    echo "🛑 正在停止服务..."
    kill $BACKEND_PID 2>/dev/null
    kill $FRONTEND_PID 2>/dev/null
    echo "👋 再见！"
    exit 0
}

trap cleanup SIGINT SIGTERM

# 等待进程
wait
