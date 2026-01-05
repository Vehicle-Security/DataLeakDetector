@echo off
chcp 65001 >nul
echo ========================================
echo 启动 Win Monitor (管理员权限)
echo ========================================
echo.

REM 检查是否以管理员身份运行
net session >nul 2>&1
if %errorLevel% == 0 (
    echo [OK] 以管理员身份运行
) else (
    echo [ERROR] 需要管理员权限！
    echo 请右键此脚本，选择"以管理员身份运行"
    pause
    exit /b 1
)

REM 切换到项目目录
cd /d d:\code\DataLeakDetector\ScreenMonitor\win_monitor
echo 当前目录: %CD%

echo.
echo 初始化 Conda...
REM 尝试常见的conda安装位置
if exist "C:\Users\zbn20\miniconda3\Scripts\activate.bat" (
    call "C:\Users\zbn20\miniconda3\Scripts\activate.bat" win_monitor
) else if exist "C:\ProgramData\miniconda3\Scripts\activate.bat" (
    call "C:\ProgramData\miniconda3\Scripts\activate.bat" win_monitor
) else if exist "D:\anaconda\Scripts\activate.bat" (
    call "D:\anaconda\Scripts\activate.bat" win_monitor
) else if exist "D:\miniconda3\Scripts\activate.bat" (
    call "D:\miniconda3\Scripts\activate.bat" win_monitor
) else (
    echo [ERROR] 找不到 Conda！请手动修改此脚本中的 conda 路径
    echo 您的 conda 安装在哪里？
    pause
    exit /b 1
)

echo.
echo 启动 Web 服务器...
echo 访问地址: http://localhost:5000
echo.
python web_server.py

pause
