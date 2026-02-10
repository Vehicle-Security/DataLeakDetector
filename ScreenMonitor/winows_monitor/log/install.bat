@echo off
REM Installation script for File Monitor System
REM Requires Administrator privileges

echo ========================================
echo File Monitor System - Installation
echo ========================================

REM Check for admin privileges
net session >nul 2>&1
if %errorLevel% NEQ 0 (
    echo This script requires Administrator privileges.
    echo Please run as Administrator.
    pause
    exit /b 1
)

echo.
echo Step 1: Installing Minifilter Driver...
echo ----------------------------------------

REM Install the driver
cd /d "%~dp0src\driver"
if not exist "FsFilter.inf" (
    echo Error: FsFilter.inf not found!
    pause
    exit /b 1
)

REM Right-click Install equivalent
rundll32.exe setupapi,InstallHinfSection DefaultInstall 132 .\FsFilter.inf

if %ERRORLEVEL% NEQ 0 (
    echo Driver installation failed!
    pause
    exit /b 1
)

echo Driver installed successfully.

echo.
echo Step 2: Starting Driver...
echo ----------------------------------------

sc start FsFilter
if %ERRORLEVEL% NEQ 0 (
    echo Warning: Driver may already be running or needs manual start
)

echo.
echo Step 3: Installing User-mode Service...
echo ----------------------------------------

cd /d "%~dp0bin"
if not exist "MonitorService.exe" (
    echo Error: MonitorService.exe not found!
    echo Please build the service first using build.bat
    pause
    exit /b 1
)

MonitorService.exe /install
if %ERRORLEVEL% NEQ 0 (
    echo Service installation failed!
    pause
    exit /b 1
)

echo.
echo Step 4: Starting Service...
echo ----------------------------------------

sc start FileMonitorService
if %ERRORLEVEL% NEQ 0 (
    echo Warning: Service may need manual start
)

echo.
echo ========================================
echo Installation Complete!
echo ========================================
echo.
echo Driver: FsFilter
echo Service: FileMonitorService
echo Log Location: C:\Logs\FileMonitor.log
echo.
echo To verify installation:
echo   sc query FsFilter
echo   sc query FileMonitorService
echo.
pause
