@echo off
REM Uninstallation script for File Monitor System
REM Requires Administrator privileges

echo ========================================
echo File Monitor System - Uninstallation
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
echo Step 1: Stopping and Removing Service...
echo ----------------------------------------

sc stop FileMonitorService
sc delete FileMonitorService

echo.
echo Step 2: Stopping and Removing Driver...
echo ----------------------------------------

sc stop FsFilter
sc delete FsFilter

echo.
echo Step 3: Removing Files...
echo ----------------------------------------

cd /d "%~dp0bin"
if exist "MonitorService.exe" (
    MonitorService.exe /uninstall
)

echo.
echo ========================================
echo Uninstallation Complete!
echo ========================================
echo.
echo Note: Log files in C:\Logs\ were not deleted.
echo You can manually delete them if needed.
echo.
pause
