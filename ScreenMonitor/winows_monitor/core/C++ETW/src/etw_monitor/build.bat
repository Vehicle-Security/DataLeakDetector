@echo off
cd /d "%~dp0"
echo Building ETW Monitor...

REM Setup VS environment (Hardcoded for reliability)
if exist "D:\VS2022\VC\Auxiliary\Build\vcvars64.bat" (
    call "D:\VS2022\VC\Auxiliary\Build\vcvars64.bat"
) else (
    echo Warning: D:\VS2022 not found, trying default...
    set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
    for /f "usebackq tokens=*" %%i in (`"%VSWHERE%" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do (
        set "VS_PATH=%%i"
    )
    if exist "%%VS_PATH%%\VC\Auxiliary\Build\vcvars64.bat" (
            call "%%VS_PATH%%\VC\Auxiliary\Build\vcvars64.bat"
    )
)

if not exist "..\..\bin" mkdir "..\..\bin"

cl.exe /EHsc /O2 /DUNICODE /D_UNICODE /Fe:..\..\bin\EtwMonitorV2.exe EtwMonitor.cpp /link advapi32.lib tdh.lib psapi.lib

if %ERRORLEVEL% EQU 0 (
    echo.
    echo Build SUCCESS! Output: ..\..\bin\EtwMonitorV2.exe
) else (
    echo.
    echo Build FAILED.
)
