@echo off
REM Build script for File Monitor Service (User-mode)
REM Requires Visual Studio C++ Build Tools

echo ========================================
echo Building File Monitor Service
echo ========================================

REM Try to find Visual Studio using vswhere
set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
if not exist "%VSWHERE%" (
    echo Error: vswhere.exe not found. Is Visual Studio installed?
    pause
    exit /b 1
)

for /f "usebackq tokens=*" %%i in (`"%VSWHERE%" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do (
    set "VS_PATH=%%i"
)

if not defined VS_PATH (
    echo Error: Visual Studio with C++ tools not found!
    echo Please install "Desktop development with C++" workload.
    pause
    exit /b 1
)

echo Found Visual Studio at: %VS_PATH%
if exist "%VS_PATH%\VC\Auxiliary\Build\vcvars64.bat" (
    call "%VS_PATH%\VC\Auxiliary\Build\vcvars64.bat"
) else (
    echo Error: vcvars64.bat not found in %VS_PATH%
    pause
    exit /b 1
)

REM Create output directory
if not exist "..\..\bin" mkdir "..\..\bin"

echo.
echo Compiling MonitorService.cpp...
cl.exe /EHsc /O2 /DUNICODE /D_UNICODE /Fe:..\..\bin\MonitorService.exe ^
    /I"..\common" ^
    MonitorService.cpp ^
    /link fltLib.lib Advapi32.lib

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo Build failed!
    pause
    exit /b 1
)

echo.
echo ========================================
echo Build completed successfully!
echo Output: ..\..\bin\MonitorService.exe
echo ========================================
pause
