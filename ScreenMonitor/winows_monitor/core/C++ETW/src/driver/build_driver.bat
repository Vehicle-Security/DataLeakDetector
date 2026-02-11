@echo off
REM Build script for File Monitor Service (User-mode)
REM Requires Visual Studio C++ Build Tools

echo ========================================
echo Building File Monitor Driver
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

REM Build Driver
echo.
echo Building FsFilter.sys...
msbuild FsFilter.vcxproj /p:Configuration=Release /p:Platform=x64 /p:SpectreMitigation=false /p:InfVerif_Enabled=false

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo Driver Build failed!
    pause
    exit /b 1
)

REM Copy output to current dir
if exist "x64\Release\FsFilter.sys" (
    copy /y "x64\Release\FsFilter.sys" "FsFilter.sys"
    echo Driver copied to src\driver\FsFilter.sys
)

echo.
echo ========================================
echo Driver Build completed!
echo ========================================
pause
