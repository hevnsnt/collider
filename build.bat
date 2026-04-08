@echo off
REM ============================================================
REM theCollider Windows Build Script
REM Run from: x64 Native Tools Command Prompt for VS 2022
REM ============================================================

setlocal enabledelayedexpansion

REM Enable ANSI colors in Windows terminal
for /f "tokens=3" %%A in ('reg query HKCU\Console /v VirtualTerminalLevel 2^>nul') do set "VT=%%A"
if not defined VT (
    reg add HKCU\Console /v VirtualTerminalLevel /t REG_DWORD /d 1 /f >nul 2>&1
)

REM Configuration
set PROJECT_DIR=D:\theCollider
set BUILD_TYPE=Release

REM Colors - using escape character
for /f %%a in ('echo prompt $E^| cmd') do set "ESC=%%a"
set "GREEN=%ESC%[92m"
set "YELLOW=%ESC%[93m"
set "RED=%ESC%[91m"
set "CYAN=%ESC%[96m"
set "RESET=%ESC%[0m"

echo %CYAN%============================================================%RESET%
echo %CYAN%  theCollider Build Script%RESET%
echo %CYAN%============================================================%RESET%
echo.

REM Check if running from VS Developer Command Prompt
where cl.exe >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo %RED%ERROR: cl.exe not found in PATH%RESET%
    echo %YELLOW%Please run this from "x64 Native Tools Command Prompt for VS 2022"%RESET%
    echo.
    echo Or run this first:
    echo "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
    exit /b 1
)

REM Get cl.exe path dynamically
for /f "delims=" %%i in ('where cl.exe') do set "CL_PATH=%%i"
echo %GREEN%[+]%RESET% Found compiler: %CL_PATH%

REM Navigate to project directory
cd /d %PROJECT_DIR%
if %ERRORLEVEL% neq 0 (
    echo %RED%ERROR: Could not navigate to %PROJECT_DIR%%RESET%
    exit /b 1
)
echo %GREEN%[+]%RESET% Working directory: %CD%

REM Pull latest changes
echo.
echo %CYAN%[*] Pulling latest changes...%RESET%
git pull
if %ERRORLEVEL% neq 0 (
    echo %YELLOW%[!] Git pull failed or no remote configured%RESET%
)

REM Check if clean build is needed
set CLEAN_BUILD=0
if "%1"=="clean" set CLEAN_BUILD=1
if "%1"=="-c" set CLEAN_BUILD=1
if "%1"=="--clean" set CLEAN_BUILD=1

if %CLEAN_BUILD%==1 (
    echo.
    echo %CYAN%[*] Clean build requested - removing build directory...%RESET%
    if exist build rmdir /s /q build
)

REM Create build directory if needed
if not exist build (
    echo %CYAN%[*] Creating build directory...%RESET%
    mkdir build
)

cd build

REM Run CMake if needed (check for build.ninja)
if not exist build.ninja (
    echo.
    echo %CYAN%[*] Running CMake configuration...%RESET%

    REM Convert backslashes to forward slashes for CMake
    set "CL_PATH_CMAKE=!CL_PATH:\=/!"

    cmake .. -G Ninja -DCMAKE_BUILD_TYPE=%BUILD_TYPE% -DCMAKE_CXX_COMPILER="!CL_PATH_CMAKE!"
    if %ERRORLEVEL% neq 0 (
        echo %RED%ERROR: CMake configuration failed%RESET%
        exit /b 1
    )
) else (
    echo %GREEN%[+]%RESET% build.ninja exists, skipping CMake (use 'build.bat clean' to reconfigure)
)

REM Build with Ninja
echo.
echo %CYAN%[*] Building with Ninja...%RESET%
ninja
if %ERRORLEVEL% neq 0 (
    echo.
    echo %RED%============================================================%RESET%
    echo %RED%  BUILD FAILED%RESET%
    echo %RED%============================================================%RESET%
    exit /b 1
)

echo.
echo %GREEN%============================================================%RESET%
echo %GREEN%  BUILD SUCCESSFUL%RESET%
echo %GREEN%============================================================%RESET%
echo.
echo Executable: %PROJECT_DIR%\build\collider.exe
echo.
echo Usage:
echo   collider.exe --help
echo   collider.exe --puzzle 71
echo   collider.exe --brainwallet --bloom addresses.bloom
echo.

endlocal
