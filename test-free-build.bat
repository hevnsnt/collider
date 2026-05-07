@echo off
REM ============================================================================
REM Local FREE-build test harness.
REM
REM Configures, builds, and runs ctest with -DCOLLIDER_PRO=OFF in a dedicated
REM build directory (build-free) so it does not stomp on a Pro build sitting in
REM build-wave1 / build-wave0. Mirrors the configure flags used by the GitHub
REM Actions release workflow (.github/workflows/release.yml) so a green run
REM here is the same shape as a green CI run.
REM
REM Expected: 15 tests pass. test_gpu_hash160 is excluded by CMake when
REM COLLIDER_PRO=OFF because it links against the brain-wallet pipeline.
REM
REM Usage:
REM   .\test-free-build.bat            REM incremental build
REM   .\test-free-build.bat clean      REM wipe build-free first
REM ============================================================================

setlocal enabledelayedexpansion

set BUILD_DIR=build-free
set BUILD_TYPE=Release
set REPO_ROOT=%~dp0
if "%REPO_ROOT:~-1%"=="\" set REPO_ROOT=%REPO_ROOT:~0,-1%

echo =====================================================
echo theCollider FREE Build (COLLIDER_PRO=OFF)
echo Build dir: %REPO_ROOT%\%BUILD_DIR%
echo =====================================================
echo.

REM ----- Step 1: VS 2022 environment ----------------------------------------
where cl.exe >nul 2>&1
if not errorlevel 1 (
    echo [*] VS 2022 tools already in PATH.
    goto :vs_ready
)

set "VCVARSALL="
for %%E in (Enterprise Professional Community) do (
    set "_C=C:\Program Files\Microsoft Visual Studio\2022\%%E\VC\Auxiliary\Build\vcvarsall.bat"
    if exist "!_C!" if not defined VCVARSALL set "VCVARSALL=!_C!"
)
set "_C=C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvarsall.bat"
if exist "!_C!" if not defined VCVARSALL set "VCVARSALL=!_C!"

if not defined VCVARSALL (
    echo [!] ERROR: VS 2022 not found in any standard location.
    exit /b 1
)
call "!VCVARSALL!" x64 >nul
if errorlevel 1 (
    echo [!] vcvarsall.bat x64 failed.
    exit /b 1
)
:vs_ready

REM ----- Step 2: vcpkg detection --------------------------------------------
if defined VCPKG_ROOT (
    if exist "!VCPKG_ROOT!\scripts\buildsystems\vcpkg.cmake" (
        goto :vcpkg_ready
    )
    set "VCPKG_ROOT="
)
for %%P in (
    "C:\src\vcpkg"
    "C:\vcpkg"
    "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\vcpkg"
) do (
    if exist "%%~P\scripts\buildsystems\vcpkg.cmake" (
        if not defined VCPKG_ROOT set "VCPKG_ROOT=%%~P"
    )
)
if defined USERPROFILE (
    if exist "!USERPROFILE!\vcpkg\scripts\buildsystems\vcpkg.cmake" (
        if not defined VCPKG_ROOT set "VCPKG_ROOT=!USERPROFILE!\vcpkg"
    )
)
if not defined VCPKG_ROOT (
    echo [!] vcpkg not found. Set VCPKG_ROOT before running.
    exit /b 1
)
:vcpkg_ready
echo [*] VCPKG_ROOT=!VCPKG_ROOT!

REM ----- Step 3: Optional clean ---------------------------------------------
if /I "%~1"=="clean" (
    if exist "%REPO_ROOT%\%BUILD_DIR%" (
        echo [*] Wiping %BUILD_DIR%...
        rmdir /s /q "%REPO_ROOT%\%BUILD_DIR%"
    )
)

REM ----- Step 4: Configure --------------------------------------------------
echo.
echo [*] Configuring (CUDA, NATIVE arch)...
cmake -B "%REPO_ROOT%\%BUILD_DIR%" -S "%REPO_ROOT%" -G Ninja ^
    -DCMAKE_BUILD_TYPE=%BUILD_TYPE% ^
    -DCMAKE_CUDA_ARCHITECTURES=native ^
    -DCOLLIDER_PRO=OFF ^
    -DCOLLIDER_BUILD_TESTS=ON ^
    "-DCMAKE_TOOLCHAIN_FILE=!VCPKG_ROOT!\scripts\buildsystems\vcpkg.cmake"
if errorlevel 1 (
    echo [!] CMake configure FAILED
    exit /b 1
)

REM ----- Step 5: Build ------------------------------------------------------
echo.
echo [*] Building...
cmake --build "%REPO_ROOT%\%BUILD_DIR%" --config %BUILD_TYPE% --parallel
if errorlevel 1 (
    echo [!] Build FAILED
    exit /b 1
)

REM ----- Step 6: Test -------------------------------------------------------
echo.
echo [*] Running ctest (expect 15 tests; test_gpu_hash160 is Pro-only)...
ctest --test-dir "%REPO_ROOT%\%BUILD_DIR%" --output-on-failure
set CTEST_EXIT=%ERRORLEVEL%

echo.
if %CTEST_EXIT% EQU 0 (
    echo [+] FREE build: ALL TESTS PASSED.
) else (
    echo [!] FREE build: one or more tests FAILED.
)
exit /b %CTEST_EXIT%
