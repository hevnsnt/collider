@echo off
REM ============================================================
REM Wave 0 Validation - Windows CUDA
REM
REM Purpose: confirm the new GPU correctness tests fail-as-expected
REM against current code, proving Wave 1 has real signal to drive.
REM
REM Can be run from PowerShell, cmd.exe, or the VS Native Tools prompt.
REM Automatically initializes VS 2022 environment if not already active.
REM
REM Expected results on UNFIXED code (pre-Wave-1):
REM   HashVectors          - PASS  (10/10; C-CRIT-4 eliminated)
REM   Secp256k1Inv         - FAIL  (most inversions wrong; C-CRIT-2 confirmed)
REM   EcTableConsistency   - FAIL  (table entries off-curve; C-CRIT-2 confirmed)
REM   GpuHash160           - FAIL  (no/few matches; C-CRIT-1 confirmed)
REM
REM After Wave 1 crypto fixes: all four should PASS.
REM ============================================================

setlocal enabledelayedexpansion

set BUILD_DIR=build-wave0
set BUILD_TYPE=Release

echo =====================================================
echo Wave 0 Validation Build
echo =====================================================
echo.

REM =========================================================
REM Step 1: Auto-initialize VS 2022 x64 environment if needed
REM =========================================================
where cl.exe >nul 2>&1
if not errorlevel 1 (
    echo [*] VS 2022 tools already in PATH.
    goto :vs_ready
)

echo [*] VS tools not in PATH -- searching for VS 2022...
set "VCVARSALL="

for %%E in (Enterprise Professional Community) do (
    set "_C=C:\Program Files\Microsoft Visual Studio\2022\%%E\VC\Auxiliary\Build\vcvarsall.bat"
    if exist "!_C!" if not defined VCVARSALL set "VCVARSALL=!_C!"
)
set "_C=C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvarsall.bat"
if exist "!_C!" if not defined VCVARSALL set "VCVARSALL=!_C!"

if not defined VCVARSALL (
    echo [!] ERROR: VS 2022 Build Tools not found in any standard location.
    echo [!] Install from: https://visualstudio.microsoft.com/downloads/
    echo [!]   Choose "Build Tools for Visual Studio 2022"
    echo [!]   Workloads: "Desktop development with C++" + CUDA tools
    exit /b 1
)

echo [*] Initializing VS 2022 x64 environment from:
echo     !VCVARSALL!
call "!VCVARSALL!" x64 >nul
if errorlevel 1 (
    echo [!] vcvarsall.bat x64 failed.
    exit /b 1
)
echo [*] VS 2022 x64 environment ready.

:vs_ready
echo.

REM =========================================================
REM Step 2: Auto-detect VCPKG_ROOT if not already set/valid
REM =========================================================
if defined VCPKG_ROOT (
    if exist "!VCPKG_ROOT!\scripts\buildsystems\vcpkg.cmake" (
        echo [*] VCPKG_ROOT already set: !VCPKG_ROOT!
        goto :vcpkg_ready
    )
    echo [!] VCPKG_ROOT set but vcpkg.cmake not found there -- trying auto-detection.
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
    echo [!] vcpkg not found. Set VCPKG_ROOT before running this script.
    echo [!]
    echo [!]   PowerShell:  $env:VCPKG_ROOT = "C:\path\to\vcpkg"
    echo [!]   cmd.exe:     set VCPKG_ROOT=C:\path\to\vcpkg
    echo [!]
    echo [!] Common locations: C:\src\vcpkg  or  C:\vcpkg
    exit /b 1
)

:vcpkg_ready
echo [*] Using VCPKG_ROOT=!VCPKG_ROOT!
echo.

REM =========================================================
REM Step 3: Configure
REM =========================================================
if exist %BUILD_DIR% (
    echo [*] Wiping previous %BUILD_DIR% for clean state...
    rmdir /s /q %BUILD_DIR%
)

echo [*] Configuring with CUDA (NATIVE arch -- detects your GPU, skips unused archs)...
cmake -B %BUILD_DIR% -G Ninja ^
    -DCMAKE_BUILD_TYPE=%BUILD_TYPE% ^
    -DCMAKE_CUDA_ARCHITECTURES=native ^
    "-DCMAKE_TOOLCHAIN_FILE=!VCPKG_ROOT!\scripts\buildsystems\vcpkg.cmake"
if errorlevel 1 (
    echo [!] CMake configure FAILED
    exit /b 1
)

echo.
echo [*] Building test targets...
cmake --build %BUILD_DIR% --config %BUILD_TYPE% --parallel --target test_hash_vectors test_secp256k1_inv test_ec_table_consistency test_gpu_hash160
if errorlevel 1 (
    echo [!] Build FAILED -- report to Claude with the error above.
    exit /b 1
)

echo.
echo =====================================================
echo Running Wave 0 Tests
echo =====================================================
echo.

ctest --test-dir %BUILD_DIR% --output-on-failure -R "HashVectors|Secp256k1Inv|EcTableConsistency|GpuHash160"
set CTEST_EXIT=%ERRORLEVEL%

echo.
echo =====================================================
echo Result Interpretation
echo =====================================================
echo.
echo Expected on pre-Wave-1 code (C-CRIT-1/C-CRIT-2 still present):
echo   HashVectors:        PASS    (C-CRIT-4 already fixed)
echo   Secp256k1Inv:       FAILED  (mod_inv broken -- C-CRIT-2 confirmed)
echo   EcTableConsistency: FAILED  (table off-curve -- C-CRIT-2 confirmed)
echo   GpuHash160:         FAILED  (low/zero matches -- C-CRIT-1 confirmed)
echo.
echo If any EXPECTED-FAIL test passed: that is a surprise -- report it.
echo If any test did NOT BUILD or CRASHED: that is a real problem -- report it.
echo.

REM CTest exits non-zero when tests fail. For Wave 0 we EXPECT failures.
exit /b 0
