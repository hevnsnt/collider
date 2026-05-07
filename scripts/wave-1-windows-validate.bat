@echo off
REM ============================================================
REM Wave 1+ Validation - Windows CUDA
REM
REM Run after pulling the full-opto branch. Confirms all crypto
REM fixes and new tests are green on Windows + CUDA.
REM
REM Can be run from PowerShell, cmd.exe, or the VS Native Tools prompt.
REM Automatically initializes VS 2022 environment if not already active.
REM
REM Expected results (full-opto, all waves landed):
REM   HashVectors          - PASS  (10/10)
REM   Secp256k1Inv         - PASS  (mod_inv correct)
REM   EcTableConsistency   - PASS  (table on-curve)
REM   PuzzleOptimizedInv   - PASS  (puzzle_optimized.cu mod_inv)
REM   GpuHash160           - PASS  (fused mod_reduce + bloom fix)
REM   EcMulKnownAnswers    - PASS  (secp256k1 pubkeys k=1,2,3,7)
REM   KangarooSmallPuzzle  - PASS  (ec_mul_glv pubkeys k=1,2,3,7)
REM   RuleEngineKAT        - PASS  (38 hashcat rule vectors)
REM   JLPPoolHandshake     - PASS  (TLS + JLP frame; SKIP if offline)
REM   CLIParser            - PASS  (Wave 5 CLI matrix, 67+ rows)
REM   PriorityQueue        - PASS
REM   RuleEngine           - PASS
REM   Platform             - PASS
REM
REM Any FAIL here is a real bug -- report it with the output above.
REM Any test that DID NOT BUILD is a build regression -- report it.
REM ============================================================

setlocal enabledelayedexpansion

set BUILD_DIR=build-wave1
set BUILD_TYPE=Release

echo =====================================================
echo Wave 1+ Validation Build
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
REM Step 3: Configure and build
REM =========================================================
if exist %BUILD_DIR% (
    echo [*] Wiping previous %BUILD_DIR% for clean state...
    rmdir /s /q %BUILD_DIR%
)

echo [*] Configuring with CUDA + Pro features (NATIVE arch -- detects your GPU, skips unused archs)...
cmake -B %BUILD_DIR% -G Ninja ^
    -DCMAKE_BUILD_TYPE=%BUILD_TYPE% ^
    -DCMAKE_CUDA_ARCHITECTURES=native ^
    -DCOLLIDER_PRO=ON ^
    "-DCMAKE_TOOLCHAIN_FILE=!VCPKG_ROOT!\scripts\buildsystems\vcpkg.cmake"
if errorlevel 1 (
    echo [!] CMake configure FAILED
    exit /b 1
)

echo.
echo [*] Building all targets...
cmake --build %BUILD_DIR% --config %BUILD_TYPE% --parallel
if errorlevel 1 (
    echo [!] Build FAILED -- report to Claude with the error above.
    exit /b 1
)

echo.
echo =====================================================
echo Running All Tests
echo =====================================================
echo.

ctest --test-dir %BUILD_DIR% --output-on-failure
set CTEST_EXIT=%ERRORLEVEL%

echo.
echo =====================================================
echo Pool Smoke Test (manual)
echo =====================================================
echo.
echo To verify the JLP pool client manually:
echo   1. Run a local collision-protocol server.
echo   2. %BUILD_DIR%\collider.exe --pool jlps://localhost:8443 --worker bc1q...
echo   3. Confirm AUTH succeeds and DPs are submitted.
echo.
echo To verify TLS hostname rejection:
echo   1. Connect to a server presenting a cert for a different hostname.
echo   2. Client must REFUSE the connection (D-H1).
echo.

echo =====================================================
echo Brainwallet Smoke Test (manual)
echo =====================================================
echo.
echo To verify the full brain wallet pipeline against a known answer:
echo   1. Build a tiny bloom containing the puzzle-65 hash160 (or any known address).
echo   2. %BUILD_DIR%\collider.exe --brainwallet --bloom test.blf --wordlist wordlist.txt
echo   3. The known passphrase should produce a hit.
echo.

if %CTEST_EXIT% EQU 0 (
    echo [+] CTest reports ALL TESTS PASSED.
    echo [+] All waves validated on this Windows + CUDA box.
) else (
    echo [!] One or more tests FAILED. See output above.
    echo [!] This is a real bug -- report to Claude with the full output.
)

exit /b %CTEST_EXIT%
