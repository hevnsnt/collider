@echo off
REM ============================================================================
REM Compute-sanitizer wrapper for collider_pro V2 multi_addr fault diagnosis.
REM
REM What this does:
REM   Runs collider_pro.exe under NVIDIA compute-sanitizer in memcheck mode.
REM   Memcheck instruments every device-side memory access; the first OOB
REM   load or store prints its file, line, kernel name, thread, and address.
REM   Throughput drops 5-10x, so a fault that normally hits at minute ~40
REM   should land within 3-7 hours of sanitized scanning. The slowdown
REM   actually INCREASES the rate of bad-input encounters per wall-clock
REM   hour for content-dependent bugs, so the first OOB tends to land
REM   sooner than the slowdown ratio suggests.
REM
REM Usage:
REM   scripts\run_compute_sanitizer.bat [--brainwallet-v2 --resume ...]
REM
REM   All arguments after the script name are forwarded to collider_pro.exe.
REM   The user's reproducible faulting command was:
REM     collider_pro.exe --brainwallet-v2 --resume
REM
REM Output:
REM   sanitizer_report.txt in the current directory. The first
REM   "Invalid __global__ read of size N" / "Invalid __global__ write of
REM   size N" stanza is what we want; it carries kernel name + source line
REM   + thread id + offending address. After the first violation memcheck
REM   keeps running (--print-limit 1 caps it to one report per kernel) so
REM   the report doesn't blow up if the same access fires across many
REM   threads.
REM
REM   Both stderr (sanitizer output) and stdout (collider's normal scan
REM   output) end up in the same file so the time-of-fault correlates
REM   against the scan rate and any TUI events that printed before the
REM   crash.
REM
REM Stopping:
REM   The sanitizer process honours Ctrl+C the same way collider_pro does.
REM   Stop it as soon as you see the first [BLOOM_OOB] / "Invalid
REM   __global__ read" stanza so you don't burn cycles producing
REM   duplicate reports.
REM ============================================================================

setlocal

REM Locate compute-sanitizer. CUDA 12.x ships it under the CUDA install
REM directory; v13 keeps the same path. Prefer the highest version found.
set "SANITIZER="
for %%V in (v13.0 v12.9 v12.8 v12.6 v12.4 v12.3 v12.0) do (
    if not defined SANITIZER (
        if exist "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\%%V\compute-sanitizer\compute-sanitizer.exe" (
            set "SANITIZER=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\%%V\compute-sanitizer\compute-sanitizer.exe"
        )
    )
)

if not defined SANITIZER (
    echo [!] Could not find compute-sanitizer.exe under any installed
    echo     C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v*\.
    echo     Install the CUDA Toolkit (the matching driver alone does not
    echo     include it) and re-run.
    exit /b 2
)

REM Prefer the diagnostic build (build_dbg/) if it exists. The diagnostic
REM build is the one configured with -DCOLLIDER_DEBUG_FUSED_BOUNDS=ON and
REM has the in-kernel printf bounds-check; sanitizer + printf together
REM gives belt + braces. If only the regular Pro build exists, fall back
REM to that so the wrapper is still useful before the user reconfigures.
set "COLLIDER_EXE="
if exist "%~dp0..\build_dbg\collider_pro.exe" (
    set "COLLIDER_EXE=%~dp0..\build_dbg\collider_pro.exe"
    echo [+] Using diagnostic build with bounds-check printf
) else if exist "%~dp0..\build_dbg\collider.exe" (
    set "COLLIDER_EXE=%~dp0..\build_dbg\collider.exe"
    echo [+] Using diagnostic build with bounds-check printf
) else if exist "%~dp0..\build_pro\collider_pro.exe" (
    set "COLLIDER_EXE=%~dp0..\build_pro\collider_pro.exe"
    echo [+] Using regular Pro build, no in-kernel bounds-check
)
if "%COLLIDER_EXE%"=="" (
    echo [!] No collider binary found.
    echo     Build first: build_pro.bat (or the build_dbg recipe documented
    echo     in fused_pipeline.cu bloom_check_inline).
    exit /b 2
)

echo Using sanitizer: %SANITIZER%
echo Target:          %COLLIDER_EXE%
echo Args:            %*
echo Logging to:      sanitizer_report.txt
echo.

REM --tool memcheck             : instrument every device memory access
REM --print-limit 1             : one error report per access site (avoids
REM                               drowning in duplicates from sister warps)
REM --launch-timeout 0          : never abort on slow launches
REM --target-processes all      : follow any child processes the runner spawns
REM --log-file sanitizer_report.txt : capture both sanitizer and stdout
REM --save sanitizer_report.bin : keep a binary trace for later inspection
"%SANITIZER%" --tool memcheck ^
    --print-limit 1 ^
    --launch-timeout 0 ^
    --target-processes all ^
    --log-file sanitizer_report.txt ^
    --save sanitizer_report.bin ^
    "%COLLIDER_EXE%" %*

set "RC=%ERRORLEVEL%"
echo.
echo compute-sanitizer exited with code %RC%.
echo Open sanitizer_report.txt and search for "Invalid __global__" or
echo "[BLOOM_OOB]" to find the first violation.
exit /b %RC%
