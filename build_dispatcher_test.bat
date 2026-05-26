@echo off
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" > nul
cmake --build build_pro --target test_bip_gpu_dispatcher --parallel 8
if errorlevel 1 exit /b %errorlevel%
build_pro\test_bip_gpu_dispatcher.exe
