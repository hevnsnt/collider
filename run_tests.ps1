$vcvars = 'C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat'
$out = cmd /c "`"$vcvars`" > nul 2>&1 && ctest --test-dir D:\theCollider\build-wave1 --output-on-failure 2>&1"
$out
