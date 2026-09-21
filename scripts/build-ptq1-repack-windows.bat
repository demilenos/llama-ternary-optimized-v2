@echo off
setlocal
pushd "%~dp0.."
call "C:\PROGRA~2\Intel\oneAPI\setvars.bat" intel64
if errorlevel 1 exit /b %errorlevel%
icx /O2 /EHsc /Qstd=c++17 /Iinclude /Iggml/include /Iggml/src tools\ptq1_to_q2.cpp build-sycl-ptq1\ggml\src\ggml-base.lib /link /OUT:build-sycl-ptq1\bin\ptq1-to-q2.exe
set "REPACK_BUILD_RC=%errorlevel%"
popd
exit /b %REPACK_BUILD_RC%
