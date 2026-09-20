@echo off
setlocal
set "SRC=%~dp0.."
set "BUILD=%SRC%\build-sycl-ptq1"
call "C:\PROGRA~2\Intel\oneAPI\setvars.bat" intel64
if errorlevel 1 exit /b %errorlevel%
cmake -S "%SRC%" -B "%BUILD%" -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icx -DGGML_SYCL=ON -DGGML_SYCL_DNN=ON -DGGML_SYCL_GRAPH=ON -DGGML_SYCL_TARGET=INTEL -DGGML_SYCL_SUPPORT_LEVEL_ZERO_API=ON -DGGML_VULKAN=OFF -DGGML_CUDA=OFF -DENABLE_TRY_SYCL_COMPILE=OFF
if errorlevel 1 exit /b %errorlevel%
cmake --build "%BUILD%" --target llama-server -j 8
exit /b %errorlevel%
