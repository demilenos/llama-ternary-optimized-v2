param(
    [string]$Model = "C:\AI\bonsai2_27b\Ternary-Bonsai-2-27B-PTQ1_0.gguf",
    [int]$Port = 9941,
    [int]$Device = 0,
    [int]$GpuLayers = 99,
    [int]$Context = 4096
)
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$server = Join-Path $root "build-sycl-ptq1\bin\llama-server.exe"
if (!(Test-Path -LiteralPath $server)) { throw "Build llama-server first: $server" }
if (!(Test-Path -LiteralPath $Model)) { throw "Model not found: $Model" }
$env:PATH = "C:\Program Files (x86)\Intel\oneAPI\compiler\latest\bin;C:\Program Files (x86)\Intel\oneAPI\mkl\latest\redist\intel64;C:\Program Files (x86)\Intel\oneAPI\tbb\latest\redist\intel64\vc14;" + $env:PATH
$env:GGML_SYCL_DEVICE = "$Device"
$args = @("-m", $Model, "-c", "$Context", "-ngl", "$GpuLayers", "--port", "$Port", "--host", "127.0.0.1")
Write-Host "Starting SYCL llama-server on port $Port (device $Device)"
& $server @args
