# Bonsai2 SYCL build

This checkout is a source-only SYCL build recipe for the Bonsai2 PTQ1 model.
The model file is intentionally kept outside the repository.

## Build on Windows

Open an x64 Intel oneAPI environment, then run:

```bat
scripts\build-sycl-ptq1-windows.bat
```

The build uses Intel oneAPI `icx`, oneDNN, Level Zero, and the Intel SYCL
backend. Build output is written to `build-sycl-ptq1` and is ignored by Git.

## Run

```powershell
pwsh -File scripts/run-bonsai2-sycl.ps1 `
  -Model C:\AI\bonsai2_27b\Ternary-Bonsai-2-27B-PTQ1_0.gguf `
  -Port 9941 -Device 0 -GpuLayers 99
```

Use `http://127.0.0.1:9941/health` and the OpenAI-compatible `/v1/chat/completions`
endpoint for a smoke test. Check the server log for `SYCL` device selection and
`offloaded` layer counts. PTQ1 kernels are not registered in `ggml-sycl` in this
snapshot. The current build detects the Arc GPU but aborts when the first
PTQ1 matvec reaches MMVQ (`unsupported data type=ptq1_0`). A successful
Bonsai2 GPU run requires implementing that kernel.
