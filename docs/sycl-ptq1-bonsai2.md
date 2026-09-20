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
endpoint for a smoke test. The SYCL backend now includes PTQ1 MMVQ support for
Q8_1 activations, PTQ1-to-F16/F32 conversion for prefill paths, and the
reference-accurate trit decoder (including uint8 wrapping). CPU token embedding
fallback remains intentional because GET_ROWS is not advertised by this
backend.

The implementation was validated with `test-backend-ops` PTQ1 MUL_MAT cases
(38/38 on SYCL0) and a real Bonsai2 response (`4`) with `--reasoning off`,
`-ngl 99`, and the Arc GPU selected. Use `scripts/bench-bonsai2-sycl.ps1` for
matched PP512, TG128, and TG1024 measurements; its logs record the executable
and backend DLL hashes. Build outputs and model weights remain outside Git.
