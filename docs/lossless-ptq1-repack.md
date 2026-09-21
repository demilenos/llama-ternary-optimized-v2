# Lossless PTQ1 to Q2_0 repacking

This optional CPU tool changes storage, not weights: each 128-weight PTQ1
block becomes two 64-weight Q2_0 blocks. The fp16 scale bits are copied to
both blocks, and every -1/0/+1 trit is stored exactly. Q2_0 needs 36 bytes
instead of 28. This is not floating-point requantization.

Build the SYCL project first, then use an Intel oneAPI command prompt:

```bat
scripts\build-ptq1-repack-windows.bat
build-sycl-ptq1\bin\ptq1-to-q2.exe input.gguf output.gguf
build-sycl-ptq1\bin\ptq1-to-q2.exe --verify-only input.gguf output.gguf
```

The build requires the existing `ggml-base.lib` and uses `/O2`; verification
visits billions of values, so an unoptimized build is expensive. Keep the
oneAPI runtime directories on PATH, as for the existing llama programs.
The tool refuses to overwrite an existing output or temporary file. It
writes `output.gguf.tmp`, independently verifies the disk output, then renames
it. Verify-only mode does not modify the candidate.

Metadata and non-PTQ tensors are preserved, except `general.file_type` becomes
`LLAMA_FTYPE_MOSTLY_Q2_0`. Output tensor types, offsets and sizes change to
match Q2_0. Verification compares the expected serialized metadata, every
non-PTQ payload byte, all scale bits, and every ternary value using the CPU
16-byte/8-byte/tail decoding stages independently of the converter's position
mapping. It also checks alignment and complete file size. Processing is
chunked; no second full-model allocation is required.

## Validated Bonsai2 artifact

- Original: `Ternary-Bonsai-2-27B-PTQ1_0.gguf`, 5,946,648,928 bytes.
- Derived: `Bonsai2-27B-lossless-Q2_0.gguf`, 7,626,008,928 bytes.
- Converted 402 PTQ1 tensors; copied 449 other tensors; total 851.
- Validated 209,920,000 PTQ1 blocks / 26,869,760,000 ternary weights.
- Original SHA256: `53107F530AA52EB00912263AB1EE29BD199261C87CD7B4AD4CA1318C1FE33EE3`.
- Derived SHA256: `4F99AED01B8A877E153F9AA6569A4440FE17C59701CC0953E36D7F460549E70E`.

The original file remains unchanged. The derived file is a local artifact;
model weights are not committed to Git. Repacking correctness does not prove
GPU speed or runtime equivalence; see the separate performance measurements.

## Initial SYCL runtime comparison

The lossless Q2 file loads all 65/65 layers, with CPU token embeddings and a
6921.11 MiB SYCL model buffer. Q2 MUL_MAT CPU-reference tests passed 38/38.
With the same rebuilt DLL, TG128 r3 measured 8.431436 t/s (stddev 0.003038),
versus original PTQ1 19.921373 t/s (stddev 0.037021). The existing Q2 kernel
regresses despite exact weights; the original PTQ1 path remains preferred.
No Q2 TG1024 run was warranted for this baseline.

Both models returned `4`, `안녕하세요!`, and `Jupiter` for the three saved
quality prompts with temperature 0, seed 1, context 4096, max_tokens 64,
and `--reasoning off`; all finished with `stop` and no reasoning content.
This checks these prompts only, not general model equivalence.

Benchmark settings: SYCL0, GPU layers 99, batch/ubatch 512, Q8_0 K/V cache,
`token_embd.weight=CPU`, no prompt, 128 generated tokens, three repetitions,
warmup enabled, graph and operation profiling disabled. DLL SHA256:
`D39377FB7C842AE9A9087ED6EB9311323976285BC295903B6A792CCAFC7CF2B1`.
Local evidence is under
`build-sycl-ptq1/profile-ops/lossless-comparison/{original,q2}/`, including
`tg128-r3.json` and `reasoning-off/response-*.json`; backend evidence is
`build-sycl-ptq1/profile-ops/q2_0-baseline.log`.

## Optional full-block Q2 single-token kernel

Set `GGML_SYCL_Q2_FULL64=1` to let each lane decode a complete 64-weight
Q2 block into signed packed bytes for DP4A. Unset the variable to use the
existing helper (the switch tests presence, so `0` also enables it).
Multi-column dispatch and the original PTQ1 kernel are unchanged.

Forced Q2 CPU-reference tests passed 38/38. On the same DLL, TG128 r3 rose
from 8.428174 t/s (stddev 0.007010) to 11.432572 t/s (stddev 0.009512),
about 35.7%. The same three quality prompts passed with normal stops and
no reasoning content. This remains slower than original PTQ1; use it only
when evaluating the lossless Q2 derivative. It does not meet the 30 t/s goal.

DLL SHA256: `404FB93D8874F7274C4E69992331848F2D8AA3F9F459045AE8934F46919F58E`.
Evidence: `profile-ops/q2-full64-tests.log`,
`profile-ops/q2-full64-build-final.log`, and under the Q2 comparison directory,
`tg128-helper-r3.json`, `tg128-full64-r3.json`, and `full64-quality/`.
