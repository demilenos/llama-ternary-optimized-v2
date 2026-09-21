# Bonsai2 PTQ1 SYCL performance evidence

This note records the small, reproducible `llama-bench` measurements currently
available for the Intel Arc A750 path.  These are short benchmark cases, not a
claim about sustained server throughput at a full context.

## Matched benchmark settings

Both runs used the same model and settings: `Ternary-Bonsai-2-27B-PTQ1_0.gguf`,
`SYCL0`, `-ngl 99`, `-b 512`, `-ub 512`, Q8_0 K/V cache, CPU token embedding,
one parallel sequence, no speculative decoding, three samples, and the default
llama-bench warmup.  The commands were:

```powershell
pwsh -File scripts/bench-bonsai2-sycl.ps1 -CaseName pp512
pwsh -File scripts/bench-bonsai2-sycl.ps1 -CaseName tg128
pwsh -File scripts/bench-bonsai2-sycl.ps1 -CaseName tg1024 -TimeoutSeconds 1200
```

The harness preserves each run under a timestamped directory in
`build-sycl-ptq1/bench-bonsai2-sycl/`, including JSON, stderr, and metadata.

## Measurements

| Run | Samples (tokens/s) | Average | Std. dev. | Result |
|---|---:|---:|---:|---|
| Baseline PP512 | 194.341, 193.685, 193.379 | 193.801922 | 0.491245 | valid |
| Baseline TG128 | 4.55760, 4.54914, 4.52417 | 4.543639 | 0.017381 | valid |
| Candidate TG128 | 6.06720, 6.03734, 6.01825 | 6.040929 | 0.024674 | valid |
| Baseline TG1024 | no samples | — | — | exit 124 timeout; invalid throughput |

The candidate TG128 increase over the recorded baseline is 1.497290 tokens/s,
or 32.954% ((6.040929 / 4.543639 - 1) * 100).  It remains below the 30
tokens/s target.  PP512 has no post-optimization candidate measurement here.

## Provenance and limits

Baseline artifacts are in
`build-sycl-ptq1/bench-bonsai2-sycl/20260921-073730-c0520c00/`.
Their metadata reports git commit `3dee394f8a751e841c8fbf0b230195c3480fc10c`,
dirty status `docs/sycl-ptq1-bonsai2.md`, executable SHA256
`FC7541C4E343DD48C2ADBFB03EF1EE609D42F9BCEA653FDDAC6D44DC5C057CC9`, and
`ggml-sycl.dll` SHA256
`1863292B462C4B0ACB0207DD79B573413F1D6F18C3BB87EC0E98B2CE1A53FFC9`.

Candidate TG128 is in
`build-sycl-ptq1/bench-bonsai2-sycl/20260921-080216-c40ee4d7/`.
Its metadata was captured at commit
`0be57db6bc4e637c33c1edebbda0b5a08698628f`, dirty with
`ggml/src/ggml-sycl/vecdotq.hpp`; the accepted source commit is
`bd43bf090814f8c5d4ecb1c78e26c1ab2cba440`.  The executable SHA256 is unchanged;
the candidate `ggml-sycl.dll` SHA256 is
`84EB49E4B45724D6FB98DD53D29F1D67518084767B8351998F08EE95A160D3F4`.
The old baseline and candidate therefore differ in backend DLL as well as
source state.

The PTQ1 numerical logs `ptq1-packed4-test2.log` and
`ptq1-backend-test-n512-all.log` each report `39/39 tests passed` and
`Backend SYCL0: OK`.  Their scope is the existing backend operation suite's
PTQ1 MUL_MAT CPU-vs-SYCL cases, including ncols-1 and selected matrix sizes;
this is not full-model correctness.  Normal context-4 content was verified
separately, but its original JSON artifact was deleted, so no result is
reconstructed here.

The TG128 figures are short `llama-bench` decode measurements.  They should not
be presented as sustained server throughput, full-context performance, or
speculative-decoding performance.

## Full masked decoder candidate

The follow-up masked decoder candidate replaces the first 120 trit groups with
uint32 two-lane masked arithmetic and retains scalar decoding only for the qh
8-value tail. The q8 packed word index remains the `j` word index used by the
signed dp4a dot. It passed the same 39/39 SYCL0 PTQ1 tests, including n=512.

Candidate run: `build-sycl-ptq1/bench-bonsai2-sycl/20260921-081601-144bda3e/`.
Samples were 11.2458, 11.2402, and 11.2395 tokens/s; average 11.241808.
The source is committed as `f8153aa3e9e634620a6198fe619f7f05227d65de`.
This remains below the 30 tokens/s target and has no full-context TG1024
measurement yet.

## One-lane full-block candidate

For single-column TG dispatch, the PTQ1 launcher now uses one lane per full
128-value block (`QI=1`, `VDR=1`) and sums four explicit 32-value dot calls.
The multi-column path is unchanged. The candidate passed 39/39 SYCL0 PTQ1
cases including n=512.

TG128 run: `build-sycl-ptq1/bench-bonsai2-sycl/20260921-082617-efab95f7/`,
with samples 18.0495, 18.0735, and 18.0308 tokens/s; average 18.051252.
The preceding masked decoder TG1024 run completed successfully at 11.157880
tokens/s in `20260921-081806-eb3d6782/`. These remain short benchmark
measurements and below the 30 tokens/s target.

## Workgroup Y tuning result

The one-lane full-block path was tested with local workgroup Y=4 and Y=8,
leaving Y=1 as the selected baseline. Both candidates passed 39/39 SYCL0
PTQ1 cases, but neither improved TG128: Y=4 averaged 17.990086 tokens/s
(run `20260921-083605-1733e9b9`) and Y=8 averaged 17.931614 tokens/s
(run `20260921-083926-d45b733b`). The selected Y=1 result remains 18.051252
tokens/s, so the source was restored without a tuning commit.

## Final selected f65354d evidence

The selected one-lane full-block implementation completed the long TG1024
run successfully: samples 17.8395, 17.8232, and 17.8287 tokens/s; average
17.830466, standard deviation 0.008279. Artifacts are in
`build-sycl-ptq1/bench-bonsai2-sycl/20260921-084226-e80b57e4/`.

A current f65354d server with `--reasoning off`, `-ngl 99`, and the Bonsai2
PTQ1 model answered the fixed request `What is 2 + 2? Answer with only the
number.` with content `4` and finish reason `stop`. The preserved response is
`build-sycl-ptq1/quality/sycl-quality-response-final.json`.

## Optional synchronized operation profiling

`GGML_SYCL_PROFILE_OPS=1` logs each executed operation scope, including fused
paths charged to their first node. With `llama-bench`, add `-v` to enable logs.
The graph number is an invocation counter. All initialized context queues are
waited before the timer and after submission; profiling disables SYCL graph
capture. These are synchronized host wall times, including submission and wait
overhead, not GPU event times or production throughput. The default is off.

TG8 diagnostic artifacts: `build-sycl-ptq1/profile-ops/tg8-on-verbose.log`
and parsed `tg8-ops.csv`. Excluding startup, the final four decode invocations
(5 through 8) attribute 249616 us to PTQ1 MUL_MAT scopes and 153261 us to F32
MUL_MAT scopes. Of the latter, 133394 us is repeated 1024x1024 transforms with
5, 6, or 17 right-hand columns. The existing FWHT dispatcher stopped at 512,
so the 1024 Hadamard transform is the next measured optimization candidate.
Synchronization strongly magnifies small-operation costs; these totals must
not be interpreted as fractions of normal asynchronous token latency.

Default-off validation passed all 39 SYCL0 PTQ1 cases in
`build-sycl-ptq1/profile-ops/ptq1-default-off.log`. The matched TG128 rerun is
`build-sycl-ptq1/bench-bonsai2-sycl/20260921-091442-20216588/`.
