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

## Shared-memory FWHT1024

The first attempt extending the subgroup/register FWHT to 1024 passed 33/33
Hadamard cases but regressed TG128 to 13.262185 t/s (stddev 0.018270).
It was replaced with a 256-thread workgroup per row using 4 KiB local memory
and a barrier after each butterfly stage. The existing 64–512 paths remain
unchanged. Existing Hadamard tests, including signed 1024 transforms and
non-multiple row counts, pass 33/33 on SYCL0 against CPU reference.

With profiling off and the same Q8 caches, offload and embedding settings:

| Case | Mean tokens/s | Stddev |
| --- | ---: | ---: |
| TG128, 3 repeats | 19.942413 | 0.007598 |
| TG1024, 3 repeats | 19.556190 | 0.030401 |

Evidence is under `build-sycl-ptq1/profile-ops/`: `build-fwht1024-local.log`,
`hadamard-1024-local.log`, `fwht-local-tg128-r3-off.log`,
`fwht-local-tg1024-r3-off.log`, and `fwht-local-tg8-on-verbose.log`.
Measured DLL SHA256:
`4A10F587C1A5C55795BD7025631E304A71F7EDB23F4B257E38D97597E01DCF70`.
The synchronized diagnostic 1024 transform mean fell to 90.49 us versus
210.44 us in the rejected register-only candidate. This diagnostic mean is
not normal asynchronous GPU kernel time. The 30 t/s target remains unmet.

## Rejected command-graph experiment

Removing the stale CONCAT compatibility gate allowed real SYCL command-graph
capture and submission (nine of each in the TG8 diagnostic), and CONCAT CPU
reference tests passed 192/192. However this A750 runtime reports
`graph_update_support=0`, so the backend finalizes a fresh executable graph
on each invocation. A same-DLL TG128 comparison with debug and profiling off
regressed from 19.894006 t/s (stddev 0.019536, graph off) to 14.273919 t/s
(stddev 0.171003, graph on). The candidate gate change and temporary markers
were discarded; the default graph setting remains off.

Logs under `build-sycl-ptq1/profile-ops/`: `graph-markers-tg8.log`,
`concat-graph-gate.log`, `graph-ab-off-tg128-r3.json`, and
`graph-ab-on-tg128-r3.json`. Llama's separate `graphs reused` counter was not
used as evidence of SYCL command-graph replay.

## Rejected initial ESIMD PTQ1 kernel

A four-workitem-per-row ESIMD kernel reused scalar packed-trit loads and
native `esimd::dp4a`. Intel ESIMD takes the accumulator first, unlike the
existing dpct helper; correcting that order produced 39/39 CPU-reference
PTQ1 passes with explicit dispatch markers. However matched TG128 r3 on the
same DLL regressed from 19.921700 t/s (stddev 0.017255, ESIMD off) to
1.619624 t/s (stddev 0.009667, ESIMD on). The initial candidate was discarded;
using a native dot instruction alone did not improve this kernel design.
No TG1024 run was warranted.

Evidence under `build-sycl-ptq1/profile-ops/`:
`build-ptq1-esimd-dp4a-fix.log`, `ptq1-esimd-on-dp4a-fix.log`,
`ptq1-esimd-off-tg128-r3.json`, and `ptq1-esimd-on-tg128-r3.json`.

## PTQ1 subgroup8 scheduling

An optional single-token launcher keeps the full128 decoder unchanged and
uses eight lanes per row. This divides the 40 and 136 PTQ1 blocks in the
5120- and 17408-wide FFN matrices without a half-full final iteration.
Set `GGML_SYCL_PTQ1_SG8=1`; unset it for the original launcher. The gate
tests presence, so setting it to `0` also enables it. This requires a device
supporting subgroup size 8 and was validated on the Intel Arc A750.
Multi-column dispatch remains unchanged.

Forced PTQ1 CPU-reference tests passed 39/39. Same-DLL TG128 r3 improved
from 19.874161 t/s to 20.892033 t/s (candidate stddev 0.006450), about 5.1%.
TG1024 r3 reached 20.548063 t/s (stddev 0.004126). The three saved quality
prompts returned `4`, `안녕하세요!`, and `Jupiter`, all with normal stops and
no reasoning content, using `--reasoning off`, temperature 0 and seed 1.
The 30 t/s target remains unmet.

The original PTQ1 GGUF, SYCL0, GPU layers 99, CPU token embeddings, Q8 K/V,
batch/ubatch 512, and graph/profiling-off settings match earlier measurements.
DLL SHA256: `9D66C76A755AE3568081CDDCEDDB0FF1DD03E44016E83B9721B72AFAFE8D72E6`.
Local evidence: `build-sycl-ptq1/profile-ops/ptq-sg8-{build,tests}.log`,
`ptq-sg8-{off,on,tg1024}.json`, and `ptq-sg8-quality/`.

## PTQ1 gate/up/SwiGLU fusion

With `GGML_SYCL_PTQ1_FFN_FUSION=1`, eligible single-token contiguous PTQ1
gate/up matmuls share one Q8 activation conversion and write the SwiGLU result
directly. Eligibility preserves identical activation/weight shape, graph
consumer closure, unswapped SwiGLU, contiguous F32 output and non-split buffers.
The default remains opt-in; unset the variable to disable it (`0` is still
present and therefore enables it). Existing Q4_K fusion remains unchanged.

Actual dispatch was verified before accepting numerical results. The graph
suite passed 37/37, including production K=5120, for both subgroup8 and the
build's default subgroup16. Clean same-DLL TG128 r3 with subgroup8 improved
from 20.940706 to 21.626604 t/s (candidate stddev 0.031351), about 3.3%.
TG1024 r3 reached 21.196389 t/s (stddev 0.006685). The three saved quality
prompts passed with `--reasoning off`, normal stops and no reasoning content.
Earlier runs with an unreachable predicate or debug logging enabled are not
performance evidence. The 30 t/s target remains unmet.

Enable the validated combination in an initialized oneAPI shell:

```powershell
$env:GGML_SYCL_PTQ1_SG8 = '1'
$env:GGML_SYCL_PTQ1_FFN_FUSION = '1'
$env:GGML_SYCL_ENABLE_GRAPH = '0'
$env:GGML_SYCL_PROFILE_OPS = '0'
$env:GGML_SYCL_DEBUG = '0'
.\scripts\bench-bonsai2-sycl.ps1 -CaseName tg128,tg1024
```

DLL SHA256: `5B9D7BA092DE8E8C7DAF6158D01351966F17089BB29A6B688C0EE5791857FF29`.
Local evidence under `build-sycl-ptq1/profile-ops/`:
`ptq-fusion-k5120-tests.log`, `ptq-fusion-k5120-sg32-tests.log` (despite the
filename this records subgroup16), `ptq-fusion-valid-off.json`,
`ptq-fusion-perf-on.json`, `ptq-fusion-tg1024.json`, and `ptq-fusion-quality/`.

## Signed FWHT fusion

`GGML_SYCL_FWHT_SIGNED_FUSION=1` folds the sign-vector MUL into the 1024-point
FWHT load for the exact MUL/RESHAPE/hinted-MUL_MAT graph pattern. Consumer
closure, contiguous F32 tensors, matching sign width and non-split buffers
are required. Sign rows repeat across tokens; shared intermediates and
unsupported layouts fall back. Unset the variable to disable this opt-in
path (presence, including `0`, enables it).

The final DLL passed 33/33 Hadamard CPU-reference tests with actual signed
fusion dispatch, including multi-token sign repetition. With SG8 and FFN
fusion explicitly enabled in both runs, clean same-DLL TG128 r3 improved
from 21.479940 to 22.325428 t/s (stddev 0.036170). TG1024 r3 reached
21.830764 t/s (stddev 0.003340). Earlier signed-only and diagnostic runs are
not the combined configuration's performance evidence.

Final server verification used the same DLL with all three opt-ins, graph
and profiling disabled, context 4096, Q8 K/V, CPU embeddings and
`--reasoning off`. Temperature-0/seed-1 responses were `4`, `안녕하세요!`,
and `Jupiter`, all normal stops without reasoning content.
Add `$env:GGML_SYCL_FWHT_SIGNED_FUSION = '1'` to the preceding benchmark
example to enable the validated combination. The 30 t/s target is unmet.

DLL SHA256: `8BF1782496D730C18A0F936A9A4C8D7093C915792790167106EF2B6166ECB317`.
Evidence under `build-sycl-ptq1/profile-ops/`:
`fwht-signed-parent-final-tests.log`, `fwht-fusion-correct-{off,on,tg1024}.json`
with matching metadata files, and `fwht-parent-quality/` with explicit
settings and responses. Earlier failed build/test invocations were not used
as validation of the final DLL.

## Runtime adapter comparison

A same-DLL TG128 r3 comparison with all three PTQ1 optimizations enabled
measured 22.280382 t/s with `SYCL_UR_USE_LEVEL_ZERO_V2=0` and 8.641881 t/s
with `=1`. Both exits were 0; graph, profiling and debug were disabled,
and immediate-command-list variables were unset. V2 is not recommended for
this validated A750 configuration. The variable is described in Intel's
[environment documentation](https://intel.github.io/llvm/EnvironmentVariables.html);
its behavior and performance remain runtime/device-specific.

Evidence: `build-sycl-ptq1/bench-bonsai2-sycl/20260921-115656-449262f4/`
and `20260921-115734-88f49269/`, including exact environment metadata and
DLL SHA256. These runs also verified the harness under Windows PowerShell
5.1. The harness now records a fixed allowlist of relevant performance
variables and supports both Windows PowerShell and PowerShell 7 process APIs.

Fixed command-list batches did not improve the accepted configuration:
TG128 r3 measured 20.813260 t/s for batch 16 and 22.328139 t/s for batch 64
(stddev 0.035304 for batch 64), essentially matching the dynamic baseline.
Both fixed-batch runs had V2 unset and immediate lists disabled. No fixed
batch setting is recommended from this comparison. Evidence:
`bench-bonsai2-sycl/20260921-115934-25b1102e/` and
`bench-bonsai2-sycl/20260921-120014-8f9ef48c/` under the build directory.
