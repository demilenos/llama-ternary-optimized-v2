# PTQ1 XMX8 evaluation

Source supplied by the user: `C:\AI\tools\ptq1-sycl-probe-v0.1.9`.
This is an evaluation, not an enabled llama.cpp backend feature.

The supplied A750 M=10240, N=1, K=5120 measurements report baseline
447.9685 us total versus 300 us for `all` (1.493x). The baseline is the
probe's ESIMD/DPAS kernel, not this repository's SG8 kernel. Its result
cannot be multiplied directly into current model TG throughput.

Source inspection confirms XMX8 is a byte permutation: each group of eight
28-byte PTQ1 blocks remains 224 bytes, with 208 payload bytes transposed by
byte position and 16 scale bytes. The probe allocates either row-major or
XMX8 device weights, not both. No expanded resident weight buffer is needed.
A llama.cpp port must preserve this property to respect the user's 6 GB
GPU-weight limit; tensor reads, prefill and fallback must also understand
any changed device layout.

The original integer probe uses one activation scale per 128 values and
checks three output rows. Current llama.cpp uses Q8_1 scales per 32 values.
A workspace-only comparison copy is being prepared with per-32 half-rounded
scales, independent DPAS chunk scaling and full-output checking at N=1.
Current llama.cpp also has a matching M=10240/N=1/K=5120 performance case
under evaluation. Original probe sources are not modified.
Current SG8 backend measurement at the same matrix shape reported
48.75 us/run (20988 repeated operations, 2.15 TFLOPS) with graph/profile/debug
disabled. Evidence: `build-sycl-ptq1/profile-ops/ptq-sg8-10240-perf.log`.
The backend performance harness repeats operations in a graph and measures
amortized host wall time, whereas the standalone probe uses profiled event
spans with per-iteration waits. Quantization also differs. This is a strong
reason to require the same-harness comparison, not proof of an exact 6x
advantage under real-model traffic. The same-size performance test case is
retained for reproducible follow-up.

## Same-harness Q32 comparison

The local comparison copy now uses one activation scale per 32 values,
float-scale quantization followed by FP16 scale rounding, ties-away rounding,
and separate scaling of all four DPAS chunks. SG8 uses the current packed
trit decoder/dot formula and subgroup reduction, with the probe's shared
activation buffers. Thus it is a kernel comparison, not an exact reproduction
of the full backend's Q8_1 buffer layout. All 10240 output rows were checked
against CPU reference, including finite-value checks; layout tests passed.

A750 driver 1.15.39183+4, M=10240/N=1/K=5120, split-K 1, warmup 5, repeats 30:

| Kernel | GEMM us | Total us | p95 us | Max absolute error |
|---|---:|---:|---:|---:|
| Packed SG8 |52.656|69.948|78.7606|1.31130219e-6|
| S2 baseline |421.927|442.1875|456.2814|4.76837158e-7|
| S2 XMX8 all |281.3015|306.823|324.604|4.76837158e-7|

All modes keep 11468800 resident weight bytes. XMX8 improves its own S2
baseline, but is 4.39x slower than SG8 in total time here. A wholesale XMX
port is therefore not justified by this experiment. Same-size layout/block
load ideas may still be evaluated separately in the native kernel.

The original probe is unchanged. Working copy:
`build-sycl-ptq1/probe-xmx8-q32/`. Final logs and binary/source hashes are in
`docs/ptq1-xmx8-evidence.txt`; local original-to-copy patches preserve the
comparison implementation. This synthetic repeated-weight probe does not
establish real-model TG or cold-memory performance.
