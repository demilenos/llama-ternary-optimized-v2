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
