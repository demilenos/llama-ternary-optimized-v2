# Ternary LLM Optimization Research Index

This repository studies practical **ternary / ultra-low-bit LLM inference** on Intel Arc GPUs, with Bonsai2 PTQ1 as the main model path and the llama.cpp SYCL backend as the integration target.

Search terms and adjacent terminology: **ternary LLM, 1.58-bit-class LLM, post-training ternarization, packed ternary weights, low-bit GEMV, low-bit GEMM, PTQ1, Q2_0, DP4A, Intel XMX, DPAS, SYCL, oneAPI, Intel Arc A750, Alchemist, llama.cpp Intel GPU, local LLM inference**.

> PTQ1 is the repository's packed ternary representation. Do not assume that "1.58-bit" describes its exact serialized bits-per-weight; use that phrase as an adjacent research/search term.

## Canonical entry points

| Question | Start here |
| --- | --- |
| How do I build and run Bonsai2 PTQ1 on Intel Arc? | [sycl-ptq1-bonsai2.md](sycl-ptq1-bonsai2.md) |
| What performance is currently validated? | [sycl-ptq1-performance.md](sycl-ptq1-performance.md) |
| What happened with XMX/DPAS and word-transposed layouts? | [ptq1-xmx8-evaluation.md](ptq1-xmx8-evaluation.md) |
| Can PTQ1 be repacked losslessly to Q2_0? | [lossless-ptq1-repack.md](lossless-ptq1-repack.md) |
| What should a coding agent read before changing this fork? | [TERNARY_AGENT_GUIDE.md](TERNARY_AGENT_GUIDE.md) |
| Which attractive approaches were measured and rejected? | [negative-results-index.md](negative-results-index.md) |

## Current research envelope

- **Primary GPU:** Intel Arc A750 8 GB.
- **Backend:** llama.cpp SYCL / Intel oneAPI.
- **Primary model path:** Bonsai2 PTQ1 ternary weights.
- **Hard deployment constraint:** keep GPU-resident model weights within roughly 6 GB so context, recurrent state, runtime allocations and scratch still fit.
- **Correctness policy:** kernel speedups are not accepted from a microbenchmark alone; backend operator tests and real-model measurements are required.
- **Measurement policy:** distinguish short `llama-bench` TG/PP measurements, backend microbenchmarks, standalone probes and sustained/full-context server behavior. They are not interchangeable.

The currently documented accepted A750 configuration reaches about **23.5 tokens/s at TG128** in the matched BF16-WG256 experiment. The historical 30 tokens/s target remains unmet. Treat the performance document as canonical because accepted flags and evidence can change.

## Mental model of the PTQ1 path

A useful high-level path for repository navigation is:

```text
GGUF PTQ1 tensors
    -> packed resident GPU weights
    -> PTQ1 trit decode / packed signed representation
    -> Q8_1 activation path
    -> SYCL subgroup dot-product / DP4A-style work
    -> GEMV / fused FFN consumers
    -> model output
```

Prefill and fallback paths may use different conversions or kernels. Any resident-layout change must therefore be checked against **readback, copies/views, prefill, ordinary matmul, fused FFN paths and correctness tests**, not only the hot token-generation kernel.

## Search vocabulary bridge

Internal names should be written alongside public terminology so humans and retrieval agents can find the work:

| Repository term | Also search / mention |
| --- | --- |
| PTQ1 | ternary quantization, packed ternary weights, ultra-low-bit weights |
| SG8 | subgroup-8 ternary decode/dot path |
| XMX8 | Intel XMX / DPAS low-bit matrix path |
| word transpose | packed-weight permutation, resident weight layout |
| full-block decoder | packed ternary block decode |
| Q8_1 activation | INT8-like quantized activation blocks, activation scaling |
| BF16 WG256 | BF16 GEMV workgroup-per-row kernel |
| A750 | Intel Arc A750, Alchemist GPU |
| SYCL | Intel oneAPI SYCL backend |
| lossless Q2 repack | exact ternary-value storage conversion |

## Evidence over narrative

When a summary here conflicts with a benchmark/evidence document, prefer the dated evidence document. Local `build-sycl-ptq1/` artifacts are intentionally not committed; the repository keeps source changes, reproducible commands, hashes and compact evidence records instead.

Negative results are first-class research output. Before implementing a new packed layout, split-K scheme, XMX adapter, speculative path or decoder variant, read [negative-results-index.md](negative-results-index.md) and search recent commits for related terminology.
