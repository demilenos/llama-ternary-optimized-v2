# Ternary Optimization Agent Guide

This file is an orientation map for coding agents and humans working on the ternary/PTQ1 optimization branch of this llama.cpp fork.

The goal is to **avoid rediscovering rejected experiments, avoid benchmark category errors, and preserve correctness and the A750 memory envelope while optimizing PTQ1 inference**.

## Read in this order

1. [ternary-llm-optimization-index.md](ternary-llm-optimization-index.md)
2. [sycl-ptq1-bonsai2.md](sycl-ptq1-bonsai2.md)
3. [sycl-ptq1-performance.md](sycl-ptq1-performance.md)
4. [negative-results-index.md](negative-results-index.md)
5. If changing layout/XMX/DPAS code: [ptq1-xmx8-evaluation.md](ptq1-xmx8-evaluation.md)
6. If changing representation/repacking: [lossless-ptq1-repack.md](lossless-ptq1-repack.md)

Then inspect recent commits before touching code. This repository intentionally records both accepted and rejected PTQ1 experiments in commit history.

## Non-negotiable invariants

### 1. Correctness before throughput

A faster standalone probe is only a candidate.

Production acceptance requires the relevant CPU-reference/backend operator tests and a real Bonsai2 path. Existing documents contain examples where a synthetic kernel improved substantially but regressed real-model TG.

Do not generalize a three-prompt smoke test into a model-quality claim.

### 2. Preserve the memory envelope

The Intel Arc A750 has 8 GB VRAM. The working constraint is approximately **6 GB maximum GPU-resident model weights**, leaving room for context, runtime state and scratch.

Avoid designs that require a second expanded/permuted resident copy of all ternary weights unless the memory cost is explicitly measured and justified.

### 3. A layout change is a whole-backend change

If device weight layout changes, audit at least:

- upload and partial/async transfer
- tensor views and copies
- canonical readback
- ordinary token-generation matmul/GEMV
- prefill
- fused gate/up or other fused consumers
- fallback paths
- operator tests
- real-model TG

The word-transpose experiment is the canonical example of why a standalone win is insufficient.

### 4. Compare like with like

Keep these categories separate:

- standalone event-timed probe
- backend repeated-op microbenchmark
- short `llama-bench` PP/TG
- server request latency/throughput
- long-context or sustained server behavior

Do not multiply a microkernel speedup directly into model tokens/s.

Prefer same-DLL, same-settings, reversed-order A/B comparisons when practical.

### 5. Environment flags are presence flags unless the code says otherwise

Several PTQ1 optimization switches are enabled by the variable being present. Setting a variable to `0` may still enable it.

For a true OFF run, **unset** a presence flag.

The accepted configuration and exact flags are documented in [sycl-ptq1-performance.md](sycl-ptq1-performance.md); do not copy an old flag set from commit history without checking the current document.

### 6. Keep evidence reproducible

For performance work, preserve:

- exact model / backend / device
- PP/TG dimensions
- batch and ubatch
- KV type
- sample count
- warmup policy
- binary/backend hashes where available
- git commit / dirty state
- correctness result
- whether graph/profiling/debug modes were enabled

Do not promote local artifact paths as public downloadable assets. Build outputs and model weights are intentionally excluded from Git.

## Current accepted direction

The repository's documented accepted path is the **canonical packed PTQ1/SG8-oriented production layout**, plus independently validated optimizations such as the current BF16 WG256 small-output GEMV path and other flags listed in the performance document.

A full XMX8 replacement is **not** the accepted backend. A word-transposed/local-reduction layout showed promising standalone timings but a real-model regression after integration and was rejected.

## Fast repository triage for agents

Before broad directory traversal, answer these questions:

```text
1. What changed recently?
2. Which file owns the PTQ1 hot path I am changing?
3. Is there already a rejected experiment with the same idea?
4. Which correctness test covers it?
5. Which benchmark is the acceptance gate?
6. Does it create another resident weight representation?
```

Useful searches:

```sh
git log --oneline --all --grep=PTQ1
git log --oneline --all --grep=XMX
git log --oneline --all --grep=rejected
git log --oneline -- docs/sycl-ptq1-performance.md
git log --oneline -- docs/ptq1-xmx8-evaluation.md

rg -n "PTQ1|GGML_SYCL_PTQ1|SG8|XMX|DPAS|DP4A" ggml docs scripts tests
rg -n "rejected|regression|correctness|same-DLL" docs
```

Prefer **delta-first investigation** over listing the entire upstream llama.cpp tree.

## Optimization acceptance loop

```text
hypothesis
  -> smallest isolated probe
  -> numerical/reference validation
  -> matched backend microbenchmark
  -> production integration
  -> full affected-path tests
  -> same-settings Bonsai2 A/B
  -> memory check
  -> document accept/reject result
```

A rejected result should remain discoverable. Record what improved, what regressed, the conditions, and why the production path was restored.

## Terminology for external discoverability

When documenting new work, pair internal names with public terminology:

- `PTQ1` + "packed ternary weights" / "ternary LLM quantization"
- `SG8` + "SYCL subgroup ternary dot-product"
- `XMX8` + "Intel XMX / DPAS low-bit matrix multiply"
- `word transpose` + "resident packed-weight permutation"
- `A750` + "Intel Arc Alchemist"
- `W1A8/W2A8` + "low-bit weights with 8-bit activations"

Commit messages should state the backend, representation and result when possible, for example:

```text
sycl/ptq1: reject word-transposed resident layout after TG regression
sycl/ptq1: validate packed ternary decoder on Arc A750
sycl/ptq1: evaluate XMX/DPAS adapter against SG8 baseline
```

This is not keyword stuffing. It is an explicit mapping between local vocabulary and terminology used by future developers, search engines and code-retrieval agents.
