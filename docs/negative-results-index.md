# PTQ1 / Ternary Negative Results Index

Negative results are part of the optimization record. This page makes expensive dead ends searchable so later developers and coding agents do not repeat them without a new hypothesis.

**Canonical evidence remains in the linked documents.** Numbers below describe the measured conditions in those documents, not universal hardware claims.

## XMX8 / DPAS standalone path: faster than its own baseline, slower than SG8

Evidence: [ptq1-xmx8-evaluation.md](ptq1-xmx8-evaluation.md)

At A750 M=10240, N=1, K=5120 in the same Q32 comparison harness:

- packed SG8 total: about 69.95 us
- S2 XMX8 `all` total: about 306.82 us
- XMX8 improved its own S2 baseline, but remained roughly **4.39x slower than SG8** in that comparison.

Conclusion recorded by the project: a wholesale XMX port was not justified by this experiment. XMX-derived layout/block-load ideas may still be tested independently.

**Do not infer that XMX is intrinsically bad for ternary inference.** The rejected result is for this representation, adapter and harness.

## Native word-transpose + local reduction: synthetic win, production loss

Evidence: [ptq1-xmx8-evaluation.md](ptq1-xmx8-evaluation.md)

The four-subgroup local-reduction probe improved main-kernel time across several tested model shapes while keeping one resident weight copy.

After production integration, the same-size resident layout passed transfer/correctness checks but real-model TG128 regressed:

- canonical SG8: about 21.99 t/s
- word transpose/local reduction: about 18.25 t/s
- measured regression: about **17%**

Backend M10240/N1/K5120 also regressed. Compile-time separation and an activation-storage variant did not close the gap.

Conclusion: **rejected; accepted DLL restored**.

This is the repository's strongest warning against promoting a standalone low-bit kernel result without a production traffic measurement.

## Canonical-layout local reduction: slower in the measured shapes

Evidence: [ptq1-xmx8-evaluation.md](ptq1-xmx8-evaluation.md)

A control that applied local reduction without the word-transposed layout was slower on all four documented model shapes.

Conclusion: no accepted production change.

## Workgroup Y=4 / Y=8 on the one-lane full-block path: no TG gain

Evidence: [sycl-ptq1-performance.md](sycl-ptq1-performance.md)

The one-lane PTQ1 token-generation path was tested with local workgroup Y=4 and Y=8 against selected Y=1.

- Y=1: about 18.05 t/s in the documented historical run
- Y=4: about 17.99 t/s
- Y=8: about 17.93 t/s

Conclusion: source restored to Y=1 for that stage of development.

These are historical measurements; later accepted optimizations changed absolute throughput.

## Lossless PTQ1 -> Q2_0 repack: exact values, slower existing runtime

Evidence: [lossless-ptq1-repack.md](lossless-ptq1-repack.md)

The conversion preserves every ternary value and scale bits while changing storage from a 128-weight PTQ1 block to two Q2_0 blocks.

In the documented initial SYCL comparison:

- original PTQ1 TG128: about 19.92 t/s
- lossless Q2_0 TG128: about 8.43 t/s

An optional full-block Q2 kernel improved the Q2 result to about 11.43 t/s, still below PTQ1.

Conclusion: the converter is useful for representation/runtime experiments, but the existing Q2 path is not the preferred A750 performance path.

## Cache-history ngram speculation: regression on the fixed tool-history workload

Evidence: [sycl-ptq1-performance.md](sycl-ptq1-performance.md)

A matched 12-request fixed tool-history test showed lower late-turn TG with cache-history speculation than with no speculation. A draft-length configuration also produced a large shared-GPU-memory increase; reducing draft length removed most of that increase but did not remove the throughput regression.

Conclusion recorded in the performance document: local speculation is disabled for this workload. The exact allocation attribution and an older unrelated correctness incident remain unresolved.

Do not generalize this to every speculative-decoding workload.

## How to reuse a negative result

A rejected path is worth reopening when at least one material condition changes, for example:

- a different data layout eliminates the measured overhead
- a different activation representation removes conversion cost
- a compiler/driver change alters code generation
- dispatch/reduction can be fused
- the real-model bottleneck moved after another optimization
- a new correctness-preserving hardware primitive changes the trade-off

When reopening one, explicitly cite the old failure mode and design the new experiment to distinguish the new hypothesis from the rejected implementation.
