# Bonsai2 severe corruption and performance investigation

Goal: fix repeatable majority-output corruption, verify reported performance bottlenecks, and retain only measured improvements.

User acceptance: isolated multilingual words or malformed characters are tolerated when task meaning and progress survive. Repeated meaningless output, loss of usable answers, and stalled generation are failures. GPU weights must stay below 6 GB on the 8 GB Arc A750. Keep a visible server on port9931 when experiments finish.

Approach: preserve failing responses and probabilities; reproduce healthy-to-broken transitions before changing kernels. Parent owns GPU experiments; bounded Luna agents audit independent source paths. Runtime changes are single-variable controls. A short successful answer is not full correctness validation.

- [x] Capture current severe failure with cache_prompt false and log probabilities.
- [x] Check thinking-off on the same poisoned process (also repeats BBBB).
- [x] Restart unchanged configuration; test short/long/short request sequence.
- [ ] Isolate speculation, custom fusion, and recurrent/cache behavior against a known-good control. Preserve exact requests and build hashes.
- [ ] Add a failing regression to existing test infrastructure for the evidenced root cause, implement minimal fix, and rerun the reproduction plus state reuse/rollback cases.
- [x] Audit supplied PTQ1 FP32 prefill and long-context FA claims against source and actual build settings. Distinguish Q5 runtime from Q8 estimates.
- [ ] After correctness, measure fresh PP128/512/1024 and fixed-depth decode without profiling overhead. Probe only an evidenced bottleneck; accept a change only with numerical and end-to-end evidence.
- [ ] Commit independently verified units with concise evidence; restore visible server and report remaining limitations.

Review focus: long prefill followed by an unrelated short request; cached prefix edits and rollback; speculative partial acceptance; thinking/tool history; VRAM pressure and queue stalls. The original non-speculative BBBB incident must not be dismissed by disabling ngram alone.

Evidence directory: build-sycl-ptq1/correctness-incident/. Current baseline source be206a8. User-supplied performance analysis is an investigation lead, not benchmark proof.

User steering: prioritize the two low-cost high-impact performance probes before deeper correctness work. Both remain individually switchable; packed MMQ and chunked GDN are deferred. Fresh short/28931-token prompt/short succeeded unchanged; persistent BBBB was captured before restart in both thinking and non-thinking modes.

Performance priority outcome: accepted opt-in PTQ1 large-N FP16 after 40/40 CPU-reference cases in both modes and matched/reverse-order PP measurements. PP512 improved from about 206 to 447 t/s. Rejected FA256 after no measurable fixed-depth TG improvement. Persistent corruption remains unresolved; clean cache/tool-history replay is being compared with speculation on/off.
