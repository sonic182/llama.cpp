# Qwen3-0.6B CPU performance plan

## Goal and rules

Improve CPU-only Qwen3-0.6B Q4_K_M inference on the Ryzen 5 3550H, prioritizing steady-state decode while protecting prompt processing and model quality.

- Builds, profiles, correctness checks, and benchmarks run on the host, not in Docker.
- Change one major variable at a time. Preserve a reproducible baseline and record compiler/build options, model hash, thread settings, affinity, prompt/context sizes, and repetitions.
- A candidate is worth keeping when the target workload improves by about 2% or more, the result repeats across paired runs, and other important workloads regress by no more than 1%.
- Treat changes below 1% as noise until stronger measurements show otherwise. Preserve output quality; any numerical or algorithmic change needs a quality check.
- Stop and re-profile when a phase shows that its assumed bottleneck is not material.

## Existing evidence

- [x] CPU-only host baseline recorded for Qwen3-0.6B Q4_K_M. Best observed: pp128 139.67 t/s at 4 threads; pp512 127.34 t/s at 4 threads; tg128 at context 512 31.34 t/s at 6 threads.
- [x] A two-accumulator Q4_K AVX2 experiment passed 110 existing Q4_K/Q6_K MUL_MAT correctness cases, but its Q4_K microbenchmark was slower and decode regressed 2.23%. The source change was reverted.
- [x] Static FFN-skip calibration tested every layer individually. Even the least sensitive layer had +10.1% PPL and 85.3% top-token agreement; the temporary selector was removed.
- [x] OpenBLAS was slower for the tested MiniLM embedding workload. This does not rule out benefits for other workloads, but is not a reason to add BLAS to this Qwen CPU build.

Evidence files:

- `scripts/evals/qwen3-0.6b-cpu-kernel-experiment.md`
- `scripts/evals/qwen3-0.6b-ffn-skip-calibration.md`
- `scripts/evals/qwen3-0.6b-cpu-baseline-prefill.json`
- `scripts/evals/qwen3-0.6b-cpu-baseline-decode.json`

## Phase 1 - Profile the real bottleneck (P0)

- [x] Verify `perf` is installed and hardware-counter access is available. `perf` 6.12.107 is installed, but this runtime blocks CPU events (`perf_event_paranoid=3`, no `CAP_PERFMON`); profiling remains pending access to PMU counters.
- [x] Profile pp128, pp512, and tg128 at context 512 separately using the same model/build and baseline thread settings. Results: `scripts/evals/qwen3-0.6b-cpu-perf-profile.md`.
- [x] Run repeated `perf stat` trials for available counters: cycles, instructions, IPC, cache misses, branch misses, context switches, and CPU migrations. L1 events worked; L2/L3 and aggregate DRAM aliases were listed but unusable through `perf stat` on this host.
- [x] Record a call-graph profile for tg128 and identify the top five samples and their combined share. One OpenMP frame (9.97%) remains unresolved; the first five account for 73.0%.
- [ ] Classify the decode limit as primarily compute, cache/memory, frontend, or scheduling/thread overhead. Current evidence shows kernel-heavy execution, lower IPC, and more L1 misses in decode, but cannot separate compute from cache latency or DRAM bandwidth.
- [ ] Do not begin kernel changes until this profile identifies a material target.

## Phase 2 - Thread count, affinity, and SMT

- [ ] Record the logical CPU to physical-core/SMT-sibling map with host topology tools. Do not assume CPU numbering reflects core topology.
- [x] Initial unpinned baseline sweep already covers 1, 2, 4, 6, and 8 threads.
- [ ] Repeat 4, 6, and 8 threads under controlled warmup/paired runs, and add 3, 5, and 7 threads for pp and decode.
- [ ] Compare unpinned execution with affinity layouts for physical cores first, then physical cores plus selected SMT siblings, then all logical CPUs.
- [ ] Capture tok/s, perf counters, context switches, migrations, and affinity masks for every result.
- [ ] Repeat baseline and candidate runs in alternating order to reduce thermal/frequency drift.
- [ ] Keep affinity changes only if they meet the improvement gate and do not hurt other target workloads by more than 1%.

## Phase 3 - Compare weight quantizations

- [ ] Compare Q4_K_M with Q4_0, Q5_K_M, Q6_K, and Q8_0 GGUFs of the same Qwen3-0.6B model. Download only missing variants and record each model hash.
- [ ] Measure pp128, pp512, tg128 at context 512, and peak RSS with the same build, batch settings, and selected thread/affinity configurations.
- [ ] Compare quality on the same corpus and tokenizer using PPL and logit/KL metrics. Prefer an F16 reference if practical; otherwise explicitly report that results are relative to Q4_K_M.
- [ ] Select a speed/quality candidate only from measured results. Do not assume Q4_0 or any other format is faster on Zen+.

## Phase 4 - Inspect generated code for hot functions

- [x] Use `perf report` and `perf annotate` on the decode profile before inspecting assembly.
- [x] Inspect compiler-generated x86 code for the hottest relevant functions with `objdump -d -Mintel`.
- [x] Check for spills, repeated conversions/unpacking, unnecessary loads/stores, branchy inner loops, and weak instruction scheduling. The Q4_K 8-output GEMV has substantial stack traffic consistent with register spills; Q6_K has a smaller frame and repeated unpack/shuffle work.
- [x] Record a profile-backed hypothesis before changing intrinsics or assembly: reduce live vectors and spills in the existing Q4_K 8-output path with a static AVX2 fast path, retaining the current path as fallback.
- [ ] Prefer compiler intrinsics first. Consider handwritten assembly only if the generated code has a clear defect and a controlled experiment can isolate it.

## Phase 5 - Multi-row Q4_K x Q8_K kernel

- [ ] Proceed only if profiling shows this dot-product path materially contributes to decode time for the selected quantization.
- [ ] Keep the prototype in `ggml/src/ggml-cpu/arch/x86/quants.c`; review dispatch in `ggml/src/ggml-cpu/ggml-cpu.c` and retain the generic fallback in `ggml/src/ggml-cpu/quants.c`.
- [ ] Prototype 2-row and 4-row variants that reuse activation-side work across output rows; begin without changing GGUF weight layout.
- [ ] Keep the generic implementation and other architecture paths unchanged. Add/use an x86 dispatch path only for supported CPU features.
- [ ] Validate `nrc`/row-stride behavior, tails, alignment, and supported row sizes with existing `test-backend-ops` coverage; do not add test files without approval.
- [ ] Run the same end-to-end pp/decode benchmark and quality comparison as the baseline. Keep only variants meeting the improvement/regression gate.

## Phase 6 - Quantization/dequantization and activation reuse

- [ ] Use profiling to determine whether Q8_K activation quantization or weight unpacking is a meaningful share of runtime.
- [ ] If it is, measure whether quantized activations are recomputed or can be safely reused across the relevant operations.
- [ ] Test fusion of activation quantization with the consuming matmul only when it removes real data movement or work.
- [ ] Avoid introducing temporary dequantized tensors or longer-lived buffers unless total memory traffic and end-to-end latency improve.
- [ ] Re-run correctness and numerical-quality checks after any change to quantization or reuse semantics.

## Phase 7 - Shape specialization

- [ ] Specialize only shapes shown to dominate the profiled Qwen workload.
- [ ] First implement a small static C/C++/intrinsics fast path with the existing generic fallback; avoid Qwen graph/model-specific branching for CPU math kernels.
- [ ] Benchmark each specialized shape against the same generic path and check whether the benefit survives full inference.
- [ ] Avoid adding shape-specific dispatch when it merely duplicates generic code or does not produce a repeatable end-to-end gain.

## Phase 8 - Fusion and approximate math

- [ ] Consider adjacent-operation fusion only where the profile shows meaningful intermediate-tensor traffic or dispatch overhead.
- [ ] Consider approximate SiLU, exp, softmax, RMSNorm, reciprocal, or rsqrt only if the operation accounts for at least 2-3% of end-to-end runtime.
- [ ] Measure speed, numerical error, PPL/KL, and representative output behavior for each approximation.
- [ ] Reject changes whose quality cost exceeds the user's tolerance or whose speed gain misses the improvement gate.

## Phase 9 - KV cache and context-length scaling

- [ ] Benchmark decode at context lengths 512, 2k, 4k, and 8k if those lengths match intended usage.
- [ ] Build a baseline/candidate context-length curve before changing attention or KV storage.
- [ ] If long-context results show a memory/bandwidth bottleneck, compare F16, Q8, and Q4 KV cache options for latency, memory, and quality.
- [ ] Do not prioritize KV optimization based only on the current context-512 result.

## Phase 10 - Cache topology and optional memory experiments

- [ ] Use the Phase 2 topology map to confirm whether the best affinity keeps workers near shared caches; NUMA tuning is not expected to matter on this 4-core laptop CPU.
- [ ] Try software prefetch only if profiling shows a suitable streaming pattern. Compare no prefetch with small distances such as 1, 2, and 4 blocks; the hardware prefetcher may already be better.
- [ ] Treat transparent/huge pages as a low-priority experiment. Measure TLB events and end-to-end performance, and keep it only if the result is repeatable.

## Phase 11 - JIT / AsmJit gate

- [ ] Do not add AsmJit yet. First demonstrate that a static shape-specialized C++/intrinsics kernel wins repeatably.
- [ ] Revisit JIT only if there are a few stable shapes and CPU variants where generated code can remove meaningful generic overhead.
- [ ] If justified, isolate the JIT implementation behind an optional x86 build feature, retain the generic fallback, cache generated code by shape and ISA, and measure compile/startup cost separately from steady-state inference.
- [ ] Compare the JIT path against the best static kernel with correctness, quality, memory, startup, prefill, and decode measurements.

## Completion criteria

- [ ] The chosen change reaches at least about 2% repeatable improvement in its target workload.
- [ ] No important measured workload regresses by more than 1%, unless the user explicitly accepts that tradeoff.
- [ ] Correctness and quality results are recorded alongside performance, build flags, model hash, CPU, thread count, and affinity.
- [ ] Rejected experiments and their measurements remain documented; unhelpful code changes are reverted.
