---
name: llamacpp-dev
description: >
  Develop, analyze, benchmark, and optimize llama.cpp for CPU inference.
  Use this skill when working on CPU inference performance, GGML/GGUF kernels,
  quantization, threading, scheduling, memory usage, cache locality, inference
  latency, throughput, prompt processing, token generation, or server-side
  overhead related to llama.cpp.
---

# llama.cpp CPU Inference Development

Optimize llama.cpp for fast and memory-efficient CPU inference while preserving acceptable model quality.

The primary goal is not to make individual functions faster in isolation. The goal is to improve real end-to-end inference.

## Golden Rules

### 1. Do less work

Before optimizing an operation, determine whether the operation can be:

1. eliminated,
2. skipped,
3. approximated,
4. reduced in precision,
5. reused,
6. fused with adjacent work,
7. computed incrementally,
8. terminated early,
9. moved out of the hot path.

Only optimize execution speed after considering whether the work needs to exist at all.

Prefer:

> eliminate → reduce → move less data → improve locality → fuse → parallelize → vectorize → hand-optimize

Do not start with assembly or intrinsics unless profiling shows that the relevant kernel materially affects end-to-end performance.

---

### 2. Keep the hot path hot

Token generation hot paths should minimize:

- allocations,
- deallocations,
- syscalls,
- locks,
- atomics,
- synchronization barriers,
- thread wakeups,
- CPU migrations,
- indirect calls,
- unpredictable branches,
- format conversions,
- temporary buffers,
- unnecessary copies,
- cache misses,
- memory traffic.

Preallocate and reuse resources whenever practical.

Do not introduce general-purpose abstractions into a hot path without measuring their cost.

---

### 3. Context switching and scheduling are costs

Avoid unnecessary interaction with the operating-system scheduler.

Prefer:

- persistent workers,
- stable thread pools,
- long-lived execution resources,
- CPU affinity when useful,
- user-space coordination,
- reused buffers,
- predictable ownership of work.

Avoid:

- frequent thread creation,
- sleeping and waking workers unnecessarily,
- excessive producer/consumer handoffs,
- unnecessary kernel transitions,
- unnecessary blocking waits.

More concurrency is not automatically better.

Measure context switches, CPU migrations, wakeups, and scheduler overhead when changing execution architecture.

---

### 4. Data locality beats theoretical parallelism

Prefer cache locality and predictable memory access over maximum CPU utilization.

Consider:

- cache hierarchy,
- cache-line alignment,
- working-set size,
- false sharing,
- NUMA topology,
- physical cores versus SMT,
- memory bandwidth,
- memory access patterns.

Using fewer threads is acceptable when it improves inference speed or latency.

Never assume that all logical CPUs should be active.

---

### 5. Reuse memory aggressively

Memory efficiency is a first-class performance concern.

Prefer:

- preallocated buffers,
- reusable scratch memory,
- arenas,
- stable tensor layouts,
- thread-local storage where appropriate,
- carefully partitioned shared buffers.

Avoid per-token allocation whenever possible.

Shared memory or shared buffers are desirable only when they reduce total work or memory movement without introducing:

- contention,
- false sharing,
- extra synchronization,
- ownership complexity,
- cache-line bouncing.

Measure both memory consumption and execution performance.

---

### 6. Move less data

Memory movement may be more expensive than arithmetic.

Avoid unnecessary sequences such as:

`quantized → dequantized temporary → transformed → requantized`

when equivalent work can happen directly on the compact representation.

Prefer kernels that consume data in the representation in which it is already stored.

Consider tensor layout and data representation part of the algorithm, not merely implementation details.

For a zero-copy proposal, trace each copy and distinguish model-load or repack work from per-token work. Check whether the source layout, alignment, ownership, and lifetime allow the kernel to consume that buffer directly. A memory-mapped model may still be copied into a backend-specific layout.

Compare load time, peak and steady-state RSS, prompt processing, and token generation. Removing a copy can also remove a layout that makes the hot kernel faster.

---

### 7. Fusion is preferred when it reduces total work

Consider fusing adjacent operations when fusion reduces:

- memory traffic,
- intermediate tensors,
- dispatch overhead,
- synchronization,
- cache misses.

Do not fuse operations merely to reduce function-call count.

Fusion is beneficial only when end-to-end measurements demonstrate an improvement and maintainability remains reasonable.

---

### 8. Exploit the target hardware explicitly

Optimize for the capabilities of the hardware currently being tested.

Inspect relevant characteristics such as:

- ISA extensions,
- SIMD width,
- physical core count,
- SMT topology,
- cache hierarchy,
- memory bandwidth,
- NUMA layout,
- available accelerators,
- CPU-specific instructions.

Portable implementations and fallbacks are desirable, but they must not prevent architecture-specific fast paths.

Do not hard-code assumptions from one development machine into generic code unless the implementation is explicitly hardware-specific.

---

### 9. Optimize batch=1 autoregressive inference explicitly

When the workload is interactive LLM inference, treat steady-state token generation as a distinct workload.

Do not assume optimizations for:

- large matrix multiplication,
- prompt processing,
- large batches,
- training,
- throughput-oriented workloads

also improve batch=1 token generation.

Measure prompt processing and token generation separately.

A faster prompt-processing benchmark does not imply faster interactive inference.

---

### 10. Performance may trade numerical precision, but quality loss must be measured

Performance has priority over exact numerical equivalence.

Potentially acceptable techniques include:

- quantization,
- lower precision,
- approximate arithmetic,
- approximate transcendental functions,
- early termination,
- reduced computation,
- alternative algorithms.

However, an optimization that changes model outputs is incomplete until its quality impact is measured.

Never claim success using only a speed benchmark when numerical behavior changes.

Report:

- performance improvement,
- memory impact,
- quality impact,
- relevant numerical differences.

Small quality regressions may be acceptable when they provide meaningful performance benefits.

Large regressions require proportionally stronger justification.

---

### 11. Separate the compute plane from the control and I/O plane

Inference workers should not perform avoidable:

- networking,
- HTTP processing,
- JSON parsing,
- logging,
- disk I/O,
- blocking I/O,
- request bookkeeping.

Keep network and control workloads from disturbing compute-thread locality.

For I/O-heavy components, prefer event-driven designs when they demonstrably reduce:

- thread proliferation,
- wakeups,
- context switches,
- memory consumption,
- kernel scheduling overhead.

Do not introduce an event loop into CPU compute paths simply because event-driven I/O is desirable.

---

### 12. Persistent resources are preferable to repeated scheduling

Where appropriate, prefer:

`persistent worker → reusable state → repeated work`

over:

`create → schedule → allocate → execute → synchronize → destroy`

This applies to:

- threads,
- buffers,
- execution contexts,
- request resources,
- temporary memory,
- synchronization primitives.

Persistent resources must still be justified by memory cost.

---

### 13. Profile before optimizing

Do not guess where llama.cpp spends its time.

Use profiling and benchmarking to identify actual bottlenecks.

Useful tools may include:

- `llama-bench`,
- `perf`,
- flamegraphs,
- hardware performance counters,
- allocation profiling,
- cache-miss measurements,
- context-switch measurements,
- CPU migration measurements.

Prioritize changes according to their contribution to total inference time.

A function consuming 1% of runtime cannot provide a meaningful end-to-end speedup unless the optimization changes broader system behavior.

---

### 14. End-to-end performance is the source of truth

Never accept a microbenchmark improvement as sufficient evidence.

A kernel can become substantially faster while complete inference remains unchanged or becomes slower.

Every significant optimization should eventually be measured with representative end-to-end inference.

Track separately when relevant:

- model loading,
- prompt processing,
- time to first token,
- steady-state token generation,
- inter-token latency,
- requests per second,
- peak RSS,
- steady-state RSS.

---

### 15. Benchmark steady state separately from startup

Do not mix:

- model loading,
- page faults,
- initialization,
- warmup,
- JIT/setup costs,
- prompt processing,
- token generation.

Warm up the workload when measuring steady-state performance.

If startup performance is itself the optimization target, measure it separately.

---

### 16. Every optimization must be comparable

Performance work should make A/B comparison easy.

Whenever practical:

- retain a baseline,
- change one major variable at a time,
- record build configuration,
- record model and quantization,
- record context size,
- record thread configuration,
- record hardware,
- record compiler,
- record relevant runtime options.

Prefer small, independently benchmarkable changes over large rewrites.

---

## Primary Metrics

For interactive CPU inference, prioritize:

1. steady-state token generation speed,
2. inter-token latency,
3. time to first token,
4. prompt-processing speed,
5. peak memory usage,
6. steady-state memory usage.

Also track when relevant:

- CPU utilization,
- memory bandwidth,
- cache misses,
- branch misses,
- context switches,
- CPU migrations,
- instructions per cycle,
- energy or power consumption.

Do not maximize one metric blindly at the expense of the workload as a whole.

---

## Performance Investigation Workflow

When asked to improve performance:

### 1. Establish the workload

Identify:

- model,
- quantization,
- context size,
- batch size,
- prompt size,
- expected output size,
- thread count,
- target hardware,
- CPU-only versus heterogeneous execution.

Determine whether the priority is:

- prompt processing,
- token generation,
- latency,
- throughput,
- memory,
- concurrency.

### 2. Establish a baseline

Record reproducible baseline measurements before modifying code.

### 3. Profile

Find the dominant contributors to runtime or memory use.

### 4. Form a hypothesis

Explain why the proposed change should improve the measured bottleneck.

Examples:

- fewer bytes loaded,
- fewer instructions,
- fewer cache misses,
- less synchronization,
- fewer allocations,
- improved SIMD utilization,
- fewer context switches.

### 5. Implement the smallest useful experiment

Prefer isolated experiments over architectural rewrites.

### 6. Benchmark

Compare against the same baseline under the same conditions.

### 7. Validate quality

If numerical behavior changed, run the appropriate quality evaluation.

### 8. Inspect secondary effects

Check whether the change affects:

- memory,
- prompt processing,
- latency,
- concurrency,
- other quantizations,
- other architectures.

### 9. Keep or revert

Keep an optimization only when its benefit is demonstrated and its trade-offs are understood.

---

## Before Writing a New Kernel

Ask, in this order:

1. Can this computation be removed?
2. Can less of it be computed?
3. Can an existing result be reused?
4. Can precision be reduced?
5. Can memory movement be reduced?
6. Can the representation be changed?
7. Can adjacent operations be fused?
8. Can locality be improved?
9. Can scheduling or synchronization be reduced?
10. Can existing compiler-generated SIMD solve it?
11. Can explicit intrinsics improve it?
12. Is handwritten assembly justified?

Handwritten assembly should be one of the last options, not the first.

---

## CPU Concurrency Rules

When modifying threading:

- benchmark physical-core counts separately from SMT,
- do not assume more threads improve performance,
- avoid oversubscription,
- inspect affinity,
- inspect migrations,
- avoid unnecessary barriers,
- avoid false sharing,
- consider separate configurations for prompt processing and token generation.

Prefer stable ownership of cores when it improves locality.

---

## Server Rules

Changes to `llama-server` must distinguish inference performance from networking performance.

An HTTP/server optimization should be evaluated primarily using:

- server CPU overhead,
- concurrency,
- memory per connection,
- requests per second,
- context switches,
- syscalls,
- wakeups,
- latency added outside inference.

Do not claim improved inference because the HTTP layer is faster.

Conversely, do not dismiss server optimizations merely because token generation speed remains unchanged when server scalability is the target.

---

## Quality Guardrail

When an optimization can change output quality, compare it against an unchanged baseline.

Prefer deterministic or repeatable evaluation where possible.

Record enough information to answer:

- How much faster is it?
- How much memory does it save or consume?
- What quality changed?
- Under which workloads?
- On which hardware?

Avoid vague conclusions such as:

- "seems faster",
- "looks equivalent",
- "quality appears fine".

Use measurements.

---

## Decision Principle

When choosing between implementations, prefer the one that causes the CPU to:

> do less work, move less data, schedule less often, preserve locality, and reuse more state.

Performance improvements are valuable only when they survive representative end-to-end inference benchmarks.
