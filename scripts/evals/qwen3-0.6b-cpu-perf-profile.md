# Qwen3-0.6B CPU profile

## Setup

- Model: `/home/sonic182/.cache/llama-qwen3/Qwen3-0.6B.Q4_K_M.gguf`
- CPU: AMD Ryzen 5 3550H, 4 cores / 8 threads, AVX2
- Build: host Release build, commit `95887577`, CPU backend
- Batch/ubatch: 512 / 512
- Prefill threads: 4; decode threads: 6
- `perf` access was enabled temporarily with `scripts/perf-enable.sh`.
- `perf stat -r 3` launched three separate `llama-bench -r 5` runs per counter set. Counts below are the average per `llama-bench` process and include model loading and warmup.

## Current unprofiled throughput

One clean `llama-bench -r 5` pass without `perf` was run after the instrumented trials.

| Workload | Threads | Current | Previous baseline | Change |
| --- | ---: | ---: | ---: | ---: |
| pp128 | 4 | 140.99 t/s | 139.67 t/s | +0.94% |
| pp512 | 4 | 130.53 t/s | 127.34 t/s | +2.51% |
| tg128 @ ctx512 | 6 | 31.18 t/s | 31.34 t/s | -0.51% |

These are consistent with the previous baseline; no code or model changed.

## Hardware counters

Main counters were collected without L1 events to avoid multiplexing. Each row is the `perf stat -r 3` average for one complete `llama-bench -r 5` process.

| Workload | Cycles | Instructions | IPC | Cache misses | Branch misses | Context switches | CPU migrations |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| pp128 | 69.36 B | 96.10 B | 1.39 | 104.0 M | 29.0 M | 1,458 | 50 |
| pp512 | 289.68 B | 389.66 B | 1.35 | 475.8 M | 97.3 M | 3,995 | 167 |
| tg128 @ ctx512 | 451.32 B | 397.34 B | 0.88 | 1.849 B | 95.0 M | 12,997 | 246 |

L1 counters were collected in separate runs:

| Workload | L1 loads | L1 load misses | Miss rate |
| --- | ---: | ---: | ---: |
| pp128 | 91.72 B | 1.539 B | 1.68% |
| pp512 | 371.81 B | 9.047 B | 2.43% |
| tg128 @ ctx512 | 192.02 B | 6.470 B | 3.37% |

`perf list` displayed L2/L3 event aliases, but `perf stat` could not resolve the aliases on this host. The aggregated DRAM-byte alias was also unavailable. A single Data Fabric channel event worked, but was excluded because it is system-wide and does not represent total memory traffic.

## Decode call graph

Captured with `perf record -F 99 -g --call-graph dwarf,8192` during `tg128 @ ctx512`, 6 threads. There were 14,768 samples and no lost samples. Self-sample share:

| Symbol | Share |
| --- | ---: |
| `ggml_gemv_q4_K_8x8_q8_K` | 23.04% |
| `ggml_vec_dot_q6_K_q8_K` | 15.37% |
| `ggml_compute_forward_flash_attn_ext` | 12.58% |
| `ggml_gemm_q4_K_8x8_q8_K` | 12.04% |
| Unresolved frame in `libgomp.so.1.0.0` | 9.97% |
| `ggml_vec_dot_f16` (next symbol) | 9.86% |

The first five entries account for 73.0% of self samples. The unresolved OpenMP frame prevents a complete source-level top-five attribution. In the call tree, the graph's major operation families are repacked Q4_K matmul (37.1%), flash attention (36.6%), and other matmul work (17.7%). These inclusive operation totals contain the kernel samples above and must not be added to them.

Source locations for the main named kernels:

- `ggml_gemv_q4_K_8x8_q8_K`: `ggml/src/ggml-cpu/arch/x86/repack.cpp:1464`
- `ggml_vec_dot_q6_K_q8_K`: `ggml/src/ggml-cpu/arch/x86/quants.c:2426`
- `ggml_compute_forward_flash_attn_ext`: `ggml/src/ggml-cpu/ops.cpp:9348`

The raw profile is at `/tmp/qwen3-0.6b-tg128.perf.data`; it is a 119 MB temporary file. Review it with `perf report -i /tmp/qwen3-0.6b-tg128.perf.data` before that file is removed.

## Generated x86 code inspection

The inspected library is `/home/sonic182/.cache/llama-qwen3-build/bin/libggml-cpu.so.0`. The host build uses Release, `GGML_NATIVE=ON`, `-O3`, and `-march=native`; the generated hot paths already use AVX2 integer unpack/multiply-add instructions. `perf annotate` has symbol-level samples, but the library has no DWARF line information.

The Q4_K GEMV implementation in `ggml/src/ggml-cpu/arch/x86/repack.cpp:1464` is already an eight-output interleaved kernel. It reuses each Q8_K activation block across eight Q4_K columns, so adding a separate generic multi-row kernel would duplicate the current design.

Its generated function reserves a 0x268-byte (616-byte) stack frame and saves/reloads many YMM temporaries inside the vector loop. The source has no explicit YMM scratch array of that size, so this is strong evidence of compiler spills from register pressure. Stack-resident vector operands receive samples in the profile, including 2.43% at one `vpand` and 1.36% at one `vpaddd`. The rest of the hot instructions are spread across packed-weight loads, nibble unpack/shuffle, and integer multiply-add; there is no single instruction substitution that explains the cost.

The Q6_K dot kernel in `ggml/src/ggml-cpu/arch/x86/quants.c:2426` has a much smaller 0x28-byte frame. Its loop still reloads vector masks from stack and spends work unpacking 6-bit values, shuffling scales, and performing `vpmaddubsw`/`vpmaddwd`. It is a secondary target after Q4_K GEMV.

### Next kernel hypothesis

Prototype a static AVX2 Q4_K fast path that reduces the live vector set in the existing eight-output loop, then measure the kernel and full `tg128 @ ctx512`/`pp512` workloads. Keep the current repacked path as the fallback. The hypothesis is that fewer spill/reload operations can improve end-to-end decode; it is not yet proven. Do not start with AsmJit: the current build already targets the host ISA, and JIT will only help if a static variant first demonstrates that shape specialization or a lower-pressure schedule is valuable.

## Assessment and next step

Decode is dominated by CPU kernel work across repacked Q4_K matrix-vector/matrix kernels, Q6_K dot products, and attention. Its aggregate IPC is lower than prefill (0.88 versus 1.35-1.39), and its L1 miss rate is higher. This supports investigating both instruction efficiency and cache behavior, but does not establish whether decode is limited by compute, cache latency, or DRAM bandwidth. More precise L2/L3 and DRAM counters are unavailable through the current perf event aliases.

Do not change a kernel from this profile alone. Next, investigate the thread/SMT and affinity phase, then profile or benchmark a narrowly scoped Q4_K multi-row hypothesis. Preserve the current throughput as the control; profiling counters alone are not a speed result.
