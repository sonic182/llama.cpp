# Qwen3-0.6B Q4_K GEMV streaming experiment

## Plan and result

- [x] Establish a host baseline for `pp512` and `tg128 @ ctx512`.
- [x] Stream pairs of 32-byte Q4_K weight vectors through nibble extraction and both integer accumulators.
- [x] Keep the previous AVX2 kernel as the fallback for `nr != 1` and for builds without AVX2.
- [x] Run CPU backend correctness checks, the targeted operator microbenchmark, and end-to-end benchmarks.
- [x] Inspect generated x86 code for stack-resident vector operands.
- [x] Repeat the comparison with alternating control and candidate runs.
- [x] Reject and remove the candidate because prefill regressed beyond the 1% limit and decode did not improve.

The streamed implementation passed the supported Q4_K/F32 CPU `MUL_MAT` backend-op checks. In the alternating A/B evaluation, `pp512` regressed by 1.84%, `tg128 @ ctx512` was unchanged within noise (-0.09%), and the operator microbenchmark was unchanged within noise (+0.28% faster). The candidate did not meet the acceptance gate and was removed. The original kernel remains in the source tree.

## Setup

- Model: `/home/sonic182/.cache/llama-qwen3/Qwen3-0.6B.Q4_K_M.gguf`
- Host: AMD Ryzen 5 3550H, 4 cores / 8 threads, AVX2
- Build: host Release, `GGML_NATIVE=ON`, CPU backend, BLAS/CUDA/Vulkan disabled
- `pp512`: 4 threads, batch/ubatch 512, five `llama-bench` repetitions per run
- `tg128 @ ctx512`: 6 threads, batch/ubatch 512, five repetitions per run
- Three independent runs per variant and workload, with control and candidate builds alternated as A-B, B-A, A-B
- The two builds used separate copies of `libggml-cpu.so.0` selected through `LD_LIBRARY_PATH`. `ldd` confirmed the selected library for each variant.

## End-to-end results

Throughput in tokens/second; each value is one `llama-bench -r 5` run.

| Workload | Baseline runs | Candidate runs | Baseline mean | Candidate mean | Change |
|---|---:|---:|---:|---:|---:|
| `pp512` | 130.261, 128.513, 124.601 | 128.501, 128.024, 119.815 | 127.792 | 125.447 | -1.84% |
| `tg128 @ ctx512` | 31.601, 30.185, 30.974 | 31.201, 30.780, 30.690 | 30.920 | 30.891 | -0.09% |

An earlier unpaired sequence appeared to improve decode by 1.56% and leave prefill unchanged (+0.06%). The alternating sequence did not reproduce that gain. The drift across runs, especially in `pp512`, makes the paired result the more useful decision point, though it remains noisy.

The prefill slowdown cannot be attributed confidently to this GEMV schedule from throughput alone. Even treating it as measurement noise, the experiment has no repeatable decode gain.

## Operator checks and microbenchmark

The CPU backend-op test covered `MUL_MAT` with Q4_K weights and F32 inputs. All supported cases passed; F16 input combinations were reported as unsupported by the CPU backend and are not failures.

Targeted case: `MUL_MAT(type_a=q4_K,type_b=f32,m=4096,n=1,k=14336)`.

| Build | Run times (us/run) | Mean | Change vs control |
|---|---:|---:|---:|
| Control | 1171.78, 1213.45, 1176.30 | 1187.18 | - |
| Streaming candidate | 1179.07, 1200.87, 1171.68 | 1183.87 | 0.28% faster |

The recorded historical control was 1165.68 us/run. Comparing the candidate only with that single older result suggested a 1.51% regression, but the interleaved A/B measurements did not reproduce it. The microbenchmark difference is too small to establish a kernel speedup or regression.

## Generated code inspection

The candidate emitted 34 YMM stack-memory references in its measured loop region; the preserved old path emitted 36 in its corresponding loop. This is only a small reduction. The compiler inlined the fallback into the public function, so the combined function reserved 0x2a8 (680) bytes of stack, compared with 0x268 (616) bytes in the earlier baseline profile. The assembly therefore did not show a strong enough reduction in spill traffic to justify keeping the additional path.

The schedule keeps the same packed-weight reads and arithmetic. A reduction of two stack-memory references in the inspected loop is unlikely to change overall runtime much; this is an inference from the assembly and benchmarks, not a measured breakdown of memory and execution costs.

## Decision

Rejected. The alternating runs showed no decode gain and a 1.84% prefill regression. The small reduction in YMM stack references does not justify the additional code. `ggml/src/ggml-cpu/arch/x86/repack.cpp` was restored to the original Q4_K GEMV implementation after this evaluation.
