# Qwen3-0.6B Q4_K GEMV assembly experiment

## Status

The third AVX2 assembly prototype remains in `ggml/src/ggml-cpu/arch/x86/repack.cpp` for further experiments. It is not an accepted performance improvement yet. The original intrinsics implementation remains the fallback outside the guarded x86-64/GNU/AVX2 path.

The current prototype processes all four pairs of packed weight vectors for one 64-value segment in one inline assembly block. Each pair is unpacked and consumed before the next pair is loaded. The second and third prototypes reuse each broadcast Q8_K activation across two weight pairs. The integer accumulation order and Q4_Kx8 layout are unchanged.

## Setup

- Model: `/home/sonic182/.cache/llama-qwen3/Qwen3-0.6B.Q4_K_M.gguf`
- CPU: AMD Ryzen 5 3550H, 4 cores / 8 threads, AVX2
- Build: host Release build with `GGML_NATIVE=ON`, CPU backend, no BLAS/CUDA/Vulkan
- Control and candidate: separate `libggml-cpu.so.0` copies selected with `LD_LIBRARY_PATH`
- Operator case: `MUL_MAT(type_a=q4_K,type_b=f32,m=4096,n=1,k=14336)`
- Full model: `pp512` with 4 threads and `tg128 @ ctx512` with 6 threads, batch/ubatch 512
- A/B order for each measured series: control-candidate, candidate-control, control-candidate

## Correctness and generated code

The current prototype builds on the host and passes all 64 supported CPU `MUL_MAT` Q4_K/F32 cases in `test-backend-ops`. The first and second prototypes also passed this check.

| Build | Stack frame | YMM stack references in the symbol | `vbroadcasti128` instructions in the symbol | `vpmaddubsw` instructions in the symbol |
| --- | ---: | ---: | ---: | ---: |
| Original kernel | 616 bytes | 36 | - | - |
| First ASM prototype, one pair per block | 232 bytes | 12 | 8 | 16 |
| Second ASM prototype, two pairs per block | 264 bytes | 15 | 4 | 16 |
| Third ASM prototype, four pairs per block | 264 bytes | 14 | 4 | 16 |

The stack-reference counts come from `objdump` on the public GEMV symbol. They are a code inspection heuristic, not measured spill traffic. Reusing activation broadcasts reduced their static count, but it did not reduce the stack references relative to the first ASM prototype.

## Operator microbenchmark

Times are microseconds per run; lower is better. Each value is one independent `test-backend-ops perf` process with hundreds of timed operator calls.

| Prototype | Control times | Candidate times | Control mean | Candidate mean | Candidate time change |
| --- | --- | --- | ---: | ---: | ---: |
| First, one pair | 1214.02, 1208.43, 1171.52 | 1149.35, 1230.93, 1171.13 | 1197.99 | 1183.80 | -1.18% |
| Second, two pairs | 1190.64, 1167.52, 1202.01 | 1194.09, 1158.61, 1216.14 | 1186.72 | 1189.61 | +0.24% |
| Third, four pairs | 1231.62, 1187.49, 1159.77 | 1139.06, 1186.89, 1166.58 | 1192.96 | 1164.18 | -2.41% |

The third prototype's paired differences were inconsistent: the first pair favored the candidate strongly, the second was nearly equal, and the third favored the control slightly. Its mean is promising but does not establish a repeatable gain.

## Full-model measurements

The first ASM prototype had these initial A/B observations (three independent `llama-bench -r 5` processes per build):

| Workload | Control runs, t/s | Candidate runs, t/s | Observed mean change |
| --- | --- | --- | ---: |
| `tg128 @ ctx512` | 32.318, 31.211, 31.520 | 32.705, 32.454, 33.043 | +3.32% |
| `pp512` | 124.969, 116.456, 117.057 | 120.215, 116.026, 115.916 | -1.76% |

An additional alternating `pp512` sequence gave control 128.525, candidate 130.363, candidate 126.653, control 121.244, control 116.748, candidate 116.680 t/s. That sequence has a large time trend, so its paired differences are not reliable evidence of a kernel effect.

For the third prototype, the attempted `tg128 @ ctx512` sequence started with control 31.353 and candidates 33.159 and 32.720 t/s. The run was stopped before the remaining controls because the host conditions and throughput were drifting. There is no complete end-to-end A/B result for this prototype. The host runs multiple services and containers. `loadavg` also includes the benchmark's own threads, so its increase during the run does not identify external interference.

## Zero-copy question

`GGML_CPU_REPACK` is enabled in this build. `ggml_backend_cpu_repack_buffer_set_tensor` calls the Q4_K repacker, which writes `block_q4_Kx8` data into a separate CPU buffer from the original Q4_K input. `llama-bench --repack 0` disables that extra buffer path within the same binary. The measured load time, RSS, prefill, and decode trade-off is recorded in [qwen3-0.6b-cpu-repack-zero-copy.md](qwen3-0.6b-cpu-repack-zero-copy.md).

## Next decision

Keep this prototype as experimental code. Repeat full-model A/B with per-run CPU frequency, temperature, and competing-process observations before applying the acceptance rule: reproducible end-to-end improvement and no measured workload worse by more than 1%. If it wins, inspect a full-kernel `.S` implementation; a separate function call for every 32-byte pair would add overhead inside the hot loop.
