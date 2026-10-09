# llama.cpp experiments summary

Updated: 2026-09-27

## Scope and environment

- Work is on branch `qwen3-cpu-kernel-experiment` in this private fork.
- The MiniLM embedding service is CPU-only, has no Vulkan backend, and is reachable by containers at `10.0.1.24:8080`; Compose does not publish a host port.
- Later kernel builds and checks were done on the host in Release mode with `GGML_NATIVE=ON`, CPU/OpenMP enabled, and BLAS/Vulkan/other accelerators disabled.
- Qwen benchmark host: AMD Ryzen 5 3550H, AVX2/FMA, 8 logical CPUs.

## All-MiniLM-L6-v2 embedding service

- Added a Docker Compose service built from the local llama.cpp source, using the F16 GGUF from `gaianet/All-MiniLM-L6-v2-Embedding-GGUF`, mean pooling, and the existing private Docker network.
- Downloaded/loaded the model and smoke-tested text-to-embedding through the OpenAI-compatible `/v1/embeddings` endpoint. Responses were normalized 384-dimensional vectors.
- Added `scripts/embedding_bench.py`, a `uv` script pinned to `aiosonic==1.0.7`. It supports warmups, async concurrency, latency percentiles, throughput, response validation, and JSON output.
- Recorded baseline: 100/100 requests succeeded at concurrency 8; 185.07 requests/s; average latency 41.53 ms; p50 43.16 ms; p95 48.35 ms; p99 51.79 ms.
- Details: `scripts/evals/minilm-embedding-baseline.md` and `scripts/evals/minilm-*.json`.

## OpenBLAS trial

- Built and evaluated an OpenBLAS-enabled CPU server against the baseline for short and long embedding inputs. The BLAS libraries loaded, but the candidate was slower in both cases.

| Input | CPU baseline | OpenBLAS default | OpenBLAS, 1 thread |
| --- | ---: | ---: | ---: |
| Short throughput | 178.24 req/s | 82.66 req/s | 86.03 req/s |
| Long throughput | 28.23 req/s | 23.40 req/s | 23.07 req/s |

- The BLAS candidate container was stopped at the user's request. The measurements reject OpenBLAS for this service workload; they do not establish that BLAS cannot help other models or batch sizes.
- Details and trial files: `scripts/evals/minilm-openblas-evaluation.md` and `scripts/evals/minilm-{base,blas,blas1}-*.json`.

## Qwen3-0.6B CPU kernel experiment

- Downloaded `Qwen3-0.6B.Q4_K_M.gguf` from `QuantFactory/Qwen3-0.6B-GGUF` into the external cache at `/home/sonic182/.cache/llama-qwen3/`. SHA-256: `7af3fdf842f87b24672f8a7f1dd50404043f0bfb71093ff91c31d2b49df4631d`.
- Built llama.cpp and ran checks on the host. No Vulkan or BLAS was enabled.
- Baseline best observed rates from the thread sweep: prefill 128 tokens, 139.67 t/s (4 threads); prefill 512, 127.34 t/s (4 threads); decode 128 tokens at context depth 512, 31.34 t/s (6 threads).
- Tried alternating Q4_K AVX2 quant blocks across two accumulators in `ggml/src/ggml-cpu/arch/x86/quants.c`. Existing Q4_K/Q6_K `MUL_MAT` checks passed (110 cases), but the Q4_K microbenchmark got slower: 1165.68 to 1261.62 us/run. End-to-end changes were +0.25% for 128-token prefill, +2.09% for 512-token prefill, and -2.23% for decode. The kernel change was discarded and the original source restored.
- Current preference for future tradeoffs: roughly 2% improvement may be worthwhile, with regression capped at 1%.
- Details: `scripts/evals/qwen3-0.6b-cpu-kernel-experiment.md` and adjacent baseline/candidate JSON and operator results.

## Static FFN-skip calibration

- Temporarily added a Qwen3 graph-builder selector, skipped each of the model's 28 FFNs individually, and compared logits/PPL on two 512-token WikiText-2 test chunks. The selector was removed after the experiment; no FFN-skip feature remains in the source.
- No single-layer skip was low-impact. The least disruptive by mean KLD was layer 13, but it still raised PPL by 10.1% and matched the baseline top token at only 85.3% of scored positions. Layers 0-2 and 27 were particularly sensitive.
- For layer 13, two five-repetition benchmark rounds averaged +1.10% prefill and -0.78% decode. The direction varied between rounds, and the small speed change was inconclusive relative to the quality loss.
- Conclusion: the current evidence does not support adaptive FFN skipping. Any predictor would need to avoid most of the quality impact while costing less than the small observed speed budget.
- Details: `scripts/evals/qwen3-0.6b-ffn-skip-calibration.md` and `scripts/evals/qwen3-0.6b-ffn-skip-*-bench*.json`.

## Host profiling and kernel selection

- Added `scripts/perf-enable.sh` and `scripts/perf-disable.sh` to let the user temporarily enable and restore host perf access. Profiled `pp128`, `pp512`, and `tg128 @ ctx512` separately. `perf` has since been restored.
- In decode, the largest named self-sample shares were repacked Q4_K GEMV (23.04%), Q6_K dot product (15.37%), flash attention (12.58%), and repacked Q4_K GEMM (12.04%). An unresolved OpenMP frame accounted for another 9.97%. Decode IPC was 0.88 versus 1.35-1.39 for prefill; this points to kernel and memory behavior but does not isolate compute, cache, and DRAM limits. L2/L3 and aggregate DRAM event aliases were unavailable on this host.
- The existing repacked Q4_K GEMV in `ggml/src/ggml-cpu/arch/x86/repack.cpp` already computes eight outputs and reuses activation data. Thus a new generic multi-row kernel would duplicate an existing design. Inspection of its generated AVX2 code found a 616-byte stack frame and many YMM stack references consistent with register pressure; static counts are not measured spill traffic.
- Details: `scripts/evals/qwen3-0.6b-cpu-perf-profile.md`.

## Q4_K GEMV streaming attempt

- Tried consuming each 32-byte packed-weight vector immediately after loading it, unpacking low/high nibbles and accumulating both before the next load. Kept the original intrinsics path as fallback.
- Existing supported Q4_K/F32 CPU `MUL_MAT` checks passed. The targeted operator microbenchmark was effectively unchanged (+0.28% faster). Alternating full-model runs showed `pp512` -1.84% and `tg128 @ ctx512` -0.09%; an earlier apparent decode gain did not repeat.
- Generated-code inspection found only a small reduction in YMM stack references (36 to 34 in the measured loop region). The candidate was removed because it did not improve decode and exceeded the 1% prefill-regression limit.
- Details: `scripts/evals/qwen3-0.6b-q4k-gemv-streaming.md`.

## Direct AVX2 assembly prototypes

- Implemented three guarded inline-assembly schedules for the repacked Q4_Kx8 GEMV in `ggml/src/ggml-cpu/arch/x86/repack.cpp`: one, two, then four weight-vector pairs per assembly block. They keep the original accumulation order and layout. The original intrinsics path remains the fallback outside x86-64/GNU/AVX2.
- The current third prototype builds on the host and passes all 64 supported Q4_K/F32 `test-backend-ops` cases. Its GEMV stack frame is 264 bytes and the symbol has 14 YMM stack references, compared with 616 bytes and 36 references in the original. These are assembly-inspection counts, not proof of reduced runtime memory traffic.
- For the third prototype, the operator microbenchmark mean improved 2.41%, but paired results disagreed. Its complete end-to-end A/B was stopped because host conditions and throughput drifted. Earlier first-prototype runs suggested +3.32% decode but -1.76% prefill; a further prefill sequence also drifted. None establishes a repeatable end-to-end win under the 1% regression rule.
- The third ASM prototype remains in the worktree for further experiments at the user's request; it has not been accepted as a proven speed optimization. No separate `.S` kernel or AsmJit/JIT integration was implemented.
- Details: `scripts/evals/qwen3-0.6b-q4k-gemv-asm.md`.

## CPU repack / zero-copy tradeoff

- Tested `llama-bench --repack 1` versus `--repack 0` in the same CPU-only host binary, using memory-mapped model loading. The latter bypasses the extra `CPU_REPACK` buffer and its Q4_Kx8 kernel path; it does not eliminate all internal buffers. A separate build with `GGML_CPU_REPACK=OFF` confirmed the runtime switch reaches the intended path.
- With repack disabled, warm-cache process startup fell from 0.793 to 0.548 s (30.9%); this is whole-process time, not isolated model-load time. Peak RSS fell from 766570 to 558425 KiB: 203.3 MiB or 27.2% less. A one-time active-process sample showed about 203 MiB less RSS too.
- With the current experimental ASM, four temperature-controlled runs per mode found `pp512` unchanged (~129.98 t/s) and `tg128 @ ctx512` 32.93 to 32.13 t/s (-2.43%) without repack. All four decode pairs favored repack. A control with the original CPU library was noisy and inconclusive, so this speed difference cannot be attributed solely to the repacked layout.
- The runtime default remains repack ON, but `--repack 0` is a useful, user-valued memory/startup option on this machine despite the measured decode cost; the user explicitly considered this tradeoff positive. No service or Docker configuration was changed for the test. The `llamacpp-dev` skill was updated and validated to assess zero-copy together with layout, load time, RSS, prefill, and decode.
- Details: `scripts/evals/qwen3-0.6b-cpu-repack-zero-copy.md`.

### Larger-model check: Qwen3-4B Q4_K_M

- Downloaded and tested the 2.52 GiB GGUF on the same host. Across four fresh startup probes per mode, disabling repack reduced peak RSS from 4.25 to 2.62 GiB (1.63 GiB / 38.37%) and warm-cache whole-process time from 3.07 to 1.07 s.
- At `pp128`, OFF averaged 17.42 t/s versus 16.78 ON (+3.81%), but the samples were few and one `pp64` pair disagreed. At `tg32 @ ctx512`, OFF averaged 7.91 versus 8.015 t/s ON (-1.31%); paired results conflicted, so decode impact is inconclusive.
- Decode peak RSS also fell about 1.63 GiB (36.46%). Tctl peaked at 98.4 C; CPU frequency was not recorded, so possible thermal throttling is unconfirmed. Details: `scripts/evals/qwen3-4b-cpu-repack-memory.md`.

## What remains open

- No affinity/SMT topology sweep, alternative Qwen quantization comparison, fixed-shape specialization, activation-quantization fusion, JIT/AsmJit, long-context/KV-cache study, or software-prefetch experiment has been completed. These remain candidates, not measured improvements.
- The ASM prototype still needs stable paired end-to-end evaluation. The host serves other containers and runs hot under the larger CPU-only models. Sub-2% benchmark differences need careful repetition.

## Current worktree notes

- The temporary FFN selector, rejected Q4_K dual-accumulator change, and rejected streaming variant are absent from source. The third experimental inline-ASM prototype remains in `ggml/src/ggml-cpu/arch/x86/repack.cpp`.
- At the latest check, no files were staged. `summary.md` is modified; `.agents/`, `docker-compose.yml`, and the Qwen3-4B report are untracked; `.devops/cpu.Dockerfile` is modified and `CONTRIBUTING.md` is deleted. Existing worktree changes were preserved.
