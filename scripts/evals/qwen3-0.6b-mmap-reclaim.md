# Qwen3-0.6B mmap source reclaim with CPU repack

Recorded: 2026-09-27

## Change under test

Adapted upstream [PR #24156](https://github.com/ggml-org/llama.cpp/pull/24156) as an opt-in `--reclaim-mmap-source` model-load option. On Linux, after a tensor is copied from an mmap into a separate backend buffer, the loader calls `madvise(MADV_DONTNEED)` on complete interior source pages. The mapping and source file remain unchanged. Reclaim is skipped when mmap is disabled or model pages are locked. CPU_REPACK and its Q4_Kx8 kernel remain enabled and unchanged.

`llama-bench` has a separate argument parser, so it also accepts `--reclaim-mmap-source <0|1>` for this evaluation.

## Setup

- Host: AMD Ryzen 5 3550H, 4 cores / 8 threads, AVX2; CPU-only, no BLAS, CUDA, or Vulkan.
- Model: `/home/sonic182/.cache/llama-qwen3/Qwen3-0.6B.Q4_K_M.gguf` (484,220,000 bytes).
- Build: host Release build, commit `e4fa0315`, `GGML_NATIVE=ON`, `GGML_CPU_REPACK=ON`, `GGML_BLAS=OFF`.
- Both modes used `-lm mmap --repack 1 -ngl 0`; only `--reclaim-mmap-source` changed. The model file was warm in the OS cache. Other host services were left running; no temperature gating was used.
- Each benchmark result used a fresh process and `-r 1`, alternating baseline and reclaim modes.

## Startup peak RSS

Probe: `llama-bench -p 1 -n 0 -t 4 -b 512 -ub 512 -ngl 0 -r 1 -lm mmap --repack 1 --reclaim-mmap-source <0|1> --no-warmup`, measured with `/usr/bin/time`.

| Mode | Peak RSS, KiB (4 runs) | Mean peak RSS |
| --- | --- | ---: |
| Repack, no reclaim | 766484, 766344, 766508, 766512 | 766462 KiB |
| Repack + reclaim | 559300, 559132, 559296, 559388 | 559279 KiB |

Reclaim lowered mean peak RSS by **207183 KiB (202.3 MiB, 27.0%)**. This is close to the previous `--repack 0` result of 558425 KiB, while retaining the repacked layout. The reclaim result is also stable across all four processes (range 256 KiB).

## Prefill and decode

Each value is a separate process. `pp512` used 4 threads. `tg128 @ d512` used 6 threads. Order was interleaved to reduce ordering bias.

| Workload | Repack, no reclaim (t/s) | Repack + reclaim (t/s) | Mean change |
| --- | --- | --- | ---: |
| `pp512` | 121.71, 129.95, 128.75, 131.30 | 129.75, 127.62, 128.52, 123.55 | -0.44% |
| `tg128 @ d512` | 33.24, 32.12, 32.23, 31.88, 32.49, 32.94, 32.05, 33.72 | 33.87, 33.98, 33.26, 31.31, 32.52, 33.95, 33.51, 32.08 | +1.46% |

Decode runs varied by over 8% within a mode, indicating substantial host noise. Six of eight adjacent pairs favored reclaim; two favored the baseline. Treat the small positive mean as noise, not a decode speedup. There is no repeatable evidence that reclaim slows inference. Prefill means are effectively equal relative to their run-to-run spread.

## Result

Keep the change as an opt-in experiment: on Qwen3-0.6B it recovers about **202 MiB of process peak RSS**, bringing repack-on memory close to the earlier repack-off run. The measured prefill and decode results show no consistent performance regression; the test is not precise enough to claim a speed improvement. Since only dormant mmap source pages are advised away after their contents have been copied, steady-state inference continues to use the same repacked tensors and kernels.

The test exercised model loading, prompt processing, and 128-token generation through `llama-bench`. It did not run `test-backend-ops` (the change does not modify backend kernels) or compare generated text byte-for-byte.
