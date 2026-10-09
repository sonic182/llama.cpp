# Qwen3-0.6B CPU repack and zero-copy experiment

## Result

Disabling CPU repack removed about 203 MiB of resident memory and shortened the warm-cache process startup by 0.245 s. Prefill was unchanged within measurement noise. With the current experimental Q4_K AVX2 assembly kernel, decode fell from 32.93 to 32.13 tokens/s (-2.43%). All four temperature-controlled decode pairs favored repack, so the direct mapped path does not meet the existing limit of at most 1% decode regression. The default remains `--repack 1`.

This is a comparison of two model-load and kernel paths, not a pure copy instruction benchmark. `--repack 0` selects ordinary CPU buffers and native Q4_K weights; `--repack 1` selects `CPU_REPACK` where supported and feeds Q4_Kx8 weights to its specialized kernels. Other internal buffers still exist in both modes.

## Setup and path check

- Host: AMD Ryzen 5 3550H, 4 cores / 8 threads, AVX2; CPU-only, no BLAS/CUDA/Vulkan.
- Model: `/home/sonic182/.cache/llama-qwen3/Qwen3-0.6B.Q4_K_M.gguf` (462 MiB file).
- Main A/B: same host Release binary, `GGML_NATIVE=ON`, `GGML_CPU_REPACK=ON`, `-ngl 0`, `-lm mmap`, switching only `--repack 1` or `--repack 0`.
- `llama-bench --repack` sets `llama_model_params.use_extra_bufts`. On this CPU, the relevant extra buffer is `CPU_REPACK`; disabling it leaves the ordinary CPU buffer.
- A second host build with `GGML_CPU_REPACK=OFF` had 558260 KiB peak RSS for the startup probe, consistent with 558316-558588 KiB from `--repack 0` in the main binary. This checks that the runtime switch reaches the intended path.
- The main binary contains the experimental inline ASM Q4_Kx8 GEMV in `ggml/src/ggml-cpu/arch/x86/repack.cpp`. The direct Q4_K path does not use that ASM.

## Startup and memory

Startup probe: `llama-bench -p 1 -n 0 -t 4 -b 512 -ub 512 -ngl 0 -r 1 -lm mmap`, one fresh process per result, A-B-B-A-A-B-B-A. `wall_s` and peak RSS came from `/usr/bin/time`. This wall time includes process startup, model setup, and one-token inference; it is not isolated model-load time. The model file was warm in the OS page cache.

| Repack | Wall times (s) | Mean wall (s) | Peak RSS (KiB), four processes | Mean peak RSS (KiB) |
| --- | --- | ---: | --- | ---: |
| ON | 0.79, 0.78, 0.77, 0.83 | 0.7925 | 766436, 766568, 766708, 766568 | 766570 |
| OFF | 0.55, 0.55, 0.55, 0.54 | 0.5475 | 558588, 558400, 558396, 558316 | 558425 |

OFF reduced this startup probe by 30.9% and peak RSS by 208145 KiB (203.3 MiB, 27.2%).

One `/proc/<pid>/status` sample during `tg128 @ ctx512` checked memory while inference was running:

| Repack | VmRSS (KiB) | RssAnon (KiB) | RssFile (KiB) |
| --- | ---: | ---: | ---: |
| ON | 853592 | 376112 | 477480 |
| OFF | 645716 | 166600 | 479116 |

The active-process RSS difference was 207876 KiB (203.0 MiB). File-backed pages were nearly the same; the extra anonymous resident memory was about 205 MiB. This supports the interpretation that the repacked CPU buffer accounts for the saving. Steady RSS is one sample per mode, not a repeated average.

## End-to-end throughput with the experimental ASM

Each value below is an independent `llama-bench -r 1` process. Variants ran A-B-B-A-A-B-B-A, with a wait before each process until the reported CPU temperature was at most 72 C. `pp512` used 4 threads; `tg128 @ ctx512` used 6 threads; both used batch/ubatch 512 and CPU-only inference.

| Workload | Repack ON runs (t/s) | Repack OFF runs (t/s) | ON mean | OFF mean | OFF vs ON |
| --- | --- | --- | ---: | ---: | ---: |
| `pp512` | 132.785, 127.048, 129.251, 130.847 | 130.470, 130.557, 128.106, 130.777 | 129.983 | 129.978 | -0.004% |
| `tg128 @ ctx512` | 33.268, 33.188, 32.587, 32.670 | 32.274, 32.474, 31.935, 31.824 | 32.928 | 32.127 | -2.434% |

All four decode pairs favored ON by 2.04-3.08%. Prefill pairs were mixed and do not establish a speed difference. A preliminary longer prefill sequence reached 96.8 C and was stopped; it is excluded from this table. Temperature control limits one obvious source of drift but does not isolate the machine from other services.

## Control with the original CPU kernel

The original `libggml-cpu.so.0` saved before the ASM experiment was selected with `LD_LIBRARY_PATH`, while the same `llama-bench` binary and `--repack` switch were used. Eight `tg128 @ ctx512` processes ran with the same temperature rule and order:

| Repack ON runs (t/s) | Repack OFF runs (t/s) | ON mean | OFF mean |
| --- | --- | ---: | ---: |
| 33.014, 32.097, 31.193, 30.663 | 31.681, 32.793, 31.745, 32.336 | 31.742 | 32.139 |

The means would favor OFF by 1.25%, but the ON runs drifted downward monotonically and the pairwise results conflicted: one pair favored ON, three favored OFF. This does not establish a reproducible original-kernel speedup from disabling repack. The OFF path was around 32.13 t/s in both the current-ASM and original-library series; the reliable 2.43% penalty reported above applies to the current experimental ASM configuration.

## Decision

Keep `GGML_CPU_REPACK=ON` and the default `--repack 1` for this experimental branch while decode speed is the priority and the accepted regression limit is 1%. `--repack 0` is a useful memory/startup option when saving about 203 MiB is worth the measured decode trade-off. No model format, runtime default, Docker container, or service was changed by this experiment.
