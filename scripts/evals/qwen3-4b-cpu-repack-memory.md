# Qwen3-4B CPU repack memory check

Recorded: 2026-09-27

## Setup

- Host: AMD Ryzen 5 3550H, 12 GiB RAM, 8 GiB available before the test; existing services were left running.
- Model: `QuantFactory/Qwen3-4B-GGUF`, `Qwen3-4B.Q4_K_M.gguf`, 2716068512 bytes (2.52 GiB). SHA-256: `8d5eeb94b4214c6906bfd314dff812a3472d7688e5c96dfd7ba7f44287736794`.
- Same host Release CPU binary for both modes, with `GGML_NATIVE=ON`, `GGML_CPU_REPACK=ON`, and the experimental Q4_Kx8 ASM path. No BLAS, CUDA, or Vulkan. The model was loaded with `-ngl 0 -lm mmap`; only `--repack` changed.
- Startup probe: `llama-bench -m <model> -p 1 -n 0 -t 4 -b 512 -ub 512 -ngl 0 -r 1 -lm mmap --repack <0|1>`. Each measurement used a fresh process; `/usr/bin/time` recorded wall time and peak RSS. The model was warm in the OS page cache.

## Repeated startup and peak RSS

Order: ON, OFF, OFF, ON, ON, OFF, OFF, ON.

| Mode | Wall times (s) | Mean wall (s) | Peak RSS (KiB) | Mean peak RSS (KiB) |
| --- | --- | ---: | --- | ---: |
| Repack ON | 3.13, 3.03, 3.07, 3.05 | 3.070 | 4457504, 4457552, 4457532, 4457648 | 4457559 |
| Repack OFF | 1.05, 1.04, 1.08, 1.09 | 1.065 | 2747312, 2747236, 2747456, 2747428 | 2747358 |

With repack OFF, peak RSS fell by 1710201 KiB = 1670.1 MiB = 1.631 GiB, or 38.37% of the ON peak. Warm-cache whole-process startup fell by 2.005 s, or 65.3%. This is not an isolated model-load-time or copy-cost measurement. The result is larger than the 203 MiB / 27.2% observed for Qwen3-0.6B Q4_K_M.

## Throughput

The longer `pp128` run used four fresh processes in OFF, ON, ON, OFF order; one earlier cool-start ON run is also listed. Four threads, `-r 1`:

| Mode | Runs (t/s) | Mean (t/s) |
| --- | --- | ---: |
| Repack ON | 16.71, 16.89, 16.74 | 16.78 |
| Repack OFF | 17.65, 17.19 | 17.42 |

OFF was 3.81% faster by these small samples. One `pp64` pair disagreed (18.15 t/s ON, 17.64 t/s OFF), so prefill speed is suggestive, not conclusive.

For `tg32 @ depth512`, six threads, the order was ON, OFF, OFF, ON:

| Mode | Runs (t/s) | Mean (t/s) |
| --- | --- | ---: |
| Repack ON | 7.97, 8.06 | 8.015 |
| Repack OFF | 7.70, 8.12 | 7.91 |

OFF averaged 1.31% slower, but the paired directions conflict and two runs per mode are not enough to establish a decode regression. Peak RSS during these decode runs averaged 4689116 KiB ON and 2979264 KiB OFF, a 1669.8 MiB (36.46%) reduction. Tctl reached 98.4 C during generation; no CPU frequency trace was collected, so thermal throttling is possible but not confirmed.

## Interpretation

The RAM and startup benefit of avoiding the CPU_REPACK buffer scales materially with this larger Q4_K_M model. The two modes also select different matrix kernels, so this is not a pure zero-copy benchmark. This run does not establish the speed trade-off: prefill samples favor OFF, while decode samples are mixed. The default and deployed services were not changed. The GGUF was downloaded into `/home/sonic182/.cache/llama-qwen3/`; no Docker container was stopped or rebuilt.
