# Qwen3-0.6B CPU kernel experiment

## Setup

- Model: `QuantFactory/Qwen3-0.6B-GGUF/Qwen3-0.6B.Q4_K_M.gguf`
- SHA-256: `7af3fdf842f87b24672f8a7f1dd50404043f0bfb71093ff91c31d2b49df4631d`
- Host: AMD Ryzen 5 3550H, AVX2/FMA, 8 logical CPUs
- Build: Release, `GGML_NATIVE=ON`, CPU/OpenMP only, BLAS and accelerator backends disabled
- Baseline and candidate measurements use `llama-bench`, batch/ubatch 512, five repetitions, and CPU-only execution.

The GGUF contains 169 Q4_K tensors and 29 Q6_K tensors (plus F32 metadata/norm tensors). Q4_K was selected for the experiment because it is the dominant quantized type.

## Baseline

Best observed settings from the 1/2/4/6/8-thread sweep:

| Workload | Threads | Throughput |
| --- | ---: | ---: |
| Prefill, 128 tokens | 4 | 139.67 tokens/s |
| Prefill, 512 tokens | 4 | 127.34 tokens/s |
| Decode, 128 tokens at context depth 512 | 6 | 31.34 tokens/s |

Raw results: `qwen3-0.6b-cpu-baseline-prefill.json` and `qwen3-0.6b-cpu-baseline-decode.json`.

## Candidate and outcome

The candidate alternated Q4_K AVX2 quant blocks across two float accumulators to shorten the single-accumulator dependency chain. The generic path and other ISA paths were unchanged. The candidate passed all 110 existing Q4_K/Q6_K `MUL_MAT` correctness cases. A fixed-input KL check over three chunks found 100% same top predictions, zero reported KL divergence at output precision, and probability RMS delta below 0.001%.

| Workload | Baseline | Candidate | Change |
| --- | ---: | ---: | ---: |
| Prefill, 128 tokens, 4 threads | 139.67 | 140.02 | +0.25% |
| Prefill, 512 tokens, 4 threads | 127.34 | 130.00 | +2.09% |
| Decode, 128 tokens at context depth 512, 6 threads | 31.34 | 30.64 | -2.23% |

The targeted `test-backend-ops` Q4_K matmul microbenchmark also regressed from 1165.68 to 1261.62 us/run for its 4096x14336 case. The candidate does not meet the 5% end-to-end improvement goal and exceeds the allowed 2% regression in decode, so it was discarded. The original CPU kernel is restored in the source tree.

Candidate results are in `qwen3-0.6b-cpu-candidate-prefill.json` and `qwen3-0.6b-cpu-candidate-decode.json`. Operator checks, timings, and numerical comparison are retained in the other `qwen3-0.6b-cpu-mul-mat-*` artifacts.
