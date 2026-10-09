# Qwen3-0.6B static FFN-skip calibration

## Setup

- Model: Qwen3-0.6B Q4_K_M, CPU-only host build on AMD Ryzen 5 3550H.
- Calibration corpus: first two 512-token chunks of WikiText-2 test, with logits compared to the unmodified model.
- Each variant omitted one layer's complete FFN subgraph while retaining attention and KV updates.
- This was a temporary Qwen3 graph-builder selector. The selector has been removed; no runtime skip feature remains.
- Perplexity and logit metrics are a small screening set, not a quality certification.

## Quality impact by skipped layer

`PPL ratio` is candidate PPL divided by baseline PPL. `Same top` is the percentage of scored positions where the highest-probability token matched the baseline.

| Layer | PPL ratio | Mean KLD | Same top |
| ---: | ---: | ---: | ---: |
| 0 | 18.883 | 3.1770 | 44.31% |
| 1 | 259.395 | 6.0175 | 5.69% |
| 2 | 51.903 | 4.1397 | 32.16% |
| 3 | 1.165 | 0.3075 | 76.86% |
| 4 | 1.190 | 0.2789 | 75.49% |
| 5 | 1.271 | 0.2930 | 74.90% |
| 6 | 1.133 | 0.1791 | 81.57% |
| 7 | 1.101 | 0.1347 | 82.55% |
| 8 | 1.167 | 0.1122 | 83.73% |
| 9 | 1.044 | 0.1372 | 80.98% |
| 10 | 1.032 | 0.1826 | 79.61% |
| 11 | 1.115 | 0.1062 | 85.49% |
| 12 | 1.122 | 0.1003 | 85.69% |
| 13 | 1.101 | 0.0867 | 85.29% |
| 14 | 1.064 | 0.1067 | 83.33% |
| 15 | 1.101 | 0.0920 | 86.86% |
| 16 | 1.265 | 0.2415 | 76.28% |
| 17 | 1.086 | 0.1239 | 80.78% |
| 18 | 1.142 | 0.1521 | 80.59% |
| 19 | 1.131 | 0.2089 | 82.35% |
| 20 | 1.191 | 0.2106 | 80.59% |
| 21 | 1.340 | 0.1993 | 80.98% |
| 22 | 1.174 | 0.2045 | 83.53% |
| 23 | 1.129 | 0.1645 | 84.71% |
| 24 | 1.212 | 0.1769 | 84.90% |
| 25 | 1.147 | 0.1862 | 84.71% |
| 26 | 1.424 | 0.2824 | 79.02% |
| 27 | 1.945 | 0.4652 | 76.08% |

The least disruptive single-layer skip by mean KLD was layer 13 (0.0867). Its PPL was 20.76 versus the baseline 18.86, a 10.1% increase, and top-token agreement was 85.3%. No layer looked safe to omit outright. Layers 0-2 and 27 were especially sensitive.

## Speed check for layer 13

Two independent five-repetition `llama-bench` runs tested 128-token prefill and 128-token decode at four threads. Averaging the two run means:

| Workload | Baseline | Skip layer 13 | Change |
| --- | ---: | ---: | ---: |
| Prefill, 128 tokens | 143.52 t/s | 145.11 t/s | +1.10% |
| Decode, 128 tokens | 51.42 t/s | 51.01 t/s | -0.78% |

The direction varied between benchmark rounds, so the small speed change is inconclusive. Raw measurements are in `qwen3-0.6b-ffn-skip-*-bench*.json`.

## Takeaway

Static skipping one complete FFN saves little end-to-end time on this model and noticeably changes its output, even for the least sensitive layer. The current evidence does not support implementing adaptive FFN skipping yet. A predictor would need to avoid most of the quality loss while keeping its own decision cost below the small measured speed budget.
