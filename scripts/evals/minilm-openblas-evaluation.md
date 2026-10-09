# MiniLM OpenBLAS evaluation

Recorded: 2026-09-27

## Summary

The OpenBLAS-enabled CPU image built and served the embedding model correctly, but it was slower than the existing CPU build on both tested inputs. Pinning OpenBLAS to one thread did not recover the loss. Keep the existing CPU-only build for this workload; do not switch the service to the OpenBLAS image based on these results.

## Setup

- Model: `gaianet/All-MiniLM-L6-v2-Embedding-GGUF:F16`
- Runtime: llama.cpp CPU server; no Vulkan
- Host: x86_64 AMD Ryzen 5 3550H, 8 logical CPUs
- Existing CPU service: `http://10.0.1.24:8080`
- OpenBLAS candidate: `http://10.0.1.25:8080`, no published host ports
- Image: `local/llama.cpp:server-cpu-blas`
- Build option: `GGML_BLAS=ON`, OpenBLAS; `libggml-blas.so` and OpenBLAS were mapped in the running process
- Server initialized with 4 llama.cpp threads in both builds
- Client: `aiosonic==1.0.7`, via `scripts/embedding_bench.py`

The model was already present in the shared model cache. Candidate build command:

```bash
docker build --build-arg GGML_BLAS=ON \
  --file .devops/cpu.Dockerfile \
  --target server \
  --tag local/llama.cpp:server-cpu-blas .
```

Each benchmark used 100 measured requests, 5 warmups, and concurrency 8. The short case used the baseline sentence, `The quick brown fox jumps over the lazy dog.` The long case used the longer semantic-search paragraph in the raw JSON results. Three trials were run for each input and configuration. Each run returned 100 vectors of 384 dimensions with norm 1.0.

## Results

Values are arithmetic means across the three trials. Lower latency is better; higher throughput is better.

| Input | Build | OpenBLAS threads | Throughput (req/s) | Avg latency (ms) | Avg p95 (ms) |
| --- | --- | ---: | ---: | ---: | ---: |
| Short | CPU baseline | n/a | 178.24 | 43.23 | 53.45 |
| Short | OpenBLAS | default | 82.66 | 93.83 | 119.29 |
| Short | OpenBLAS | 1 | 86.03 | 90.07 | 119.24 |
| Long | CPU baseline | n/a | 28.23 | 275.93 | 316.78 |
| Long | OpenBLAS | default | 23.40 | 333.07 | 370.45 |
| Long | OpenBLAS | 1 | 23.07 | 337.73 | 383.33 |

Against the CPU baseline, OpenBLAS throughput was 53.6% lower for the short input and 17.1% lower for the long input. With OpenBLAS pinned to one thread, throughput was 51.7% and 18.3% lower, respectively. The short-input result is especially unfavorable; the longer input also showed no gain.

The BLAS plugin and OpenBLAS library were loaded, but these measurements do not prove which individual graph operations were delegated to BLAS. The result is sufficient to reject this configuration for the current service workload, not to conclude that BLAS cannot help other models or larger batched workloads.

## Raw results

Each row above has three per-run JSON files in this directory:

- `minilm-base-{short,long}-r{1,2,3}.json`
- `minilm-blas-{short,long}-r{1,2,3}.json`
- `minilm-blas1-{short,long}-r{1,2,3}.json` (`OPENBLAS_NUM_THREADS=1`)

The original baseline record is in [minilm-embedding-baseline.md](minilm-embedding-baseline.md).
