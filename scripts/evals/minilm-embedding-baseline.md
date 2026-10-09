# MiniLM embedding baseline

Recorded: 2026-09-27

## Target

- Service: `minilm-embeddings`
- Endpoint: `http://10.0.1.24:8080/v1/embeddings`
- Model: `all-MiniLM-L6-v2`
- GGUF source: `gaianet/All-MiniLM-L6-v2-Embedding-GGUF:F16`
- Runtime: CPU-only llama.cpp server, no Vulkan
- Vector size: 384

## Benchmark configuration

- Requests: 100
- Warmup requests: 5
- Concurrency: 8
- Text: `The quick brown fox jumps over the lazy dog.`
- Client: `aiosonic==1.0.7`

## Results

| Metric | Value |
| --- | ---: |
| Successful requests | 100/100 |
| Elapsed time | 0.540 s |
| Throughput | 185.07 requests/s |
| Latency min | 7.93 ms |
| Latency average | 41.53 ms |
| Latency p50 | 43.16 ms |
| Latency p95 | 48.35 ms |
| Latency p99 | 51.79 ms |
| Latency max | 53.21 ms |
| Vector dimensions | 384 |
| Vector norm | 1.000000 |

## Reproduce

Run from the repository root:

```bash
uv run --script scripts/embedding_bench.py \
  --requests 100 \
  --warmup 5 \
  --concurrency 8 \
  --output /tmp/minilm-embedding-baseline.json
```

Keep the request count, warmup, concurrency, text, model, and endpoint unchanged when comparing later builds.
