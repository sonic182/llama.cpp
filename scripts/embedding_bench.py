#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "aiosonic==1.0.7",
# ]
# ///

"""Benchmark an OpenAI-compatible llama.cpp embedding endpoint."""

import argparse
import asyncio
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import statistics
import time
from typing import Any

from aiosonic import HTTPClient, TCPConnector, Timeouts


DEFAULT_URL = "http://10.0.1.24:8080"
DEFAULT_MODEL = "all-MiniLM-L6-v2"
DEFAULT_TEXT = "The quick brown fox jumps over the lazy dog."


@dataclass(slots=True)
class RequestResult:
    latency_ms: float
    status_code: int | None = None
    dimensions: int | None = None
    norm: float | None = None
    error: str | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default=DEFAULT_URL, help="llama-server base URL")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--text", default=DEFAULT_TEXT)
    parser.add_argument("--requests", type=int, default=100, dest="request_count")
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--timeout", type=float, default=60.0, help="per-request timeout in seconds")
    parser.add_argument(
        "--expected-dimensions",
        type=int,
        default=384,
        help="expected vector size; use 0 to disable this check",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="write the result summary as JSON for later comparison",
    )
    args = parser.parse_args()
    if args.request_count < 1:
        parser.error("--requests must be at least 1")
    if args.concurrency < 1:
        parser.error("--concurrency must be at least 1")
    if args.warmup < 0:
        parser.error("--warmup must not be negative")
    if args.expected_dimensions < 0:
        parser.error("--expected-dimensions must not be negative")
    return args


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * weight


def embedding_from_payload(payload: Any, expected_dimensions: int) -> tuple[int, float]:
    data = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(data, list) or not data or not isinstance(data[0], dict):
        raise ValueError("response does not contain an embedding data item")

    embedding = data[0].get("embedding")
    if not isinstance(embedding, list) or not embedding:
        raise ValueError("response does not contain a non-empty embedding")
    if any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in embedding):
        raise ValueError("embedding contains a non-numeric value")

    dimensions = len(embedding)
    if expected_dimensions and dimensions != expected_dimensions:
        raise ValueError(f"expected {expected_dimensions} dimensions, got {dimensions}")
    norm = math.sqrt(math.fsum(value * value for value in embedding))
    return dimensions, norm


async def request_embedding(
    client: HTTPClient,
    endpoint: str,
    model: str,
    text: str,
    timeout: Timeouts,
    expected_dimensions: int,
    semaphore: asyncio.Semaphore,
) -> RequestResult:
    async with semaphore:
        started = time.perf_counter()
        status_code: int | None = None
        try:
            response = await client.post(
                endpoint,
                json={"input": text, "model": model},
                timeouts=timeout,
            )
            status_code = response.status_code
            if not response.ok:
                body = (await response.text()).strip().replace("\n", " ")
                raise RuntimeError(f"HTTP {status_code}: {body[:200]}")
            dimensions, norm = embedding_from_payload(await response.json(), expected_dimensions)
            error = None
        except Exception as exc:  # benchmark should report individual failures
            dimensions = None
            norm = None
            error = f"{type(exc).__name__}: {exc}"
        return RequestResult(
            latency_ms=(time.perf_counter() - started) * 1000,
            status_code=status_code,
            dimensions=dimensions,
            norm=norm,
            error=error,
        )


async def run_batch(
    client: HTTPClient,
    endpoint: str,
    model: str,
    text: str,
    count: int,
    concurrency: int,
    timeout: Timeouts,
    expected_dimensions: int,
) -> list[RequestResult]:
    semaphore = asyncio.Semaphore(concurrency)
    tasks = [
        request_embedding(
            client,
            endpoint,
            model,
            text,
            timeout,
            expected_dimensions,
            semaphore,
        )
        for _ in range(count)
    ]
    return await asyncio.gather(*tasks)


def summarize(
    args: argparse.Namespace,
    endpoint: str,
    elapsed_s: float,
    results: list[RequestResult],
) -> dict[str, Any]:
    successes = [result for result in results if result.error is None]
    latencies = [result.latency_ms for result in successes]
    norms = [result.norm for result in successes if result.norm is not None]
    dimensions = Counter(result.dimensions for result in successes)
    errors = Counter(result.error for result in results if result.error is not None)
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "url": args.url,
        "endpoint": endpoint,
        "model": args.model,
        "text_length": len(args.text),
        "requests": len(results),
        "successes": len(successes),
        "failures": len(results) - len(successes),
        "concurrency": args.concurrency,
        "warmup": args.warmup,
        "elapsed_s": elapsed_s,
        "requests_per_second": len(successes) / elapsed_s if elapsed_s else 0.0,
        "latency_ms": {
            "min": min(latencies) if latencies else 0.0,
            "avg": statistics.fmean(latencies) if latencies else 0.0,
            "p50": percentile(latencies, 0.50),
            "p95": percentile(latencies, 0.95),
            "p99": percentile(latencies, 0.99),
            "max": max(latencies) if latencies else 0.0,
        },
        "vector_dimensions": {str(key): value for key, value in dimensions.items()},
        "vector_norm": {
            "min": min(norms) if norms else 0.0,
            "avg": statistics.fmean(norms) if norms else 0.0,
            "max": max(norms) if norms else 0.0,
        },
        "errors": dict(errors),
    }


def print_summary(summary: dict[str, Any]) -> None:
    latency = summary["latency_ms"]
    print(f"endpoint:    {summary['endpoint']}")
    print(f"model:       {summary['model']}")
    print(f"requests:    {summary['successes']}/{summary['requests']} succeeded")
    print(f"elapsed:     {summary['elapsed_s']:.3f} s")
    print(f"throughput:  {summary['requests_per_second']:.2f} requests/s")
    print(
        "latency ms:  "
        f"min={latency['min']:.2f} avg={latency['avg']:.2f} "
        f"p50={latency['p50']:.2f} p95={latency['p95']:.2f} "
        f"p99={latency['p99']:.2f} max={latency['max']:.2f}"
    )
    print(f"dimensions:   {summary['vector_dimensions']}")
    print(
        "norm:         "
        f"min={summary['vector_norm']['min']:.6f} "
        f"avg={summary['vector_norm']['avg']:.6f} "
        f"max={summary['vector_norm']['max']:.6f}"
    )
    if summary["errors"]:
        print(f"errors:       {summary['errors']}")


async def benchmark(args: argparse.Namespace) -> dict[str, Any]:
    endpoint = f"{args.url.rstrip('/')}/v1/embeddings"
    timeout = Timeouts(
        sock_connect=args.timeout,
        sock_read=args.timeout,
        request_timeout=args.timeout,
    )
    client = HTTPClient(connector=TCPConnector(timeouts=timeout))
    try:
        if args.warmup:
            warmup_results = await run_batch(
                client,
                endpoint,
                args.model,
                args.text,
                args.warmup,
                args.concurrency,
                timeout,
                args.expected_dimensions,
            )
            warmup_failures = [result.error for result in warmup_results if result.error]
            if warmup_failures:
                raise RuntimeError(f"warmup failed: {warmup_failures[0]}")

        started = time.perf_counter()
        results = await run_batch(
            client,
            endpoint,
            args.model,
            args.text,
            args.request_count,
            args.concurrency,
            timeout,
            args.expected_dimensions,
        )
        elapsed_s = time.perf_counter() - started
    finally:
        await client.connector.cleanup()

    return summarize(args, endpoint, elapsed_s, results)


def main() -> None:
    args = parse_args()
    summary = asyncio.run(benchmark(args))
    print_summary(summary)
    if args.output:
        args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        print(f"saved:        {args.output}")


if __name__ == "__main__":
    main()
