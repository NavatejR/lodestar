#!/usr/bin/env python3
"""Lodestar benchmark harness.

Measures what a reviewer would ask about, on this machine, and writes both the
raw numbers and a readable report:

* **Build** — how long HNSW takes to index a synthetic clustered corpus, and
  how many vectors per second that is.
* **Recall** — recall@10 against an exact brute-force oracle (numpy), for each
  `ef_search` we sweep. Recall without an oracle is a number without a meaning.
* **Latency and throughput** — p50/p95/p99 per query, single threaded, plus a
  thread-pool run for queries per second.
* **Durability** — insert throughput through the write-ahead log, flush, and the
  cost of re-opening a collection from its memory-mapped segment.
* **HTTP** — when `--http` names a running `lodestar-server`, the same
  measurement through the API, so the overhead of the service is visible.

Everything runs locally, from the same fixture seeds, so two runs on the same
machine are comparable. Numbers from a laptop are not numbers from a server;
the report says which machine it ran on.

Usage:
    bench/run.py --suite standard
    bench/run.py --suite full --http http://127.0.0.1:8080
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import subprocess
import sys
import tempfile
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
# The package lives in the source tree, not in site-packages, so that a bench
# run always measures the checkout it was launched from.
sys.path.insert(0, str(REPO / "python"))

import lodestar  # noqa: E402  (after the path is set up)

RESULTS = REPO / "benchmarks" / "results"

# Corpus sizes per suite: (vectors, dimensions, queries).
SUITES = {
    "quick": (10_000, 64, 200),
    "standard": (50_000, 128, 1_000),
    "full": (200_000, 128, 2_000),
}

# Candidate list sizes to sweep. Recall rises and latency grows with `ef`;
# that trade-off is the whole story of a graph index.
EF_SWEEP = (16, 32, 64, 128, 256)

K = 10


def corpus(suite: str, seed: int = 7):
    """A clustered corpus plus `queries` held out from it."""
    count, dim, queries = SUITES[suite]
    ids, vectors = lodestar.sample(count=count + queries, dim=dim, metric="l2", clusters=64, seed=seed)
    vectors = np.asarray(vectors, dtype=np.float32)
    ids = np.asarray(ids, dtype=np.uint64)
    return ids[:count], vectors[count:], vectors[:count]  # indexed ids, queries, indexed vectors


def exact_neighbours(vectors: np.ndarray, queries: np.ndarray, k: int) -> np.ndarray:
    """Ground-truth top-k ids by squared euclidean distance, via BLAS."""
    norms_v = np.einsum("ij,ij->i", vectors, vectors)[None, :]
    norms_q = np.einsum("ij,ij->i", queries, queries)[:, None]
    distances = norms_q + norms_v - 2.0 * (queries @ vectors.T)
    # `argpartition` is O(n) per row, which keeps the oracle cheap enough to
    # run inside every suite.
    top = np.argpartition(distances, k, axis=1)[:, :k]
    rows = np.arange(distances.shape[0])[:, None]
    order = np.argsort(distances[rows, top], axis=1)
    return top[rows, order]


def percentiles(samples: list[float]) -> dict:
    """p50/p95/p99 in milliseconds, from per-query seconds."""
    if not samples:
        return {}
    ordered = sorted(samples)
    def pick(fraction: float) -> float:
        index = min(len(ordered) - 1, int(round(fraction * (len(ordered) - 1))))
        return ordered[index] * 1_000.0
    return {
        "mean_ms": statistics.fmean(ordered) * 1_000.0,
        "p50_ms": pick(0.50),
        "p95_ms": pick(0.95),
        "p99_ms": pick(0.99),
        "max_ms": ordered[-1] * 1_000.0,
    }


def recall_at_k(truth: np.ndarray, found: list[list[tuple[int, float]]], k: int) -> float:
    hits = 0
    for row, results in zip(truth, found):
        expected = set(int(value) for value in row)
        hits += sum(1 for identifier, _ in results if identifier in expected)
    return hits / (len(truth) * k)


def index_benchmark(suite: str, seed: int = 7) -> dict:
    ids, queries, vectors = corpus(suite, seed)
    count, dim, queries_count = SUITES[suite]
    print(f"  corpus: {count:,} x {dim} vectors, {queries_count} queries", flush=True)

    started = time.perf_counter()
    truth = exact_neighbours(vectors, queries, K)
    oracle_seconds = time.perf_counter() - started
    print(f"  exact oracle: {oracle_seconds:.2f}s", flush=True)

    index = lodestar.Index(dim=dim, metric="l2", m=16, m0=32, ef_construction=200, seed=1)
    started = time.perf_counter()
    index.add_batch(vectors, ids)
    build_seconds = time.perf_counter() - started
    print(
        f"  build: {build_seconds:.2f}s ({count / build_seconds:,.0f} vectors/s)",
        flush=True,
    )

    memory = index.memory_bytes
    sweep = []
    for ef in EF_SWEEP:
        # Warm the scratch buffer, then measure.
        index.search(queries[0], k=K, ef=ef)
        samples = []
        found = []
        started = time.perf_counter()
        for query in queries:
            begin = time.perf_counter()
            found.append(index.search(query, k=K, ef=ef))
            samples.append(time.perf_counter() - begin)
        elapsed = time.perf_counter() - started
        row = {
            "ef_search": ef,
            "recall_at_10": recall_at_k(truth, found, K),
            "qps_single_thread": len(queries) / elapsed,
            **percentiles(samples),
        }
        sweep.append(row)
        print(
            f"  ef={ef:<4} recall@10 {row['recall_at_10']:.4f}  "
            f"p50 {row['p50_ms']:.3f}ms  qps {row['qps_single_thread']:,.0f}",
            flush=True,
        )

    # Throughput with the GIL released, which is what the bindings do.
    workers = max(1, os.cpu_count() or 1)
    best = sweep[-1]["ef_search"] if sweep else 64
    queries_per_worker = max(1, len(queries) // workers)

    def worker(offset: int) -> int:
        total = 0
        for query in queries[offset : offset + queries_per_worker]:
            index.search(query, k=K, ef=best)
            total += 1
        return total

    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        served = sum(pool.map(worker, [i * queries_per_worker for i in range(workers)]))
    pool_seconds = time.perf_counter() - started
    print(
        f"  {workers} threads: {served / pool_seconds:,.0f} queries/s (ef={best})",
        flush=True,
    )

    return {
        "vectors": count,
        "dim": dim,
        "queries": queries_count,
        "metric": "l2",
        "graph": {"m": 16, "m0": 32, "ef_construction": 200},
        "build_seconds": build_seconds,
        "build_vectors_per_second": count / build_seconds,
        "oracle_seconds": oracle_seconds,
        "memory_bytes": memory,
        "bytes_per_vector": memory / count,
        "sweep": sweep,
        "throughput": {
            "threads": workers,
            "ef_search": best,
            "queries": served,
            "seconds": pool_seconds,
            "qps": served / pool_seconds,
        },
    }


def durability_benchmark(suite: str) -> dict:
    """Write-ahead-log insert throughput, flush, and reopen from the mapping."""
    count, dim, _ = SUITES[suite]
    count = min(count, 50_000)
    _, _, vectors = corpus(suite)
    vectors = vectors[:count]
    ids = np.arange(count, dtype=np.uint64)

    with tempfile.TemporaryDirectory() as directory:
        collection = lodestar.Collection.create(directory, "bench", dim, "l2")
        started = time.perf_counter()
        collection.add_batch(vectors, ids)
        insert_seconds = time.perf_counter() - started
        print(
            f"  logged insert: {insert_seconds:.2f}s ({count / insert_seconds:,.0f} vectors/s)",
            flush=True,
        )

        started = time.perf_counter()
        segment = collection.flush()
        flush_seconds = time.perf_counter() - started
        assert segment is not None

        started = time.perf_counter()
        collection.verify()
        verify_seconds = time.perf_counter() - started

        started = time.perf_counter()
        collection.search(vectors[0], k=K, ef=64)
        search_seconds = time.perf_counter() - started
        del collection

        started = time.perf_counter()
        reopened = lodestar.Collection.open(directory, "bench")
        open_seconds = time.perf_counter() - started
        assert reopened.len == count, "the reopened collection lost vectors"
        del reopened

        print(
            f"  flush {flush_seconds:.2f}s, verify {verify_seconds:.2f}s, "
            f"open {open_seconds:.3f}s",
            flush=True,
        )

    return {
        "vectors": count,
        "dim": dim,
        "insert_seconds": insert_seconds,
        "insert_vectors_per_second": count / insert_seconds,
        "flush_seconds": flush_seconds,
        "verify_seconds": verify_seconds,
        "open_seconds": open_seconds,
        "first_search_seconds": search_seconds,
    }


def http_benchmark(base: str, suite: str, workers: int = 8) -> dict:
    """Measures the running service, assuming `demo` already holds a corpus."""
    base = base.rstrip("/")

    def request(path: str, payload: dict | None = None):
        data = None if payload is None else json.dumps(payload).encode()
        method = "GET" if payload is None else "POST"
        req = urllib.request.Request(
            base + path, data=data, method=method, headers={"content-type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=60) as response:
            return json.loads(response.read())

    health = request("/healthz")
    stats = request("/v1/index/demo/stats")
    dim = stats["dim"]
    if stats["live"] == 0:
        raise SystemExit("index `demo` is empty; run the demo seeder first")
    vector = [0.05] * dim

    def timed(k: int) -> float:
        begin = time.perf_counter()
        request("/v1/search", {"index": "demo", "query": vector, "k": k})
        return time.perf_counter() - begin

    sweeps = {}
    for k in (1, 10, 100):
        timed(k)
        samples = [timed(k) for _ in range(200)]
        sweeps[f"k={k}"] = {
            "recall_at_10": None,
            **percentiles(samples),
            "qps_single_thread": len(samples) / sum(samples),
        }
        print(
            f"  http k={k:<4} p50 {sweeps[f'k={k}']['p50_ms']:.3f}ms  "
            f"p99 {sweeps[f'k={k}']['p99_ms']:.3f}ms",
            flush=True,
        )

    def worker(_: int) -> float:
        begin = time.perf_counter()
        timed(10)
        return time.perf_counter() - begin

    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(worker, range(workers * 25)))
    elapsed = time.perf_counter() - started
    throughput = (workers * 25) / elapsed
    print(f"  http {workers} threads: {throughput:,.0f} searches/s", flush=True)

    return {
        "base_url": base,
        "service": {k: health[k] for k in ("version", "kernel")},
        "live_vectors": stats["live"],
        "segments": stats["segments"],
        "sweep": sweeps,
        "throughput": {"threads": workers, "requests": workers * 25, "qps": throughput},
    }


def environment() -> dict:
    """What this ran on, so a number is never read without its machine."""
    rust = "unknown"
    try:
        rust = subprocess.run(
            ["rustc", "--version"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except Exception:  # pragma: no cover - informational only
        pass
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or "unknown",
        "cpu_count": os.cpu_count(),
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "lodestar": lodestar.__version__,
        "rustc": rust,
    }


def render(report: dict) -> str:
    """A markdown report a reviewer can read without the JSON."""
    lines = ["# Lodestar benchmark report", ""]
    env = report["environment"]
    lines += [
        f"* suite **{report['suite']}**, generated {env['timestamp']}",
        f"* {env['platform']} ({env['machine']}), {env['cpu_count']} cores",
        f"* lodestar {env['lodestar']}, python {env['python']}, numpy {env['numpy']}",
        f"* `{env['rustc']}`",
        "",
    ]
    if "index" in report:
        index = report["index"]
        lines += [
            "## In-memory HNSW",
            "",
            f"Corpus: **{index['vectors']:,} x {index['dim']}** float32, "
            f"{index['queries']:,} held-out queries, {index['metric']}.",
            "",
            f"Build: **{index['build_seconds']:.2f}s** "
            f"({index['build_vectors_per_second']:,.0f} vectors/s). "
            f"Memory: **{index['memory_bytes'] / 1e6:.1f} MB** "
            f"({index['bytes_per_vector']:.0f} bytes/vector).",
            "",
            "| ef_search | recall@10 | p50 ms | p95 ms | p99 ms | QPS (1 thread) |",
            "|---:|---:|---:|---:|---:|---:|",
        ]
        for row in index["sweep"]:
            lines.append(
                f"| {row['ef_search']} | {row['recall_at_10']:.4f} | {row['p50_ms']:.3f} | "
                f"{row['p95_ms']:.3f} | {row['p99_ms']:.3f} | {row['qps_single_thread']:,.0f} |"
            )
        throughput = index["throughput"]
        lines += [
            "",
            f"Throughput: **{throughput['qps']:,.0f} queries/s** across "
            f"{throughput['threads']} threads at ef={throughput['ef_search']} "
            f"(GIL released by the bindings).",
            "",
        ]
    if "durability" in report:
        durable = report["durability"]
        lines += [
            "## Durable collection",
            "",
            f"Inserting {durable['vectors']:,} vectors through the write-ahead log: "
            f"**{durable['insert_seconds']:.2f}s** "
            f"({durable['insert_vectors_per_second']:,.0f} vectors/s).",
            "",
            f"| operation | seconds |",
            "|---|---:|",
            f"| flush (seal the tail into a segment) | {durable['flush_seconds']:.3f} |",
            f"| verify (every checksum, re-read) | {durable['verify_seconds']:.3f} |",
            f"| open (map the segment, replay the log) | {durable['open_seconds']:.3f} |",
            f"| first search after open | {durable['first_search_seconds'] * 1_000:.3f} ms |",
            "",
        ]
    if "http" in report:
        http = report["http"]
        lines += [
            "## HTTP service",
            "",
            f"Serving {http['live_vectors']:,} vectors in {http['segments']} segment(s) "
            f"from `{http['base_url']}` "
            f"(lodestar {http['service']['version']}, kernel {http['service']['kernel']}).",
            "",
            "| query | p50 ms | p95 ms | p99 ms | QPS (1 thread) |",
            "|---|---:|---:|---:|---:|",
        ]
        for name, row in http["sweep"].items():
            lines.append(
                f"| {name} | {row['p50_ms']:.3f} | {row['p95_ms']:.3f} | "
                f"{row['p99_ms']:.3f} | {row['qps_single_thread']:,.0f} |"
            )
        lines += [
            "",
            f"Throughput: **{http['throughput']['qps']:,.0f} searches/s** across "
            f"{http['throughput']['threads']} concurrent clients.",
            "",
        ]
    lines += [
        "---",
        "",
        "Generated by `bench/run.py`; the raw numbers are next to this file as JSON.",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--suite", choices=sorted(SUITES), default="standard")
    parser.add_argument("--out", type=Path, default=RESULTS)
    parser.add_argument("--http", help="also benchmark a running lodestar-server")
    parser.add_argument("--skip-index", action="store_true")
    args = parser.parse_args()

    report: dict = {"suite": args.suite, "environment": environment()}
    if not args.skip_index:
        print("in-memory HNSW", flush=True)
        report["index"] = index_benchmark(args.suite)
        print("durable collection", flush=True)
        report["durability"] = durability_benchmark(args.suite)
    if args.http:
        print("http service", flush=True)
        report["http"] = http_benchmark(args.http, args.suite)

    args.out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    payload = args.out / f"{stamp}-{args.suite}.json"
    payload.write_text(json.dumps(report, indent=2) + "\n")
    markdown = args.out / f"{stamp}-{args.suite}.md"
    markdown.write_text(render(report))
    print(f"\nwrote {payload.relative_to(REPO)} and {markdown.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
