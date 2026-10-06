# Benchmarks

Every number here was produced by [`bench/run.py`](bench/run.py) on the machine
described below, from a fixed seed, and is committed as raw JSON next to its
report in [`benchmarks/results/`](benchmarks/results). Re-run `make bench` and
you will get the same shape of result on yours — the harness prints the
environment it ran under precisely because a benchmark without one is folklore.

**Read this page as a measurement of one laptop, not as a claim about servers.**

```bash
make bench        # 50k × 128, 1,000 queries  -> benchmarks/results/
make bench-full   # 200k × 128, 2,000 queries
make bench-quick  # a few seconds, for the harness itself
```

## Environment

| | |
|---|---|
| Machine | Apple M-series, arm64, 6 cores, 8 GB RAM, macOS 27.0.1 |
| Toolchain | `rustc 1.99.0 (b940084d7 2026-09-28)`, release profile, `lto = "thin"` |
| Python | 3.11.16, numpy 2.4.6, extension built by `maturin develop --release` |
| Data | synthetic clustered corpus (`lodestar.sample`, seed 7), l2, float32 |
| Oracle | exact k-NN over the same vectors, computed with BLAS in numpy |

The oracle matters more than the timer. Recall is only meaningful against exact
search on the same corpus with the same metric, so the harness computes
ground truth first and reports `recall@10` for every `ef` it sweeps.

## 50,000 × 128 — `make bench`

Build: **4.85 s** (10,305 vectors/s). Index memory: **34.1 MB** (683 bytes per
vector, including the ids and the graph).

| ef_search | recall@10 | p50 | p95 | p99 | QPS (1 thread) |
|---:|---:|---:|---:|---:|---:|
| 16 | 0.8014 | 0.025 ms | 0.038 ms | 0.046 ms | 35,681 |
| 32 | 0.9379 | 0.035 ms | 0.046 ms | 0.063 ms | 27,373 |
| 64 | 0.9936 | 0.054 ms | 0.071 ms | 0.097 ms | 17,687 |
| 128 | 0.9999 | 0.079 ms | 0.101 ms | 0.125 ms | 12,142 |
| 256 | 1.0000 | 0.113 ms | 0.142 ms | 0.177 ms | 8,573 |

Throughput across 6 threads at `ef=256`: **25,540 queries/s**. The bindings
release the GIL around the search, so this is real parallelism rather than a
queue behind the interpreter.

**The curve is the story.** `ef=16` is fast and wrong often enough to matter;
`ef=64` buys 19 recall points for 0.03 ms; past `ef=128` you are paying for
recall you already have. Choose `ef` from a recall target, never from habit.

## 200,000 × 128 — `make bench-full`

Build: **55.11 s** (3,629 vectors/s). Index memory: **139.9 MB** (700 bytes per
vector — the graph grows slightly superlinearly with `n`, as HNSW's degree
bound predicts).

| ef_search | recall@10 | p50 | p95 | p99 | QPS (1 thread) |
|---:|---:|---:|---:|---:|---:|
| 16 | 0.5884 | 0.071 ms | 0.098 ms | 0.153 ms | 13,014 |
| 32 | 0.7781 | 0.112 ms | 0.144 ms | 0.174 ms | 8,765 |
| 64 | 0.9246 | 0.272 ms | 0.360 ms | 0.539 ms | 3,587 |
| 128 | 0.9855 | 0.467 ms | 0.566 ms | 0.731 ms | 2,268 |
| 256 | 0.9983 | 0.701 ms | 0.806 ms | 0.985 ms | 1,545 |

Throughput across 6 threads at `ef=256`: **7,126 queries/s**.

The same `ef` is worth less at 4× the corpus: recall at `ef=64` falls from
0.994 to 0.925 because the graph has more ground to cover. That is the
expected behaviour, and it is why the recall gate pins `ef` per corpus size
instead of pinning a single number for all of them.

## Durability

50,000 vectors written through the write-ahead log, then sealed:

| operation | time |
|---|---:|
| insert (log + index, synced once per batch) | **4.88 s** — 10,244 vectors/s |
| flush (seal the tail into a segment) | **4.89 s** |
| verify (re-read every byte, recompute CRCs) | **0.002 s** |
| first search after flush | **0.100 ms** |
| open (map the segment, replay the log) | **0.002 s** |

Two numbers deserve attention. **Verify is 2 ms for 50k vectors** — it re-reads
the footers and walks the graph structure it must check anyway, which is cheap
enough to run on a schedule. **Open is 2 ms** because mapping a file does not
read it: the file's pages arrive on the first query, which is why the search
above is reported at 0.1 ms rather than at 2 ms. The harness reopens the
collection and asserts the count survived, so a benchmark that quietly loses
vectors fails instead of reporting a flattering number.

## The recall gate

Benchmarks are descriptive; the gate is normative. `make gate` runs in release
mode and fails the build if index quality drops:

| index | configuration | requirement |
|---|---|---|
| HNSW | `m=16`, `ef_construction=200` | recall ≥ 0.90 @ `ef=64`, ≥ 0.95 @ `ef=256` |
| IVF-flat | `nlist` tuned, `nprobe=16/64` | recall ≥ 0.99 |
| IVF-PQ | ADC only | recall ≥ 0.50 |
| IVF-PQ | ADC + rerank | recall ≥ 0.90 |

Measured on the gate's own dataset: HNSW **0.9560 @ ef64** and **0.9895 @
ef256**; IVF-flat **1.0000** at both probes; IVF-PQ **0.5945** by ADC and
**0.9895** after rerank. Every floor has headroom, so ordinary jitter in a
shared CI runner cannot fail the build, but a real algorithmic regression
(13 recall points was the last one) cannot pass it either.

## Method notes

* **Percentiles, not means.** p99 is what the unlucky query pays, and it is
  where allocator behaviour and page faults show up.
* **Warm-up is excluded, not hidden.** Each `ef` sweep runs a throwaway query
  before timing begins, so the loop measures steady state; cold-start cost is
  measured separately in the durability table.
* **The harness times the engine, not the timer.** Each sample is one query
  wrapped in `perf_counter`; the wrapper costs well under a microsecond against
  a smallest sample of 25 µs, and `qps_single_thread` is computed from total
  elapsed loop time rather than from the sum of the samples.
* **Synthetic data.** Clustered Gaussian blobs are honest about the index and
  dishonest about your data. Absolute numbers will move on real embeddings;
  the shape of the recall/latency curve will not.
* **`--http` compares the same measurement through `lodestar-server`**, so the
  cost of the service (JSON, base64-free float arrays, a lock) is visible
  rather than assumed:

  ```bash
  make server &
  .venv/bin/python bench/run.py --suite standard --http http://127.0.0.1:8080
  ```
