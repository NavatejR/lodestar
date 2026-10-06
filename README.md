# Lodestar

**A vector search engine written from scratch in Rust.** Graph and
inverted-file indexes, quantization, a crash-safe storage engine, a
write-ahead log, an HTTP API, Python bindings and a CLI — no search library
underneath, only `std`, `axum`, `serde`, `pyo3` and `numpy`.

Everything runs on your machine. Embeddings come from a local model or from
your own generator; the demo runs in a container with nothing leaving it.

```bash
git clone https://github.com/NavatejR/lodestar && cd lodestar
make setup && make test      # build the workspace and run every suite
make demo                    # http://localhost:8080/demo
```

| | |
|---|---|
| **Indexes** | HNSW (default), IVF-flat, IVF-PQ with asymmetric distance and optional rerank |
| **Metrics** | L2, cosine, inner product — each with NEON / AVX2+FMA / scalar dispatch |
| **Storage** | immutable memory-mapped segments, a write-ahead log, a manifest, CRC32 throughout |
| **Durability** | SIGKILL-consistent: acknowledged writes survive, torn tails are truncated, `verify` re-checks every byte |
| **Surfaces** | `lodestar` CLI · `lodestar-server` HTTP API · `pip`-installable Python package |
| **Quality** | recall gate in CI, property tests, a crash test that kills the process at random points, fuzzed parsers |

## Quick start

### Python

`make wheel && pip install dist/*.whl` builds and installs the package — one
abi3 wheel covers CPython 3.9 and newer. (`make build-py` installs it into the
repo's own virtualenv instead, which is what `make test-py` uses.)

```python
import numpy as np
import lodestar

ids = np.arange(10_000, dtype=np.uint64)
vec = np.random.randn(10_000, 128).astype(np.float32)

# An in-memory HNSW index.
index = lodestar.Index(dim=128, metric="cosine")
index.add_batch(vec, ids)
hits = index.search(np.random.randn(128).astype(np.float32), k=5, ef=64)
# -> [(id, distance), ...]

# A durable collection: write-ahead logged, memory-mapped, crash-safe.
collection = lodestar.Collection.create("./data", "papers", 128, "cosine")
collection.add_batch(vec, ids)
collection.flush()          # seals the tail into an immutable segment
collection.search(vec[0], k=10)
```

### CLI

```bash
lodestar create papers --dim 384 --metric cosine
lodestar insert papers --file vectors.jsonl        # {"id": 1, "vector": [...]}
lodestar search papers --file query.jsonl --k 10 --ef 64
lodestar compact papers && lodestar verify papers
lodestar sample demo --count 20000 --dim 128       # a synthetic corpus to play with
```

`cargo install --path crates/cli` puts `lodestar` on your `PATH`; `make release`
builds the CLI and the server optimised without installing.

### HTTP

```bash
make server                                        # or: cargo run --release -p lodestar-ann-server
curl -s localhost:8080/v1/index/papers -X PUT -H 'content-type: application/json' \
  -d '{"dim": 384, "metric": "cosine"}'
curl -s localhost:8080/v1/index/papers/upsert -X POST -H 'content-type: application/json' \
  -d '{"points": [{"id": 1, "vector": [...], "metadata": {"group": "news"}}]}'
curl -s localhost:8080/v1/search -H 'content-type: application/json' \
  -d '{"index": "papers", "query": [...], "k": 10, "filter": "group == news"}'
```

`/docs` serves a self-contained API reference, `/openapi.json` the machine-readable
document, `/demo` a console over the same API, and `/metrics` Prometheus text.
OpenAPI, metrics and the console are all compiled into the binary — no CDN, no
second container.

## Measured, not asserted

`make bench` regenerates every number below on your machine; the committed
reports live in [`benchmarks/results/`](benchmarks/results) and the full
write-up in [BENCHMARKS.md](BENCHMARKS.md). Numbers below are from an Apple
M-series laptop (6 cores, 8 GB, `rustc 1.99.0`), 50,000 × 128 float32, l2,
1,000 held-out queries, recall against exact brute force:

| ef_search | recall@10 | p50 | p99 | single-thread QPS |
|---:|---:|---:|---:|---:|
| 16 | 0.801 | 0.025 ms | 0.046 ms | 35,681 |
| 64 | 0.994 | 0.054 ms | 0.097 ms | 17,687 |
| 256 | 1.000 | 0.113 ms | 0.177 ms | 8,573 |

At 200,000 vectors the same index reaches 0.998 recall at `ef=256` with a
0.70 ms median. Writes run at ~10,000 vectors/s through the log, and re-opening
a sealed collection — mapping the segment, replaying the log — takes 2 ms.

## How it works

```
        ┌────────────┐   ┌──────────────┐   ┌──────────────────────────┐
 query ─┼ distance   ─┼──▶ index crate  ─┼──▶ HNSW traversal, IVF scan  │
        │ kernels    │   │ (graph/list) │   │ + compiled filter bitset  │
        └────────────┘   └──────┬───────┘   └────────────┬─────────────┘
                                │ ids + vectors          │ top-k
                        ┌───────▼────────────────────────▼─────────────┐
                        │ store crate: segments (mmap) · WAL · manifest │
                        └───────────────────────────────────────────────┘
```

* **`lodestar-ann-core`** — distance kernels with runtime CPU dispatch,
  scalar and product quantization, the exact-search oracle, a seeded RNG.
  Everything is deterministic: same seed, same bits.
* **`lodestar-ann-index`** — HNSW, IVF(-PQ), and a metadata filter language
  compiled to a bitset before traversal, so filtering costs no callback.
* **`lodestar-ann-store`** — append-only write-ahead log, immutable segments
  with 16-byte-aligned blocks and CRC32 in the header and footer, a JSON
  manifest written atomically, and a mapped reader that validates structure
  before it trusts a single offset.
* **`lodestar-ann-cli` / `lodestar-ann-server` / `lodestar-ann-py`** — three
  front doors over the same three crates.

Longer explanations: [ARCHITECTURE](docs/ARCHITECTURE.md) for the layout and
the concurrency model, [ALGORITHMS](docs/ALGORITHMS.md) for HNSW, IVF-PQ and
the recall measurements, [OPERATIONS](docs/OPERATIONS.md) for running it,
and [docs/adr](docs/adr) for the decisions worth arguing about.

## Repository layout

```
crates/core      distance, quantization, exact search          lodestar-ann-core
crates/index     HNSW, IVF-PQ, filters, graph traversal        lodestar-ann-index
crates/store     segments, WAL, manifest, crash-safe collections  lodestar-ann-store
crates/cli       the `lodestar` binary
crates/server    `lodestar-server`, the HTTP API and console
crates/py        PyO3 bindings (abi3, built by maturin)
python/          the typed `lodestar` package, stubs and pytest suite
bench/           the benchmark harness behind `make bench`
benchmarks/      committed benchmark reports (JSON + Markdown)
demo/            Dockerfile, compose file, bundled corpus
docs/            architecture, algorithms, operations, ADRs
```

## Development

| command | what it does |
|---|---|
| `make setup` | toolchain components + a full build |
| `make test` | every Rust suite in the workspace |
| `make test-py` | maturin + the Python suite (the extension is `extension-module`, so cargo cannot link it) |
| `make gate` | the ignored recall gate, release mode — what CI enforces |
| `make lint` | `fmt --check` and `clippy -D warnings` |
| `make bench` | regenerate the benchmark report into `benchmarks/results/` |
| `make verify` | lint + test + test-py + gate: what a reviewer should run |

The test story is deliberate rather than decorative:

* **unit and property tests** for kernels, quantizers, the filter compiler and
  the segment reader — including that a flipped bit in a header, footer or
  graph is detected rather than misread, and that arbitrary bytes never panic
  a reader;
* **a recall gate**: HNSW at `m=16, efC=200` must hold ≥ 0.90 recall at
  `ef=64` and ≥ 0.95 at `ef=256`; IVF-PQ with rerank must hold ≥ 0.90. The
  gate runs in release mode and fails the build;
* **a crash test** that SIGKILLs a child process at random points during
  writes and then asserts every acknowledged write is still there;
* **server integration tests** over a real `axum` router, and a pytest suite
  over the real extension module.

## Status

v0.1.0: feature-complete for a single-node engine. Not yet: distributed
sharding, a replication story, ANN indexes on disk (searches currently touch
the in-memory graph over mmap'd vectors), and authn/z on the HTTP API — it is
built to run behind a reverse proxy on a trusted network.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Bug reports want a reproducer;
regressions want a failing test first. Security reports go to
[SECURITY.md](SECURITY.md), not the issue tracker.

## License

Licensed under either of [Apache License, Version 2.0](LICENSE-APACHE) or
[MIT license](LICENSE-MIT) at your option.
