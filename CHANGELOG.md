# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[semantic versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `GETTING_STARTED.md`, a beginner's guide: what vector search is, and a
  copy-pasteable first index through the CLI, Python and the HTTP console.
- GitHub repository hygiene: CI/recall/MSRV/Python/fuzz workflows, a release
  pipeline (crates, PyPI trusted publishing, GHCR image on `v*` tags),
  Dependabot, issue and PR templates.

### Changed

- README restructured with badges and a documentation map.

### Fixed

- `Segment::open` no longer panics with an overflow when the level directory
  carries a corrupted link count. The header was already CRC-checked before
  those fields were multiplied; the directory is only checksummed in
  `verify()`, so the byte counts it implies are now computed with checked
  arithmetic and rejected as corruption.

## [0.1.0] — 2026-10-06

The first release: a complete single-node vector search engine, built from
scratch, with three front doors and a benchmark suite behind it.

### Added

**Engine (`lodestar-ann-core`)**
- L2, cosine and inner-product distance kernels with runtime dispatch — NEON
  on aarch64, AVX2+FMA on x86_64, scalar fallback — plus batch variants.
- Scalar quantization (SQ8) with a bounded round-trip error, and product
  quantization with k-means++/Lloyd training and asymmetric distance
  computation.
- Exact k-nearest-neighbour search, used as the oracle in tests and benchmarks.
- Fully deterministic: same seed, same data, same results.

**Indexes (`lodestar-ann-index`)**
- HNSW with `m`/`m0`/`ef_construction`/`ef_search` configuration and a
  measured, top-up-free neighbour selection heuristic.
- IVF-flat and IVF-PQ with coarse quantization, residuals and optional exact
  rerank.
- A metadata filter language compiled to a bitset before traversal.
- One graph traversal shared by in-memory and memory-mapped indexes.

**Storage (`lodestar-ann-store`)**
- Immutable, memory-mapped segments with 16-byte-aligned blocks and CRC32 in
  both header and footer.
- A CRC'd write-ahead log with torn-tail truncation, and a JSON manifest
  written atomically.
- Tombstones scoped to the segments that hold the id, plus flush and
  compaction.
- A mapped reader that validates structure before trusting any offset.

**Surfaces**
- `lodestar` CLI: create, insert, delete, search, flush, compact, verify,
  stats, list and sample.
- `lodestar-server`: axum HTTP API with `/healthz`, `/metrics`, `/docs`,
  `/demo`, `/openapi.json` and the `/v1` collection, vector, maintenance and
  search routes; graceful shutdown, CORS opt-in, read-only mode.
- Python package `lodestar` (PyO3, abi3-py39): `Index`, `Collection` and
  `sample`, with type stubs.

**Quality**
- Recall gate in CI: HNSW ≥ 0.90 @ `ef=64`, ≥ 0.95 @ `ef=256`; IVF-PQ with
  rerank ≥ 0.90.
- A crash test that SIGKILLs a child process at random points and asserts
  every acknowledged write survives.
- Corruption tests for the segment reader: flipped bits in the header, footer
  and graph are detected rather than misread, and a property test asserts the
  reader never panics on arbitrary bytes.
- Server integration tests over a real router, and a pytest suite over the
  real extension module.
- Benchmark harness (`make bench`) reporting recall against an exact oracle,
  latency percentiles, throughput and durability, with committed reports in
  `benchmarks/results/`.

[Unreleased]: https://github.com/NavatejR/lodestar/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/NavatejR/lodestar/releases/tag/v0.1.0
