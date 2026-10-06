# Architecture

Lodestar is three libraries and three front doors over them.

```
                ┌──────────────┐  ┌──────────────────┐  ┌───────────────┐
 front doors    │ lodestar CLI │  │ lodestar-server  │  │ Python (pyo3) │
                └──────┬───────┘  └────────┬─────────┘  └──────┬────────┘
                       │                   │                   │
                ┌──────▼───────────────────▼───────────────────▼───────┐
 libraries      │            lodestar-ann-store  (durability)          │
                │  segments · write-ahead log · manifest · tombstones  │
                ├──────────────────────────────────────────────────────┤
                │            lodestar-ann-index  (approximation)       │
                │  HNSW · IVF-PQ · graph traversal · filter bitsets    │
                ├──────────────────────────────────────────────────────┤
                │            lodestar-ann-core    (arithmetic)         │
                │  distance kernels · SQ8/PQ · exact search · RNG      │
                └──────────────────────────────────────────────────────┘
```

Dependencies point downward only: `store` uses `index` and `core`; `index`
uses `core`; `core` uses nothing but `std`. The front doors depend on the
crates they need and never on each other. There is no fourth layer, no plugin
system and no trait object standing between a query and a distance — the
dispatch that exists (SIMD, metric) is resolved once, outside the loop.

## The write path

A collection is a directory:

```
papers/
  manifest.json      format, config, segment list, tombstones
  wal.log            append-only, CRC'd records, replayed on open
  segment-000001.bin immutable, mmap'd, CRC'd header + footer
  segment-000002.bin
  metadata.log       (server only) append-only sidecar: id -> metadata
```

An upsert does four things in a fixed order:

1. append `(upsert, id, vector)` to the write-ahead log and `fsync` it;
2. insert the vector into the in-memory tail (an ordinary HNSW index);
3. record the id in the tail's id map;
4. acknowledge.

The log is written *before* the acknowledgement and *before* the in-memory
index is touched, so a crash between (1) and (4) can lose nothing that was
acknowledged — replay applies the log again, and applying an upsert twice is
idempotent. `flush` runs in the opposite direction: write the sealed segment to
`segment-NAME.tmp-PID`, `fsync`, `rename` (atomic on POSIX), `fsync` the
directory, update the manifest the same way, and only then truncate the log.

Sealed segments are immutable. That is what makes recovery cheap: a segment is
either the complete file it says it is, or it is not a segment at all, and the
open path checks CRCs and structural invariants before trusting a single offset.

## The read path

A search walks every source of live vectors and merges:

```
query ──▶ per-segment traversal (mapped graph, shared scratch) ─┐
       └─▶ tail traversal (in-memory graph) ────────────────────┤
                                                               ▼
                                              filter tombstones, k-way merge, top-k
```

Each traversal is `lodestar_ann_index::graph::search_with_scratch` over a
`GraphView`. The trait is the whole point: an in-memory `Hnsw` and a
memory-mapped `Segment` implement the same interface, so **the code that walks
the graph is literally the same code in both cases**, and there is no separate
"disk search path" to drift out of sync or to under-test.

Scratch buffers (candidate heap, visited set, distance scratch) are allocated
once and reused. In the server they come from a per-worker pool; in the CLI and
Python they are created per call, which is the right trade for a process that
runs one search and exits.

## Deletes

Immutable files cannot be edited, so a delete records a *tombstone* against the
segments that hold the id. Tombstones live in the manifest next to the segments
they hide and vanish when compaction rewrites those segments. Two properties
fall out of the ordering:

* a tombstone only ever applies to segments **older** than the write that
  created it, so a newer copy of an id can never hide itself;
* a tombstone is only recorded against a segment that actually holds the id,
  which keeps `sealed_live` exact rather than an under-count — the id index
  (one sorted `u64` vector per segment, eight bytes per node) is what makes
  that possible.

Compaction rewrites every live vector into a single segment and drops both the
tombstones and the log.

## Concurrency in the service

The engine is synchronous and thread-safe; the service decides how to run it.

* **Every call that touches a collection runs on the blocking pool**
  (`spawn_blocking`). A graph walk is CPU-bound and can take milliseconds; it
  must never sit on an async worker that is supposed to be accepting
  connections.
* **Searches run concurrently, writes exclusively**: one `RwLock` per
  collection, so two searches on different collections do not contend at all,
  and a flush on one does not stall a search on another.
* **Scratch comes from a pool.** A steady-state query allocates nothing beyond
  its result vector.
* **Metadata lives beside the vectors, not in them.** The engine persists ids
  and vectors; the service appends `(id, metadata)` to `metadata.log` and
  replays it at start-up. This keeps the segment format free of a schema it
  would have to version, and keeps filtering a service feature rather than a
  storage feature.
* **Bounded requests.** Points per upsert, queries per search, `k`, and body
  bytes are all capped, because an unbounded request is an unbounded
  allocation.
* **Names never become paths.** Collection names are validated against a fixed
  alphabet before they are joined onto the data root.

## Arithmetic

`lodestar-ann-core` selects a kernel once per index at construction — NEON on
aarch64, AVX2+FMA on x86_64, scalar otherwise — and calls it through a plain
function pointer. There is no per-element branch and no runtime feature test in
the inner loop. Everything is deterministic: same seed, same data, same bits,
across runs and platforms. That is not a nicety; every published recall number
depends on it.

## What is deliberately absent

* **No distributed layer.** One process, one directory. Sharding and
  replication are a different project with different failure modes.
* **No authn/z on the HTTP API.** It is built to run behind a reverse proxy on
  a trusted network; see [SECURITY.md](../SECURITY.md).
* **No on-disk graph traversal.** Vectors are mmap'd and paged by the kernel,
  but the HNSW graph for the tail lives in memory and sealed graphs are walked
  through the mapping. A 50-million-vector index would want the graph tiers on
  disk; that is the next large piece of work.
* **No query planner.** There is one obvious path per index type, and choosing
  it is a parameter (`ef`, `nprobe`), not an optimizer.

See [ALGORITHMS.md](ALGORITHMS.md) for the index internals and
[OPERATIONS.md](OPERATIONS.md) for running it.
