# Operations

Lodestar is one binary and one directory. This page is what an operator needs
after the demo: how it starts, what it writes, what to watch, and what to do
when a byte is wrong.

## Running the service

```bash
lodestar-server --root /var/lib/lodestar --addr 0.0.0.0:8080
```

| flag | environment | default | meaning |
|---|---|---|---|
| `--root` | `LODESTAR_ROOT` | `./data` | one subdirectory per collection |
| `--addr` | `LODESTAR_ADDR` | `127.0.0.1:8080` | `host:port`, `:port`, or a bare port |
| `--cors` | `LODESTAR_CORS` | *(off)* | comma-separated origins, or `*` |
| `--read-only` | `LODESTAR_READ_ONLY` | off | refuse every mutation with `403` |
| `--log` | `LODESTAR_LOG` | `info` | tracing filter, e.g. `lodestar_ann_server=debug` |

The default address is loopback. Exposing it means choosing an auth story,
which this service does not have — see [SECURITY.md](../SECURITY.md).

### Start-up semantics

At start-up the root is scanned and **every** collection with a manifest is
opened, and its segment headers and footers are validated. If one of them
cannot be opened, the process exits non-zero: a corrupt collection is not
silently skipped when it decides whether the service starts. To bring the
service up while you investigate, move that collection's directory out of the
root; it is a directory, and nothing else references it.

Collections created *after* start-up (by the CLI, by the Python bindings, by
another process sharing the volume) are picked up when `GET /v1/index` is
called. That call re-scans the root, so listing is also the discovery moment.

Graceful shutdown is on SIGINT and SIGTERM: the listener stops accepting,
in-flight requests finish, the process exits 0.

## What is on disk

```
/var/lib/lodestar/
  papers/
    manifest.json        format version, config, segment list, tombstones
    wal.log              append-only write-ahead log of unsealed writes
    segment-000001.bin   immutable, mmap'd, CRC32 in header and footer
    segment-000002.bin
    metadata.log         (server) id -> metadata, replayed at open
```

**Back up the whole directory.** Segments and the manifest are consistent at
any rename boundary because both are written temp-file → fsync → rename →
fsync-directory; the log is the only file that grows in place, and a torn tail
on it is truncated on replay rather than being an error. Copying a live
directory is safe if you copy the manifest last.

`metadata.log` is server-owned. The CLI and Python bindings do not write it,
so metadata created through the API is not visible to them — that is a
boundary, not a bug.

## Monitoring

`GET /metrics` is Prometheus text. The series worth alerting on:

| series | meaning | why it matters |
|---|---|---|
| `lodestar_errors_total` | failures by class | should be flat; a rise is a client or a bug |
| `lodestar_engine_failures_total` | the engine returned an error | non-zero means look at logs now |
| `lodestar_http_requests_total{status="5xx"}` | server errors by route | the page-level signal |
| `lodestar_search_duration_seconds` | search wall time summary | the latency SLO lives here |
| `lodestar_collection_pending_vectors` | vectors in the tail | if it never falls, nothing is flushing |
| `lodestar_collection_segments` | segments per collection | growth drives compaction |
| `lodestar_uptime_seconds` | process age | restart loops are visible here |

`GET /healthz` is liveness only — version, active SIMD kernel, uptime,
collection count. It does not touch the data, so a slow disk cannot make the
process look dead.

## Maintenance

| task | command | cost |
|---|---|---|
| seal the tail | `POST /v1/index/{n}/flush` or `lodestar flush n` | writes one segment, truncates the log |
| drop tombstones | `POST /v1/index/{n}/compact` or `lodestar compact n` | rewrites every live vector |
| check every byte | `POST /v1/index/{n}/verify` or `lodestar verify n` | ~2 ms per 50k vectors |

Flush when the pending count grows (the console and
`lodestar_collection_pending_vectors` both show it); the cost is one segment
write. Compact when tombstones and segments accumulate — each extra segment is
one more traversal per query, and every tombstone is a candidate that dies at
the end. Verify on a schedule if you care about silent bit rot: it is cheap
enough to run hourly.

Compaction is a rewrite: it needs disk for the new segment while the old one
still exists, and it takes a write lock on the collection (searches wait).

## Sizing

Roughly **700 bytes per vector** at 128 dimensions including the graph, ids and
allocator overhead (measured: 683 B at 50k, 700 B at 200k — see
[BENCHMARKS.md](../BENCHMARKS.md)). Budget:

```
memory ≈ vectors × (dim × 4 + ~190)      # tail, while vectors are unsealed
disk   ≈ vectors × (dim × 4 + ~190)      # sealed segments, same shape
```

`ef_search` is the latency/recall dial and can be changed per query; `m` and
`ef_construction` are chosen at creation and are a rebuild to change. IVF
parameters (`nlist`, `nprobe`, `pq_m`) trade memory for scan time — see
[ALGORITHMS.md](ALGORITHMS.md).

## Failure modes

| symptom | likely cause | what to check |
|---|---|---|
| process exits at start-up | a collection directory will not open | the log names it; `lodestar verify <name>`; move it aside |
| "checksum mismatch" from `verify` | a segment was damaged | restore that segment; the manifest names every file |
| WAL grows without bound | nothing is flushing | pending count, then `flush` |
| search got slower, recall unchanged | too many segments | `compact` |
| recall dropped | wrong `ef`, changed `m`, or drift in the embedding model | re-run `make gate`, compare distributions |
| `403` on every write | `--read-only` is set | the env var, not the request |
| `409` on create | the collection already exists | expected; it is not an error to retry |
| out of disk during flush | compaction/flush needs a free segment's worth | free space before compacting |

A crash mid-write is the normal case, not an outage: the log is replayed, a
torn tail is truncated, a half-written segment file is a `.tmp` that is removed
on the next open, and acknowledged writes are asserted to survive by
`crates/store/tests/crash.rs`, which SIGKILLs a child at random points.

## Upgrades

The segment format carries `FORMAT_VERSION` and the manifest carries
`MANIFEST_VERSION`; a reader refuses a newer version rather than guessing.
Check both before rolling a new binary over an existing directory, and keep
the old binary until `verify` has passed on the new one. The image tag and the
`lodestar --version` output should agree.
