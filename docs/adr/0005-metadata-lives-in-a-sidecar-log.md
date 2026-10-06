# 5. Metadata lives in a sidecar log

* Status: accepted
* Date: 2026-09

## Context

Search needs to filter on attributes — a category, a year, a tenant — while
the engine's unit of storage is a vector. Two designs were available: put
metadata in the segment, or keep it beside the segment.

## Decision

The engine persists ids, vectors and the graph. Metadata is the HTTP service's
concern: an append-only `metadata.log` next to the collection, `(id, metadata)`
records, replayed into a `HashMap` at start-up and appended on every upsert.
Filtering compiles the query to a bitset over node ids *before* traversal.

## Consequences

* The segment format has no schema, no nullable columns and no version to
  negotiate for user data. A change to what metadata *means* cannot invalidate
  a segment written last year.
* Replay is linear in the metadata log, and the log is rewritten compactly
  when it is mostly tombstones.
* The CLI and Python bindings do not write metadata, so a filter written
  against the API is not visible from them. That boundary is documented in
  [OPERATIONS.md](../OPERATIONS.md) rather than papered over.
* Filtering costs one bit test per candidate — no callback into the service
  from inside the traversal.

## Alternatives considered

* **Metadata columns in the segment** — right for a database, wrong here: it
  commits the storage format to a schema for a feature the storage does not
  use, and it forces the CLI and bindings to implement it too or diverge.
* **A separate key/value store** — a second durability story to test, when the
  append-and-replay one already exists for the WAL and is crash-tested.
* **Filtering after the search** — correct but useless: a filter applied to
  the top-k returns fewer than k results and hides matches the traversal never
  visited.
