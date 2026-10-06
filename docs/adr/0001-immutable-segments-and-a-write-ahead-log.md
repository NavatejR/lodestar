# 1. Immutable segments with a write-ahead log

* Status: accepted
* Date: 2026-09

## Context

Vectors arrive one at a time or in small batches and must be searchable
immediately, but the read path wants large immutable files it can memory-map.
Rewriting the whole index on every insert is quadratic; keeping everything in
memory loses it on restart. The durability requirement is specific: a write
that has been acknowledged must still be there after SIGKILL.

## Decision

Apply the log-structured merge shape to vectors.

* Writes append to a CRC'd write-ahead log first, then enter an in-memory
  HNSW *tail*. Acknowledgement happens after both.
* `flush` seals the tail into a segment written temp → fsync → rename →
  fsync-directory, then truncates the log.
* Sealed segments are never edited. Deletes record tombstones in the manifest;
  `compact` is the only operation that rewrites a segment.
* Open replays the log on top of the segments, truncating a torn tail.

## Consequences

* A crash at any point leaves one of three states — log entry present, segment
  present, or a `.tmp` file that is removed — and none of them loses an
  acknowledged write. `crates/store/tests/crash.rs` proves it by SIGKILLing a
  child at random points.
* Search pays a merge across the tail and each segment, and one traversal per
  segment. The cost is bounded by segment count, which is what compaction
  controls.
* Tombstones accumulate until compaction. This is visible in
  `CollectionStats::tombstoned_ids` rather than hidden.

## Alternatives considered

* **Rewrite the index per write** — correct, quadratic, and unflushable at
  any corpus size worth having.
* **In-memory only with a snapshot on shutdown** — fails the SIGKILL
  requirement, which is the requirement.
* **An append-only record file with no separate log** — reads would have to
  scan unsorted records to build the graph; the log exists precisely so reads
  never do.
