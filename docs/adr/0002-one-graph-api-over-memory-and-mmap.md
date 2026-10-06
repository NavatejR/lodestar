# 2. One graph API over memory and mmap

* Status: accepted
* Date: 2026-09

## Context

The engine has two kinds of vectors: those in the in-memory tail and those in
a sealed, memory-mapped segment. They need the same traversal, the same
filters and the same scratch buffers. The obvious implementations — a generic
over the storage, a trait object per candidate, or two copies of the search
loop — each have a cost: monomorphisation bloats the binary, a dynamic call
per node lands inside the innermost loop, and two copies drift.

## Decision

Define `GraphView` — `len`, `dim`, `metric`, `entry_point`, `max_level`,
`node_level`, `neighbors(node, level)`, `vector(node)`, `is_live`,
`external_id` — and implement it for both `Hnsw` and the mapped `Segment`.
Write `search_with_scratch` **once**, against the trait, as a static method
taking the graph as a generic parameter. Monomorphisation gives a direct call
in both cases.

## Consequences

* The in-memory and memory-mapped paths execute the same code, so a bug in
  traversal cannot be present in one and absent from the other — and a
  improvement to either is an improvement to both.
* The recall gate, which builds an in-memory graph, is testing the code path
  the disk-backed search runs.
* Adding a third storage (a paged graph, say) means implementing one trait, not
  forking the traversal.
* Trait methods are resolved statically; there is no per-node indirection to
  measure.

## Alternatives considered

* **Two search functions** — the drift risk is not theoretical; it is the
  reason engines end up with a fast path and a correct path.
* **`dyn GraphView`** — one virtual call per `neighbors` and per `vector` in
  the hot loop, for a flexibility the engine does not use: it never stores
  these as a heterogeneous collection.
* **Generics over a storage parameter instead of a view** — leaks layout into
  the algorithm and makes every signature carry it.
