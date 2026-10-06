# 4. Three front doors over one engine

* Status: accepted
* Date: 2026-09

## Context

A vector engine needs a CLI (scripts, operators, CI), an HTTP API (services,
the demo console) and Python bindings (the people who actually embed
things). The temptation is to put shared behaviour in a fourth layer —
"service logic" — or to let each front door reimplement the parts the engine
does not cover, such as batching, validation and metadata.

## Decision

The three front doors are thin. Each one parses its input, calls
`store`/`index`/`core`, and formats the output. Nothing that a collection does
lives in a front door:

* **CLI** (`crates/cli`) — a file in, a file out, progress on stderr, machine
  output behind `--json`.
* **Server** (`crates/server`) — validate, delegate to the blocking pool,
  record a metric, serialise. Handlers never touch bytes or graphs.
* **Python** (`crates/py`) — a typed wrapper over the same three crates, with
  the GIL released around index work.

Anything shared between them belongs in a library crate or it does not exist.

## Consequences

* The engine can be tested without a transport: most tests are plain
  `cargo test` in `core`, `index` and `store`.
* The HTTP API's behaviour is reproducible with the CLI, which is how the demo
  proves it — the same directory is written by one and read by the other.
* Features that would require logic in a front door (metadata, say) are
  explicitly *not* engine features: the server keeps a sidecar log rather than
  pushing a schema into the segment format. See
  [0005-metadata-lives-in-a-sidecar-log](0005-metadata-lives-in-a-sidecar-log.md).
* There is no shared "application" crate to keep in sync, and no framework to
  learn to add a fourth door.

## Alternatives considered

* **A shared application layer** — buys a place to put validation once, costs a
  layer that all three doors must model (requests? futures? cancellation?).
  The overlap turned out to be small: three small, obvious implementations
  beat one abstraction that leaks transport concerns into the engine.
* **HTTP-only, CLI as a curl wrapper** — makes every script a network
  dependency and makes the crash story harder to test.
