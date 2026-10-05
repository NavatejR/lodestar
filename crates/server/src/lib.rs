//! The Lodestar HTTP service.
//!
//! A thin, boring layer over the index and store crates: parse a request,
//! call the engine, serialise the result. Everything interesting stays in the
//! library so that the service can be replaced without touching the engine.
//!
//! The planned surface:
//!
//! * `POST /v1/index/{name}/upsert` — insert or replace vectors, with optional
//!   metadata.
//! * `POST /v1/search` — `k`-nearest-neighbour search with an optional filter
//!   expression, `ef_search` and probe count.
//! * `GET /v1/index/{name}/stats` — counters, sizes and the active SIMD kernel.
//! * `GET /healthz`, `GET /metrics` — liveness and Prometheus text exposition.
//! * `GET /docs` — OpenAPI description generated from the handlers by `utoipa`.
//!
//! This module is a placeholder: the crate currently contains no implementation.
//! It exists so that the workspace resolves and builds while the service is
//! written.

#![forbid(unsafe_op_in_unsafe_fn)]
#![warn(missing_docs)]
#![warn(clippy::all)]
