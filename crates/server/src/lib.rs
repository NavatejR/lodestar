//! The Lodestar HTTP service.
//!
//! A thin, boring layer over the index and store crates: parse a request, call
//! the engine, serialise the result. Everything interesting stays in the
//! library, so the transport can be replaced without touching the engine —
//! which is exactly how the CLI and the Python bindings already use it.
//!
//! # Endpoints
//!
//! | method | path | purpose |
//! |---|---|---|
//! | `GET` | `/healthz` | liveness, version, selected SIMD kernel |
//! | `GET` | `/metrics` | Prometheus text exposition |
//! | `GET` | `/docs` | self-contained documentation page |
//! | `GET` | `/demo` | console over the API, compiled into the binary |
//! | `GET` | `/openapi.json` | OpenAPI 3.0 description |
//! | `GET` | `/v1/index` | list collections |
//! | `PUT` | `/v1/index/{name}` | create a collection |
//! | `DELETE` | `/v1/index/{name}` | close a collection and delete its files |
//! | `GET` | `/v1/index/{name}/stats` | counters, sizes, active kernel |
//! | `POST` | `/v1/index/{name}/upsert` | insert or replace vectors, with metadata |
//! | `POST` | `/v1/index/{name}/delete` | tombstone ids |
//! | `POST` | `/v1/index/{name}/flush` | seal the in-memory tail |
//! | `POST` | `/v1/index/{name}/compact` | rewrite live vectors into one segment |
//! | `POST` | `/v1/index/{name}/verify` | full checksum pass |
//! | `POST` | `/v1/search` | `k`-NN search, batched, optionally filtered |
//!
//! Every error has the same JSON body, `{"error":{"code":...,"message":...}}`,
//! so a client branches on `code` rather than parsing prose.
//!
//! # Design
//!
//! * **The engine does the work.** Handlers validate, delegate and serialise.
//!   There is no second implementation of search, filtering or durability here.
//! * **Blocking work runs on the blocking pool.** The engine is synchronous, so
//!   every call that touches a collection goes through `spawn_blocking`; an
//!   async worker is never stalled by a graph walk.
//! * **Searches run concurrently, writes exclusively** — one `RwLock` per
//!   collection, with a pool of reusable scratch buffers behind it so a
//!   steady-state query allocates nothing.
//! * **Metadata lives beside the vectors, not in them.** The engine persists
//!   ids and vectors; the service keeps an append-only metadata log next to the
//!   collection and replays it at start-up. See [`metadata`].
//! * **Bounded requests.** [`dto::MAX_UPSERT_POINTS`] points, [`dto::MAX_QUERIES`]
//!   queries, [`dto::MAX_K`] neighbours and [`MAX_BODY_BYTES`] bytes of body per
//!   request, because an unbounded request is an unbounded allocation.
//! * **Names never become paths.** A collection name is validated against a
//!   fixed alphabet before it is joined onto the data root.
//!
//! # Running
//!
//! ```text
//! cargo run --release -p lodestar-ann-server -- --root ./data --addr 127.0.0.1:8080
//! ```
//!
//! See `crates/server/src/main.rs` for the flags, and `demo/` for a Docker
//! Compose setup with a bundled corpus.

#![forbid(unsafe_op_in_unsafe_fn)]
#![warn(missing_docs)]
#![warn(clippy::all)]
// The OpenAPI document is one large `serde_json::json!` literal, which expands
// recursively; the default limit of 128 is not enough for it. Building the
// document at compile time is worth the raised limit: a typo is a build error
// rather than a runtime one.
#![recursion_limit = "512"]

pub mod dto;
pub mod error;
pub mod handlers;
pub mod metadata;
pub mod metrics;
pub mod openapi;
pub mod state;

pub use error::ApiError;
pub use state::{AppState, CollectionHandle};

use axum::Router;
use axum::extract::DefaultBodyLimit;
use axum::middleware;
use axum::routing::{get, post, put};
use tower_http::cors::CorsLayer;
use tower_http::trace::TraceLayer;

/// Largest request body the service buffers, in bytes.
///
/// Ten times the largest upsert this API accepts, so the limit that fires is
/// the one that names the problem ([`dto::MAX_UPSERT_POINTS`]) rather than a
/// transport-level rejection nobody can explain.
pub const MAX_BODY_BYTES: usize = 64 * 1024 * 1024;

/// Builds the service without CORS.
///
/// CORS is off by default: browsers on another origin cannot read responses,
/// which is the safe default for a service that holds someone's data. The demo
/// turns it on explicitly.
pub fn router(state: AppState) -> Router {
    router_with_cors(state, None)
}

/// Builds the service, optionally with a CORS layer.
pub fn router_with_cors(state: AppState, cors: Option<CorsLayer>) -> Router {
    let routes = Router::new()
        .route("/healthz", get(handlers::healthz))
        .route("/metrics", get(handlers::metrics))
        .route("/docs", get(handlers::docs))
        .route("/demo", get(handlers::demo_page))
        .route("/openapi.json", get(handlers::openapi_document))
        .route("/v1/index", get(handlers::list_indexes))
        .route(
            "/v1/index/{name}",
            put(handlers::create_index).delete(handlers::drop_index),
        )
        .route("/v1/index/{name}/stats", get(handlers::stats))
        .route("/v1/index/{name}/upsert", post(handlers::upsert))
        .route("/v1/index/{name}/delete", post(handlers::delete_ids))
        .route("/v1/index/{name}/flush", post(handlers::flush))
        .route("/v1/index/{name}/compact", post(handlers::compact))
        .route("/v1/index/{name}/verify", post(handlers::verify))
        .route("/v1/search", post(handlers::search))
        .fallback(handlers::not_found)
        .layer(DefaultBodyLimit::max(MAX_BODY_BYTES))
        // Metrics first, so a request is measured even if a later layer
        // rejects it. The label is the matched route template, which is why
        // `MatchedPath` exists by the time this middleware runs.
        .layer(middleware::from_fn_with_state(
            state.clone(),
            handlers::track,
        ))
        .layer(TraceLayer::new_for_http());

    let routes = match cors {
        Some(layer) => routes.layer(layer),
        None => routes,
    };
    routes.with_state(state)
}
