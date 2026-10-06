//! Request handlers.
//!
//! One function per endpoint. A handler does four things and nothing else:
//! validate the request, hand the engine work to the blocking pool, record the
//! metric, and return the wire type. Validation failures become [`ApiError`],
//! which knows its own status code.
//!
//! Everything that touches a collection runs on the blocking pool. The engine
//! is synchronous — it maps files, walks a graph and computes distances — so
//! calling it from an async worker would stall every other task scheduled on
//! that worker for the duration of the search. [`blocking`] moves the work to
//! the pool the runtime keeps for exactly this, and the handler awaits the
//! result.

use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::Instant;

use axum::Json;
use axum::extract::{MatchedPath, Path, Query, Request, State};
use axum::http::{StatusCode, header};
use axum::middleware::Next;
use axum::response::{Html, IntoResponse, Response};
use lodestar_ann_core::distance::active_kernel;
use lodestar_ann_index::filter;
use lodestar_ann_index::hnsw::HnswConfig;

use crate::dto::{
    ApiJson, CreateIndexRequest, DeleteRequest, DeleteResponse, HealthResponse, Hit, IndexSummary,
    MAX_DELETE_IDS, MAX_UPSERT_POINTS, MaintenanceResponse, Point, QueryResult, SearchRequest,
    SearchResponse, StatsResponse, UpsertQuery, UpsertRequest, UpsertResponse, VerifyResponse,
};
use crate::error::ApiError;
use crate::openapi;
use crate::state::{AppState, CollectionHandle, segment_summary, validate_name};

/// Runs a blocking engine operation off the async executor.
async fn blocking<T, F>(work: F) -> Result<T, ApiError>
where
    F: FnOnce() -> Result<T, ApiError> + Send + 'static,
    T: Send + 'static,
{
    tokio::task::spawn_blocking(work)
        .await
        .map_err(|join| ApiError::Internal(format!("blocking worker failed: {join}")))?
}

/// Milliseconds with sub-microsecond resolution, as `f64`.
fn millis(elapsed: std::time::Duration) -> f64 {
    elapsed.as_secs_f64() * 1_000.0
}

/// Middleware that records one observation per request.
///
/// The label is the route *template* rather than the path, so a thousand
/// collections share one time series instead of creating a thousand.
pub async fn track(State(state): State<AppState>, request: Request, next: Next) -> Response {
    let route = request
        .extensions()
        .get::<MatchedPath>()
        .map_or_else(|| "unmatched".to_string(), |path| path.as_str().to_string());
    let started = Instant::now();
    let response = next.run(request).await;
    state
        .metrics()
        .observe_request(&route, response.status().as_u16(), started.elapsed());
    response
}

/// `GET /healthz` — liveness, version, selected kernel.
pub async fn healthz(State(state): State<AppState>) -> Json<HealthResponse> {
    Json(HealthResponse {
        status: "ok",
        service: "lodestar",
        version: env!("CARGO_PKG_VERSION"),
        kernel: active_kernel().as_str(),
        uptime_seconds: state.uptime().as_secs_f64(),
        collections: state.len(),
        read_only: state.read_only(),
    })
}

/// `GET /metrics` — Prometheus text exposition.
pub async fn metrics(State(state): State<AppState>) -> Response {
    let gauges = match blocking({
        let state = state.clone();
        move || Ok(state.gauges())
    })
    .await
    {
        Ok(gauges) => gauges,
        Err(error) => {
            // A metrics scrape must not itself turn into a missing scrape; the
            // counters below are still worth reporting without the gauges.
            tracing::warn!(%error, "could not read collection gauges for /metrics");
            Vec::new()
        }
    };
    let body = state
        .metrics()
        .render(&gauges, state.uptime(), state.read_only());
    (
        [(
            header::CONTENT_TYPE,
            "text/plain; version=0.0.4; charset=utf-8",
        )],
        body,
    )
        .into_response()
}

/// `GET /openapi.json` — the machine-readable API description.
pub async fn openapi_document() -> Json<serde_json::Value> {
    Json(openapi::document())
}

/// `GET /docs` — a self-contained page over the same description.
pub async fn docs() -> Html<&'static str> {
    Html(openapi::DOCS_HTML)
}

/// Fallback for unrouted paths, so 404s have the same shape as every other
/// error instead of an empty body.
pub async fn not_found() -> ApiError {
    ApiError::NotFound("no such endpoint; see /docs for the API".to_string())
}

/// `GET /v1/index` — every collection, sorted by name.
pub async fn list_indexes(
    State(state): State<AppState>,
) -> Result<Json<Vec<IndexSummary>>, ApiError> {
    let summaries = blocking({
        let state = state.clone();
        move || Ok(state.summaries())
    })
    .await?;
    Ok(Json(summaries))
}

/// `PUT /v1/index/{name}` — create a collection.
pub async fn create_index(
    State(state): State<AppState>,
    Path(name): Path<String>,
    ApiJson(body): ApiJson<CreateIndexRequest>,
) -> Result<(StatusCode, Json<IndexSummary>), ApiError> {
    state.require_writable()?;
    validate_name(&name)?;
    if body.dim == 0 {
        return Err(ApiError::BadRequest("`dim` must be at least 1".to_string()));
    }

    let mut config = HnswConfig::default();
    if let Some(m) = body.m {
        config.m = m;
        config.m0 = m.saturating_mul(2);
    }
    if let Some(m0) = body.m0 {
        config.m0 = m0;
    }
    if let Some(ef_construction) = body.ef_construction {
        config.ef_construction = ef_construction;
    }
    if let Some(ef_search) = body.ef_search {
        config.ef_search = ef_search;
    }
    if let Some(seed) = body.seed {
        config.seed = seed;
    }
    config.validate().map_err(ApiError::from)?;

    let metric = body.metric.unwrap_or(lodestar_ann_core::Metric::L2);
    let dim = body.dim;
    let root = state.root().to_path_buf();
    let created = name.clone();
    let handle = blocking(move || {
        CollectionHandle::create(&root, &created, dim, metric, config)
            .map(Arc::new)
            .map_err(ApiError::from)
    })
    .await?;

    let summary = handle.summary();
    state.register(Arc::clone(&handle))?;
    tracing::info!(collection = %handle.name(), dim, metric = ?metric, "created collection");
    Ok((StatusCode::CREATED, Json(summary)))
}

/// `DELETE /v1/index/{name}` — close a collection and remove its files.
pub async fn drop_index(
    State(state): State<AppState>,
    Path(name): Path<String>,
) -> Result<StatusCode, ApiError> {
    state.require_writable()?;
    let handle = state.handle(&name)?;
    state.unregister(&name);
    let directory = handle.directory().to_path_buf();
    // The mapped segments hold the directory open on some platforms; dropping
    // the handle first means nothing is still reading while it disappears.
    drop(handle);
    blocking(move || {
        std::fs::remove_dir_all(&directory).map_err(ApiError::from)?;
        tracing::info!(path = %directory.display(), "dropped collection");
        Ok(())
    })
    .await?;
    Ok(StatusCode::NO_CONTENT)
}

/// `GET /v1/index/{name}/stats` — counters, sizes and the active kernel.
pub async fn stats(
    State(state): State<AppState>,
    Path(name): Path<String>,
) -> Result<Json<StatsResponse>, ApiError> {
    let handle = state.handle(&name)?;
    let response = blocking(move || {
        let stats = handle.stats();
        Ok(StatsResponse {
            name: stats.name,
            directory: handle.directory().display().to_string(),
            dim: stats.dim,
            metric: stats.metric,
            kernel: active_kernel().as_str(),
            live: stats.sealed_live + stats.tail_live,
            segments: stats.segments,
            sealed_nodes: stats.sealed_nodes,
            sealed_live: stats.sealed_live,
            tail_nodes: stats.tail_nodes,
            tail_live: stats.tail_live,
            tombstoned_ids: stats.tombstoned_ids,
            wal_bytes: stats.wal_bytes,
            mapped_bytes: stats.mapped_bytes,
            tail_bytes: stats.tail_bytes,
            tail_max_level: stats.tail_max_level,
            id_index_bytes: stats.id_index_bytes,
            metadata_entries: handle.metadata_len(),
        })
    })
    .await?;
    Ok(Json(response))
}

/// `POST /v1/index/{name}/upsert` — insert or replace vectors.
pub async fn upsert(
    State(state): State<AppState>,
    Path(name): Path<String>,
    Query(query): Query<UpsertQuery>,
    ApiJson(body): ApiJson<UpsertRequest>,
) -> Result<Json<UpsertResponse>, ApiError> {
    state.require_writable()?;
    if body.points.is_empty() {
        return Err(ApiError::BadRequest(
            "`points` must not be empty".to_string(),
        ));
    }
    if body.points.len() > MAX_UPSERT_POINTS {
        return Err(ApiError::TooLarge(format!(
            "at most {MAX_UPSERT_POINTS} points per request, got {}",
            body.points.len()
        )));
    }
    let handle = state.handle(&name)?;
    let flush = query.flush.unwrap_or(false);
    let points: Vec<Point> = body.points;

    let response = blocking(move || {
        let dim = handle.dim();
        let mut ids = Vec::with_capacity(points.len());
        let mut vectors = Vec::with_capacity(points.len() * dim);
        let mut metadata = Vec::with_capacity(points.len());
        for point in &points {
            if point.vector.len() != dim {
                return Err(ApiError::BadRequest(format!(
                    "point {} has {} dimensions, collection `{name}` has {dim}",
                    point.id,
                    point.vector.len()
                )));
            }
            ids.push(point.id);
            vectors.extend_from_slice(&point.vector);
            metadata.push(point.metadata.clone());
        }

        // Vectors first: the engine's write-ahead log makes them durable, and
        // the metadata record that follows describes a vector that exists.
        let upserted = handle
            .with_collection_mut(|collection| collection.upsert_batch(&ids, &vectors))
            .map_err(ApiError::from)?;

        let mut entries = 0usize;
        handle
            .with_metadata_mut(|store| {
                for (id, metadata) in ids.iter().zip(&metadata) {
                    if let Some(metadata) = metadata {
                        store.set(*id, metadata.clone())?;
                        entries += 1;
                    }
                }
                Ok::<(), lodestar_ann_store::Error>(())
            })
            .map_err(ApiError::from)?;
        state.metrics().observe_upsert(upserted as u64);

        let sealed = if flush {
            // Metadata is made durable before the vectors it describes are
            // sealed, so a crash can lose neither one without the other.
            handle
                .with_metadata_mut(|store| store.sync())
                .map_err(ApiError::from)?;
            handle.flush().map_err(ApiError::from)?
        } else {
            handle
                .with_metadata_mut(|store| store.compact_if_needed())
                .map_err(ApiError::from)?;
            None
        };

        let stats = handle.stats();
        Ok(UpsertResponse {
            index: name,
            upserted,
            metadata_entries: entries,
            live: stats.sealed_live + stats.tail_live,
            // Nodes in the tail, not live tail nodes: this is the amount a
            // flush would seal.
            pending: stats.tail_nodes,
            sealed: sealed.as_ref().map(segment_summary),
        })
    })
    .await?;
    Ok(Json(response))
}

/// `POST /v1/index/{name}/delete` — tombstone ids.
pub async fn delete_ids(
    State(state): State<AppState>,
    Path(name): Path<String>,
    ApiJson(body): ApiJson<DeleteRequest>,
) -> Result<Json<DeleteResponse>, ApiError> {
    state.require_writable()?;
    if body.ids.is_empty() {
        return Err(ApiError::BadRequest("`ids` must not be empty".to_string()));
    }
    if body.ids.len() > MAX_DELETE_IDS {
        return Err(ApiError::TooLarge(format!(
            "at most {MAX_DELETE_IDS} ids per request, got {}",
            body.ids.len()
        )));
    }
    let handle = state.handle(&name)?;
    let ids = body.ids;

    let response = blocking(move || {
        // The engine reports how many ids it applied, not how many vectors
        // stopped being searchable: tombstoning is idempotent and indifferent
        // to whether the id exists. The difference in live vector count is the
        // number a client actually means by "deleted".
        let before = handle.live_len();
        handle
            .with_collection_mut(|collection| collection.delete_batch(&ids))
            .map_err(ApiError::from)?;
        let entries = handle
            .with_metadata_mut(|store| store.clear(&ids))
            .map_err(ApiError::from)?;
        let after = handle.live_len();
        let deleted = before.saturating_sub(after);
        state.metrics().observe_delete(deleted as u64);
        Ok(DeleteResponse {
            index: name,
            deleted,
            metadata_entries: entries,
            live: after,
        })
    })
    .await?;
    Ok(Json(response))
}

/// `POST /v1/index/{name}/flush` — seal the in-memory tail.
pub async fn flush(
    State(state): State<AppState>,
    Path(name): Path<String>,
) -> Result<Json<MaintenanceResponse>, ApiError> {
    state.require_writable()?;
    let handle = state.handle(&name)?;
    let response = blocking(move || {
        handle
            .with_metadata_mut(|store| store.sync())
            .map_err(ApiError::from)?;
        let sealed = handle.flush().map_err(ApiError::from)?;
        Ok(maintenance(name, &handle, sealed))
    })
    .await?;
    Ok(Json(response))
}

/// `POST /v1/index/{name}/compact` — rewrite every live vector into one segment.
pub async fn compact(
    State(state): State<AppState>,
    Path(name): Path<String>,
) -> Result<Json<MaintenanceResponse>, ApiError> {
    state.require_writable()?;
    let handle = state.handle(&name)?;
    let response = blocking(move || {
        let sealed = handle.compact().map_err(ApiError::from)?;
        // Compaction is the natural moment to bound the metadata log too: the
        // vectors just moved, so rewriting the metadata costs nothing extra in
        // terms of a stall the operator would notice.
        handle
            .with_metadata_mut(|store| store.compact_if_needed())
            .map_err(ApiError::from)?;
        Ok(maintenance(name, &handle, sealed))
    })
    .await?;
    Ok(Json(response))
}

/// Builds the shared maintenance response.
fn maintenance(
    name: String,
    handle: &CollectionHandle,
    sealed: Option<lodestar_ann_store::SegmentInfo>,
) -> MaintenanceResponse {
    let stats = handle.stats();
    MaintenanceResponse {
        index: name,
        segment: sealed.as_ref().map(segment_summary),
        live: stats.sealed_live + stats.tail_live,
        pending: stats.tail_nodes,
        segments: stats.segments,
    }
}

/// `POST /v1/index/{name}/verify` — full checksum pass.
pub async fn verify(
    State(state): State<AppState>,
    Path(name): Path<String>,
) -> Result<Json<VerifyResponse>, ApiError> {
    let handle = state.handle(&name)?;
    let response = blocking(move || {
        handle.verify().map_err(ApiError::from)?;
        // The engine checks ids and vectors; the metadata log is the service's
        // own promise, so it gets checked in the same request.
        handle
            .with_metadata_mut(|store| store.verify())
            .map_err(ApiError::from)?;
        let stats = handle.stats();
        Ok(VerifyResponse {
            index: name,
            ok: true,
            nodes: stats.sealed_nodes + stats.tail_nodes,
            segments: stats.segments,
            mapped_bytes: stats.mapped_bytes,
            metadata_entries: handle.metadata_len(),
        })
    })
    .await?;
    Ok(Json(response))
}

/// `POST /v1/search` — `k`-nearest-neighbour search, optionally filtered.
pub async fn search(
    State(state): State<AppState>,
    ApiJson(body): ApiJson<SearchRequest>,
) -> Result<Json<SearchResponse>, ApiError> {
    let handle = state.handle(&body.index)?;
    let vectors = body.query_vectors()?;
    let k = body.k()?;
    let configured = handle.config();
    let ef = match body.ef {
        // Zero and absent both mean "whatever the collection was configured
        // with", floored at `k`: a candidate list shorter than the answer is
        // not a meaningful request.
        Some(0) | None => configured.ef_search.max(k),
        Some(ef) => ef.max(k),
    };
    let filter = body.filter.clone();
    let include_metadata = body.include_metadata.unwrap_or(false);

    let response = blocking(move || {
        let started = Instant::now();
        let allowed = match &filter {
            Some(text) => {
                let expr = filter::parse(text).map_err(ApiError::from)?;
                let matched = handle.matching_ids(&expr);
                tracing::debug!(
                    collection = %handle.name(),
                    matched = matched.len(),
                    "compiled filter"
                );
                Some(matched)
            }
            None => None,
        };

        let mut results = Vec::with_capacity(vectors.len());
        let mut hits_total = 0u64;
        for query in &vectors {
            let query_started = Instant::now();
            let hits = handle.search(query, k, ef, allowed.as_ref())?;
            let metadata: BTreeMap<u64, _> = if include_metadata {
                handle.metadata_for(&hits.iter().map(|hit| hit.id).collect::<Vec<_>>())
            } else {
                BTreeMap::new()
            };
            hits_total += hits.len() as u64;
            results.push(QueryResult {
                hits: hits
                    .into_iter()
                    .map(|candidate| Hit {
                        id: candidate.id,
                        distance: candidate.distance,
                        metadata: metadata.get(&candidate.id).cloned(),
                    })
                    .collect(),
                took_ms: millis(query_started.elapsed()),
            });
        }
        state
            .metrics()
            .observe_search(vectors.len() as u64, hits_total, started.elapsed());

        let stats = handle.stats();
        Ok(SearchResponse {
            index: handle.name().to_string(),
            dim: stats.dim,
            metric: stats.metric,
            k,
            ef,
            filter,
            results,
            took_ms: millis(started.elapsed()),
        })
    })
    .await?;
    Ok(Json(response))
}
