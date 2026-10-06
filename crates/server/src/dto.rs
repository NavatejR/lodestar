//! Wire types for the HTTP API.
//!
//! Requests and responses are plain structs with `serde` derives, so the JSON
//! shape lives in exactly one place and a client reading the OpenAPI document
//! sees the same thing the code serialises.
//!
//! ## Limits
//!
//! Every endpoint that accepts a body is bounded, because an unbounded request
//! is an unbounded allocation. The limits are deliberately large enough for
//! real batch work and small enough that one request cannot exhaust a small
//! machine: [`MAX_UPSERT_POINTS`] points, [`MAX_QUERIES`] queries per search,
//! [`MAX_K`] neighbours per query and [`MAX_FILTER_LEN`] bytes of filter
//! expression.

use axum::extract::rejection::JsonRejection;
use axum::extract::{FromRequest, Request};
use lodestar_ann_core::Metric;
use lodestar_ann_index::Metadata;
use serde::{Deserialize, Serialize};

use crate::error::ApiError;

/// Maximum points in one upsert request.
///
/// Chosen against [`MAX_BODY_BYTES`](crate::MAX_BODY_BYTES): 4,096 points of
/// 3,072 `f32` components is 48 MiB, which fits inside the 64 MiB body limit.
/// A larger batch should be split by the client.
pub const MAX_UPSERT_POINTS: usize = 4_096;
/// Maximum ids in one delete request.
pub const MAX_DELETE_IDS: usize = 100_000;
/// Maximum queries in one search request.
pub const MAX_QUERIES: usize = 64;
/// Maximum neighbours (`k`) a single query may ask for.
pub const MAX_K: usize = 1_000;
/// Default neighbours returned when a request does not say.
pub const DEFAULT_K: usize = 10;
/// Maximum length of a filter expression, in bytes.
pub const MAX_FILTER_LEN: usize = 4_096;

/// A JSON body, with malformed input mapped to a 400 that names the problem.
///
/// Axum's own rejection prints its own plain-text body, which would break the
/// "every error is the same JSON shape" promise the rest of the service keeps.
pub struct ApiJson<T>(pub T);

impl<S, T> FromRequest<S> for ApiJson<T>
where
    JsonRejection: Into<ApiError>,
    axum::Json<T>: FromRequest<S, Rejection = JsonRejection>,
    S: Send + Sync,
{
    type Rejection = ApiError;

    async fn from_request(request: Request, state: &S) -> Result<Self, Self::Rejection> {
        let axum::Json(value) = axum::Json::<T>::from_request(request, state).await?;
        Ok(Self(value))
    }
}

/// `PUT /v1/index/{name}` — create a collection.
#[derive(Debug, Deserialize)]
pub struct CreateIndexRequest {
    /// Vector dimensionality. Every point in the collection must match it.
    pub dim: usize,
    /// Ranking metric: `l2` (default), `cosine` or `inner_product`.
    #[serde(default)]
    pub metric: Option<Metric>,
    /// Neighbours per node above level 0. Defaults to 16.
    #[serde(default)]
    pub m: Option<usize>,
    /// Neighbours per node at level 0. Defaults to `2 * m`.
    #[serde(default)]
    pub m0: Option<usize>,
    /// Build-time candidate list size. Defaults to 200.
    #[serde(default)]
    pub ef_construction: Option<usize>,
    /// Default search-time candidate list size. Defaults to 64.
    #[serde(default)]
    pub ef_search: Option<usize>,
    /// Seed for level assignment, so a rebuild reproduces the graph.
    #[serde(default)]
    pub seed: Option<u64>,
}

/// One vector, with optional metadata.
#[derive(Debug, Deserialize)]
pub struct Point {
    /// External id. Re-using an id replaces the vector.
    pub id: u64,
    /// The vector itself. Its length must equal the collection's dimensionality.
    pub vector: Vec<f32>,
    /// Arbitrary JSON metadata, used by `filter` at search time.
    #[serde(default)]
    pub metadata: Option<Metadata>,
}

/// `POST /v1/index/{name}/upsert` — insert or replace vectors.
#[derive(Debug, Deserialize)]
pub struct UpsertRequest {
    /// The points to write.
    pub points: Vec<Point>,
}

/// Query string of the upsert endpoint.
#[derive(Debug, Deserialize)]
pub struct UpsertQuery {
    /// Seal the in-memory tail into a segment before returning.
    #[serde(default)]
    pub flush: Option<bool>,
}

/// `POST /v1/index/{name}/delete` — tombstone ids.
#[derive(Debug, Deserialize)]
pub struct DeleteRequest {
    /// Ids to tombstone. Missing ids are ignored.
    pub ids: Vec<u64>,
}

/// `POST /v1/search` — `k`-nearest-neighbour search.
#[derive(Debug, Deserialize)]
pub struct SearchRequest {
    /// Which collection to search.
    pub index: String,
    /// A single query vector.
    #[serde(default)]
    pub query: Option<Vec<f32>>,
    /// A batch of query vectors. Mutually exclusive with `query`.
    #[serde(default)]
    pub queries: Option<Vec<Vec<f32>>>,
    /// How many neighbours to return per query. Defaults to 10.
    #[serde(default)]
    pub k: Option<usize>,
    /// Candidate list size. Larger is slower and more accurate. Defaults to the
    /// collection's configured `ef_search`.
    #[serde(default)]
    pub ef: Option<usize>,
    /// Filter expression, e.g. `category == "news" AND score >= 0.8`.
    #[serde(default)]
    pub filter: Option<String>,
    /// Include each hit's metadata in the response.
    #[serde(default)]
    pub include_metadata: Option<bool>,
}

impl SearchRequest {
    /// Validates and flattens the query vectors.
    ///
    /// # Errors
    ///
    /// [`ApiError::BadRequest`] if neither or both of `query` and `queries` are
    /// set, [`ApiError::TooLarge`] if the batch or the filter is too big.
    pub fn query_vectors(&self) -> Result<Vec<Vec<f32>>, ApiError> {
        let vectors = match (&self.query, &self.queries) {
            (Some(_), Some(_)) => {
                return Err(ApiError::BadRequest(
                    "give either `query` or `queries`, not both".to_string(),
                ));
            }
            (Some(vector), None) => vec![vector.clone()],
            (None, Some(queries)) if !queries.is_empty() => {
                if queries.len() > MAX_QUERIES {
                    return Err(ApiError::TooLarge(format!(
                        "at most {MAX_QUERIES} queries per request, got {}",
                        queries.len()
                    )));
                }
                queries.clone()
            }
            _ => {
                return Err(ApiError::BadRequest(
                    "one of `query` or `queries` is required".to_string(),
                ));
            }
        };
        if let Some(filter) = &self.filter {
            if filter.len() > MAX_FILTER_LEN {
                return Err(ApiError::TooLarge(format!(
                    "filter expression may be at most {MAX_FILTER_LEN} bytes"
                )));
            }
        }
        Ok(vectors)
    }

    /// The requested `k`, validated against [`MAX_K`].
    ///
    /// # Errors
    ///
    /// [`ApiError::BadRequest`] for `k == 0`, [`ApiError::TooLarge`] above
    /// [`MAX_K`].
    pub fn k(&self) -> Result<usize, ApiError> {
        let k = self.k.unwrap_or(DEFAULT_K);
        if k == 0 {
            return Err(ApiError::BadRequest("`k` must be at least 1".to_string()));
        }
        if k > MAX_K {
            return Err(ApiError::TooLarge(format!(
                "`k` may be at most {MAX_K}, got {k}"
            )));
        }
        Ok(k)
    }
}

/// One neighbour.
#[derive(Debug, Serialize)]
pub struct Hit {
    /// External id of the vector.
    pub id: u64,
    /// Distance under the collection's metric; smaller is closer.
    pub distance: f32,
    /// The vector's metadata, when the request asked for it.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<Metadata>,
}

/// The hits for one query vector.
#[derive(Debug, Serialize)]
pub struct QueryResult {
    /// Neighbours, closest first.
    pub hits: Vec<Hit>,
    /// Time spent searching this query, in milliseconds.
    pub took_ms: f64,
}

/// `POST /v1/search` response.
#[derive(Debug, Serialize)]
pub struct SearchResponse {
    /// Collection that was searched.
    pub index: String,
    /// Dimensionality of the collection.
    pub dim: usize,
    /// Ranking metric of the collection.
    pub metric: Metric,
    /// Neighbours requested per query.
    pub k: usize,
    /// Candidate list size actually used.
    pub ef: usize,
    /// Filter expression, echoed back for logging.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub filter: Option<String>,
    /// One entry per query, in request order.
    pub results: Vec<QueryResult>,
    /// Time spent on the whole request, in milliseconds.
    pub took_ms: f64,
}

/// `POST /v1/index/{name}/upsert` response.
#[derive(Debug, Serialize)]
pub struct UpsertResponse {
    /// Collection written to.
    pub index: String,
    /// Points accepted.
    pub upserted: usize,
    /// Metadata entries written.
    pub metadata_entries: usize,
    /// Live vectors after the write.
    pub live: usize,
    /// Vectors still sitting in the in-memory tail.
    pub pending: usize,
    /// Segment sealed by this request, when `?flush=true` sealed one.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sealed: Option<SegmentSummary>,
}

/// `POST /v1/index/{name}/delete` response.
#[derive(Debug, Serialize)]
pub struct DeleteResponse {
    /// Collection written to.
    pub index: String,
    /// Vectors this request removed from search.
    ///
    /// Tombstoning is idempotent and indifferent to whether an id exists, so
    /// this is measured as the drop in the live vector count rather than as
    /// the number of ids sent: deleting the same id twice removes one vector
    /// once.
    pub deleted: usize,
    /// Metadata entries dropped.
    pub metadata_entries: usize,
    /// Live vectors after the delete.
    pub live: usize,
}

/// A sealed segment, as reported by maintenance endpoints.
#[derive(Debug, Serialize)]
pub struct SegmentSummary {
    /// File name of the segment inside the collection directory.
    pub file: String,
    /// Nodes in the segment, tombstones included.
    pub nodes: usize,
    /// Live nodes in the segment.
    pub live: usize,
    /// Size of the segment file in bytes.
    pub bytes: u64,
    /// Creation time, milliseconds since the Unix epoch.
    pub created_unix_ms: u64,
}

/// `POST /v1/index/{name}/flush` and `/compact` response.
#[derive(Debug, Serialize)]
pub struct MaintenanceResponse {
    /// Collection maintained.
    pub index: String,
    /// The segment the operation sealed, if any.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub segment: Option<SegmentSummary>,
    /// Live vectors after the operation.
    pub live: usize,
    /// Vectors still pending in the tail.
    pub pending: usize,
    /// Sealed segments after the operation.
    pub segments: usize,
}

/// `GET /v1/index/{name}/stats` response.
#[derive(Debug, Serialize)]
pub struct StatsResponse {
    /// Collection name.
    pub name: String,
    /// Directory holding the collection.
    pub directory: String,
    /// Vector dimensionality.
    pub dim: usize,
    /// Ranking metric.
    pub metric: Metric,
    /// Which SIMD kernel this process selected.
    pub kernel: &'static str,
    /// Searchable vectors: live tail nodes plus non-tombstoned sealed nodes.
    pub live: usize,
    /// Sealed segments.
    pub segments: usize,
    /// Nodes across sealed segments, tombstones included.
    pub sealed_nodes: usize,
    /// Live nodes across sealed segments.
    pub sealed_live: usize,
    /// Nodes in the in-memory tail.
    pub tail_nodes: usize,
    /// Live nodes in the in-memory tail.
    pub tail_live: usize,
    /// Ids tombstoned in at least one sealed segment.
    pub tombstoned_ids: usize,
    /// Bytes of write-ahead log not yet sealed.
    pub wal_bytes: u64,
    /// Bytes of sealed segments, as mapped.
    pub mapped_bytes: usize,
    /// Heap bytes held by the tail.
    pub tail_bytes: usize,
    /// Highest populated level in the tail.
    pub tail_max_level: usize,
    /// Heap bytes held by the id index over sealed segments.
    pub id_index_bytes: usize,
    /// Metadata entries held by the service for this collection.
    pub metadata_entries: usize,
}

/// One row of `GET /v1/index`.
#[derive(Debug, Serialize)]
pub struct IndexSummary {
    /// Collection name.
    pub name: String,
    /// Vector dimensionality.
    pub dim: usize,
    /// Ranking metric.
    pub metric: Metric,
    /// Live vectors, sealed and pending.
    pub live: usize,
    /// Vectors still pending in the tail.
    pub pending: usize,
    /// Sealed segments.
    pub segments: usize,
    /// Bytes of sealed segments, as mapped.
    pub mapped_bytes: usize,
    /// Metadata entries held for this collection.
    pub metadata_entries: usize,
}

/// `GET /healthz` response.
#[derive(Debug, Serialize)]
pub struct HealthResponse {
    /// Always `ok`; the endpoint exists so a load balancer can stop sending
    /// traffic, not to probe the engine.
    pub status: &'static str,
    /// Service name.
    pub service: &'static str,
    /// Crate version.
    pub version: &'static str,
    /// SIMD kernel selected by the running process.
    pub kernel: &'static str,
    /// Seconds since the process started.
    pub uptime_seconds: f64,
    /// Collections currently open.
    pub collections: usize,
    /// Whether mutations are refused.
    pub read_only: bool,
}

/// `POST /v1/index/{name}/verify` response.
#[derive(Debug, Serialize)]
pub struct VerifyResponse {
    /// Collection verified.
    pub index: String,
    /// Always `true`; a failure is an error response instead.
    pub ok: bool,
    /// Nodes checked, across segments and the tail.
    pub nodes: usize,
    /// Segments checked.
    pub segments: usize,
    /// Bytes of sealed segments read back.
    pub mapped_bytes: usize,
    /// Metadata entries checked.
    pub metadata_entries: usize,
}
