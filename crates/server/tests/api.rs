//! End-to-end tests for the HTTP API.
//!
//! Every test drives the real router through `tower::ServiceExt::oneshot`, so
//! what is exercised is the same stack a client hits: routing, extraction,
//! validation, the blocking pool, the engine and error mapping. Only the
//! socket is missing, which is the one part the engine does not own.

use std::collections::BTreeSet;

use axum::Router;
use axum::body::{Body, to_bytes};
use axum::http::{Request, StatusCode, header};
use lodestar_ann_core::Rng;
use lodestar_ann_server::{AppState, router};
use serde_json::{Value, json};
use tempfile::TempDir;
use tower::ServiceExt;

/// The dimension every test collection uses.
const DIM: usize = 8;

/// A server backed by a temporary data root.
struct Harness {
    app: Router,
    state: AppState,
    // Held so the directory lives as long as the harness.
    _root: TempDir,
}

impl Harness {
    fn new() -> Self {
        Self::with_read_only(false)
    }

    fn with_read_only(read_only: bool) -> Self {
        let root = TempDir::new().expect("temporary directory");
        let state = AppState::open(root.path(), read_only).expect("open state");
        let app = router(state.clone());
        Self {
            app,
            state,
            _root: root,
        }
    }

    /// Re-opens the state over the same directory, as a restart would.
    fn restart(&mut self) {
        let state = AppState::open(self._root.path(), false).expect("reopen state");
        self.app = router(state.clone());
        self.state = state;
    }

    async fn send(&self, request: Request<Body>) -> (StatusCode, Value, String) {
        let response = self.app.clone().oneshot(request).await.expect("response");
        let status = response.status();
        let bytes = to_bytes(response.into_body(), 64 * 1024 * 1024)
            .await
            .expect("body");
        let text = String::from_utf8_lossy(&bytes).into_owned();
        let json = serde_json::from_str(&text).unwrap_or(Value::Null);
        (status, json, text)
    }

    async fn json_request(
        &self,
        method: &str,
        uri: &str,
        body: &str,
    ) -> (StatusCode, Value, String) {
        let request = Request::builder()
            .method(method)
            .uri(uri)
            .header(header::CONTENT_TYPE, "application/json")
            .body(Body::from(body.to_string()))
            .expect("request");
        self.send(request).await
    }

    /// Sends a JSON body and returns the decoded response.
    async fn call(&self, method: &str, uri: &str, body: Value) -> (StatusCode, Value) {
        let (status, json, text) = self.json_request(method, uri, &body.to_string()).await;
        assert!(
            json != Value::Null || text.is_empty(),
            "{method} {uri} returned a body that is not JSON: {text}"
        );
        (status, json)
    }

    async fn get(&self, uri: &str) -> (StatusCode, Value) {
        self.call("GET", uri, Value::Null).await
    }

    async fn text(&self, uri: &str) -> (StatusCode, String, Option<String>) {
        let request = Request::builder()
            .method("GET")
            .uri(uri)
            .body(Body::empty())
            .expect("request");
        let response = self.app.clone().oneshot(request).await.expect("response");
        let status = response.status();
        let content_type = response
            .headers()
            .get(header::CONTENT_TYPE)
            .and_then(|value| value.to_str().ok())
            .map(str::to_string);
        let bytes = to_bytes(response.into_body(), 64 * 1024 * 1024)
            .await
            .expect("body");
        (
            status,
            String::from_utf8_lossy(&bytes).into_owned(),
            content_type,
        )
    }

    /// Creates a collection and fails the test if that does not work.
    async fn create(&self, name: &str) {
        let (status, body) = self
            .call(
                "PUT",
                &format!("/v1/index/{name}"),
                json!({ "dim": DIM, "metric": "l2" }),
            )
            .await;
        assert_eq!(status, StatusCode::CREATED, "{body}");
    }

    /// Creates `docs` and fills it with `points` deterministic vectors.
    async fn seed(&self, points: usize) -> Vec<Vec<f32>> {
        self.create("docs").await;
        let vectors = deterministic_vectors(points);
        let payload: Vec<Value> = vectors
            .iter()
            .enumerate()
            .map(|(index, vector)| json!({ "id": index as u64, "vector": vector }))
            .collect();
        let (status, body) = self
            .call(
                "POST",
                "/v1/index/docs/upsert?flush=true",
                json!({ "points": payload }),
            )
            .await;
        assert_eq!(status, StatusCode::OK, "{body}");
        assert_eq!(body["upserted"].as_u64().unwrap(), points as u64);
        vectors
    }
}

/// `points` vectors in `DIM` dimensions, reproducible across runs.
fn deterministic_vectors(points: usize) -> Vec<Vec<f32>> {
    let mut rng = Rng::new(0xC0FFEE);
    (0..points)
        .map(|_| {
            (0..DIM)
                .map(|_| (rng.next_f64() * 2.0 - 1.0) as f32)
                .collect()
        })
        .collect()
}

/// The ids of every hit in a search response, in order.
fn hit_ids(body: &Value) -> Vec<u64> {
    body["results"][0]["hits"]
        .as_array()
        .expect("hits array")
        .iter()
        .map(|hit| hit["id"].as_u64().expect("id"))
        .collect()
}

fn error_code(body: &Value) -> &str {
    body["error"]["code"].as_str().unwrap_or("<missing>")
}

// ---------------------------------------------------------------------------
// service endpoints

#[tokio::test]
async fn healthz_reports_version_kernel_and_mode() {
    let harness = Harness::new();
    let (status, body) = harness.get("/healthz").await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["status"], "ok");
    assert_eq!(body["service"], "lodestar");
    assert_eq!(body["version"], env!("CARGO_PKG_VERSION"));
    assert!(body["uptime_seconds"].as_f64().is_some());
    assert_eq!(body["collections"], 0);
    assert_eq!(body["read_only"], false);
    // The kernel must be one the core crate can actually report.
    assert!(
        ["scalar", "neon", "avx2"].contains(&body["kernel"].as_str().unwrap()),
        "unexpected kernel {}",
        body["kernel"]
    );
}

#[tokio::test]
async fn metrics_are_prometheus_text_and_reflect_traffic() {
    let harness = Harness::new();
    harness.seed(16).await;
    let (status, _) = harness
        .call(
            "POST",
            "/v1/search",
            json!({ "index": "docs", "query": deterministic_vectors(1)[0], "k": 3 }),
        )
        .await;
    assert_eq!(status, StatusCode::OK);

    let (status, body, content_type) = harness.text("/metrics").await;
    assert_eq!(status, StatusCode::OK);
    let content_type = content_type.unwrap_or_default();
    assert!(
        content_type.starts_with("text/plain"),
        "content type was {content_type}"
    );
    // Route templates, not paths: a thousand collections, one series.
    assert!(
        body.contains("lodestar_http_requests_total{route=\"/v1/search\",status=\"2xx\"}"),
        "{body}"
    );
    assert!(body.contains("lodestar_search_queries_total 1\n"), "{body}");
    assert!(
        body.contains("lodestar_vectors_upserted_total 16\n"),
        "{body}"
    );
    assert!(
        body.contains("lodestar_collection_live_vectors{collection=\"docs\"} 16\n"),
        "{body}"
    );
    assert!(body.contains("lodestar_collections 1\n"), "{body}");
    assert!(body.contains("lodestar_read_only 0\n"), "{body}");
    assert!(
        body.contains("# TYPE lodestar_uptime_seconds gauge\n"),
        "{body}"
    );
}

#[tokio::test]
async fn the_docs_page_and_the_openapi_document_are_served() {
    let harness = Harness::new();
    let (status, html, content_type) = harness.text("/docs").await;
    assert_eq!(status, StatusCode::OK);
    assert!(content_type.unwrap().starts_with("text/html"));
    assert!(html.contains("Lodestar"));
    assert!(html.contains("/v1/search"));

    let (status, document) = harness.get("/openapi.json").await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(document["openapi"], "3.0.3");
    assert!(document["paths"]["/v1/search"]["post"].is_object());
}

#[tokio::test]
async fn an_unknown_endpoint_is_a_json_404() {
    let harness = Harness::new();
    let (status, body) = harness.get("/nope").await;
    assert_eq!(status, StatusCode::NOT_FOUND);
    assert_eq!(error_code(&body), "not_found");
}

// ---------------------------------------------------------------------------
// collections

#[tokio::test]
async fn creating_a_collection_returns_it_in_the_listing() {
    let harness = Harness::new();
    let (status, body) = harness
        .call(
            "PUT",
            "/v1/index/docs",
            json!({ "dim": 384, "metric": "cosine", "m": 8, "ef_construction": 100, "seed": 3 }),
        )
        .await;
    assert_eq!(status, StatusCode::CREATED, "{body}");
    assert_eq!(body["name"], "docs");
    assert_eq!(body["dim"], 384);
    assert_eq!(body["metric"], "cosine");
    assert_eq!(body["live"], 0);

    let (status, listing) = harness.get("/v1/index").await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(listing.as_array().unwrap().len(), 1);
    assert_eq!(listing[0]["name"], "docs");
    assert_eq!(listing[0]["metadata_entries"], 0);
}

#[tokio::test]
async fn creating_a_collection_twice_conflicts() {
    let harness = Harness::new();
    harness.create("docs").await;
    let (status, body) = harness
        .call("PUT", "/v1/index/docs", json!({ "dim": DIM }))
        .await;
    assert_eq!(status, StatusCode::CONFLICT);
    assert_eq!(error_code(&body), "conflict");
    assert!(body["error"]["message"].as_str().unwrap().contains("docs"));
}

#[tokio::test]
async fn impossible_graph_parameters_are_rejected() {
    let harness = Harness::new();
    for body in [
        json!({ "dim": 0 }),
        json!({ "dim": 8, "m": 1 }),
        json!({ "dim": 8, "m": 16, "m0": 4 }),
        json!({ "dim": 8, "ef_construction": 0 }),
        json!({ "dim": 8, "ef_search": 0 }),
    ] {
        let (status, response) = harness.call("PUT", "/v1/index/bad", body.clone()).await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{body} -> {response}");
        assert_eq!(error_code(&response), "bad_request");
        assert!(
            harness.state.handle("bad").is_err(),
            "{body} should not have created a collection"
        );
    }
}

#[tokio::test]
async fn names_that_could_escape_the_data_root_are_rejected() {
    let harness = Harness::new();
    for name in ["..", "%2e%2e", ".", "a%2Fb", "a%20b", "a%2Ab"] {
        let (status, body) = harness
            .call("PUT", &format!("/v1/index/{name}"), json!({ "dim": DIM }))
            .await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{name} -> {body}");
        assert_eq!(error_code(&body), "bad_request");
    }
    // And the same guard applies to reads, not just writes.
    let (status, body) = harness.get("/v1/index/..%2F..%2Fetc/stats").await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
}

#[tokio::test]
async fn an_unknown_collection_is_a_404_everywhere() {
    let harness = Harness::new();
    let (status, body) = harness.get("/v1/index/missing/stats").await;
    assert_eq!(status, StatusCode::NOT_FOUND);
    assert_eq!(error_code(&body), "not_found");

    let (status, body) = harness
        .call(
            "POST",
            "/v1/search",
            json!({ "index": "missing", "query": vec![0.0f32; DIM] }),
        )
        .await;
    assert_eq!(status, StatusCode::NOT_FOUND);
    assert_eq!(error_code(&body), "not_found");

    let (status, body) = harness
        .call(
            "POST",
            "/v1/index/missing/upsert",
            json!({ "points": [{ "id": 1, "vector": vec![0.0; DIM] }] }),
        )
        .await;
    assert_eq!(status, StatusCode::NOT_FOUND, "{body}");
}

#[tokio::test]
async fn dropping_a_collection_removes_its_directory() {
    let harness = Harness::new();
    harness.seed(8).await;
    let directory = harness
        .state
        .handle("docs")
        .unwrap()
        .directory()
        .to_path_buf();
    assert!(directory.is_dir());

    let request = Request::builder()
        .method("DELETE")
        .uri("/v1/index/docs")
        .body(Body::empty())
        .unwrap();
    let (status, _, _) = harness.send(request).await;
    assert_eq!(status, StatusCode::NO_CONTENT);
    assert!(!directory.exists(), "the directory should be gone");
    assert!(harness.state.handle("docs").is_err());

    let (status, _) = harness.get("/v1/index").await;
    assert_eq!(status, StatusCode::OK);
}

// ---------------------------------------------------------------------------
// writing

#[tokio::test]
async fn upsert_then_search_finds_the_vector_itself() {
    let harness = Harness::new();
    let vectors = harness.seed(64).await;

    let (status, body) = harness
        .call(
            "POST",
            "/v1/search",
            json!({ "index": "docs", "query": vectors[37], "k": 5, "ef": 128 }),
        )
        .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let ids = hit_ids(&body);
    assert_eq!(ids.len(), 5);
    assert_eq!(
        ids[0], 37,
        "the query's own vector must be its nearest neighbour"
    );
    let distance = body["results"][0]["hits"][0]["distance"].as_f64().unwrap();
    assert!(distance < 1e-5, "self distance was {distance}");
    assert!(body["took_ms"].as_f64().unwrap() >= 0.0);
    assert_eq!(body["dim"], DIM);
}

#[tokio::test]
async fn reusing_an_id_replaces_the_vector() {
    let harness = Harness::new();
    harness.seed(8).await;
    let mut moved = deterministic_vectors(1)[0].clone();
    for value in &mut moved {
        *value += 100.0;
    }
    let (status, body) = harness
        .call(
            "POST",
            "/v1/index/docs/upsert?flush=true",
            json!({ "points": [{ "id": 3, "vector": moved }] }),
        )
        .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["upserted"], 1);
    // Eight originals, one replaced: still eight live vectors.
    assert_eq!(body["live"], 8);

    let (_, body) = harness
        .call(
            "POST",
            "/v1/search",
            json!({ "index": "docs", "query": moved, "k": 1 }),
        )
        .await;
    assert_eq!(hit_ids(&body), vec![3]);
    let distance = body["results"][0]["hits"][0]["distance"].as_f64().unwrap();
    assert!(distance < 1e-4, "replacement distance was {distance}");
}

#[tokio::test]
async fn a_wrong_sized_vector_is_rejected_with_the_collection_size() {
    let harness = Harness::new();
    harness.create("docs").await;
    let (status, body) = harness
        .call(
            "POST",
            "/v1/index/docs/upsert",
            json!({ "points": [{ "id": 1, "vector": [1.0, 2.0, 3.0] }] }),
        )
        .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(error_code(&body), "bad_request");
    let message = body["error"]["message"].as_str().unwrap();
    assert!(message.contains("3 dimensions"), "{message}");
    assert!(message.contains(&DIM.to_string()), "{message}");

    // A query of the wrong size fails the same way.
    let (status, body) = harness
        .call(
            "POST",
            "/v1/search",
            json!({ "index": "docs", "query": [1.0] }),
        )
        .await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
}

#[tokio::test]
async fn empty_bodies_are_rejected() {
    let harness = Harness::new();
    harness.create("docs").await;
    let (status, body) = harness
        .call("POST", "/v1/index/docs/upsert", json!({ "points": [] }))
        .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(error_code(&body), "bad_request");

    let (status, body) = harness
        .call("POST", "/v1/index/docs/delete", json!({ "ids": [] }))
        .await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
}

#[tokio::test]
async fn a_malformed_body_is_a_400_with_a_json_error() {
    let harness = Harness::new();
    harness.create("docs").await;
    let (status, body, _) = harness
        .json_request("POST", "/v1/index/docs/upsert", "{ not json")
        .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(error_code(&body), "bad_request");
    assert!(
        body["error"]["message"]
            .as_str()
            .unwrap()
            .contains("invalid JSON body"),
        "{body}"
    );

    // A body that is JSON but not the right shape fails the same way.
    let (status, body, _) = harness
        .json_request("POST", "/v1/index/docs/upsert", "{\"points\": 7}")
        .await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
}

#[tokio::test]
async fn deletes_hide_vectors_from_search() {
    let harness = Harness::new();
    let vectors = harness.seed(32).await;
    let (status, body) = harness
        .call("POST", "/v1/index/docs/delete", json!({ "ids": [5, 6, 7] }))
        .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["deleted"], 3);
    assert_eq!(body["live"], 29);

    for id in 5..8u64 {
        let (_, body) = harness
            .call(
                "POST",
                "/v1/search",
                json!({ "index": "docs", "query": vectors[id as usize], "k": 3 }),
            )
            .await;
        assert!(
            !hit_ids(&body).contains(&id),
            "deleted id {id} was returned: {body}"
        );
    }

    // Deleting again is a no-op, not an error: nothing disappears twice.
    let (status, body) = harness
        .call("POST", "/v1/index/docs/delete", json!({ "ids": [5, 6, 7] }))
        .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["deleted"], 0);
    assert_eq!(body["live"], 29);

    // Deleting an id that was never written is also a no-op.
    let (status, body) = harness
        .call("POST", "/v1/index/docs/delete", json!({ "ids": [9_999] }))
        .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["deleted"], 0);
    assert_eq!(body["live"], 29);
}

#[tokio::test]
async fn queries_are_bounded() {
    let harness = Harness::new();
    harness.create("docs").await;
    let vector = vec![0.0f32; DIM];

    let (status, body) = harness
        .call(
            "POST",
            "/v1/search",
            json!({ "index": "docs", "query": vector, "k": 0 }),
        )
        .await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");

    let (status, body) = harness
        .call(
            "POST",
            "/v1/search",
            json!({ "index": "docs", "query": vector, "k": 100_000 }),
        )
        .await;
    assert_eq!(status, StatusCode::PAYLOAD_TOO_LARGE, "{body}");
    assert_eq!(error_code(&body), "payload_too_large");

    let (status, body) = harness
        .call(
            "POST",
            "/v1/search",
            json!({ "index": "docs", "queries": vec![vector.clone(); 65] }),
        )
        .await;
    assert_eq!(status, StatusCode::PAYLOAD_TOO_LARGE, "{body}");

    let (status, body) = harness
        .call(
            "POST",
            "/v1/search",
            json!({ "index": "docs", "query": vector, "queries": [vector] }),
        )
        .await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");

    let (status, body) = harness
        .call("POST", "/v1/search", json!({ "index": "docs" }))
        .await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");

    let (status, body) = harness
        .call(
            "POST",
            "/v1/search",
            json!({
                "index": "docs",
                "query": vec![0.0f32; DIM],
                "filter": "x".repeat(5_000)
            }),
        )
        .await;
    assert_eq!(status, StatusCode::PAYLOAD_TOO_LARGE, "{body}");
}

#[tokio::test]
async fn a_batch_search_returns_one_result_per_query() {
    let harness = Harness::new();
    let vectors = harness.seed(16).await;
    let (status, body) = harness
        .call(
            "POST",
            "/v1/search",
            json!({ "index": "docs", "queries": [vectors[1], vectors[2]], "k": 2 }),
        )
        .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let results = body["results"].as_array().unwrap();
    assert_eq!(results.len(), 2);
    assert_eq!(results[0]["hits"][0]["id"], 1);
    assert_eq!(results[1]["hits"][0]["id"], 2);
}

// ---------------------------------------------------------------------------
// metadata filtering

#[tokio::test]
async fn a_filter_selects_only_matching_vectors() {
    let harness = Harness::new();
    harness.create("docs").await;
    let vectors = deterministic_vectors(200);
    let points: Vec<Value> = vectors
        .iter()
        .enumerate()
        .map(|(index, vector)| {
            let group = if index % 2 == 0 { "even" } else { "odd" };
            let score = (index as f64) / 200.0;
            json!({
                "id": index as u64,
                "vector": vector,
                "metadata": { "group": group, "score": score, "tags": ["t", group] }
            })
        })
        .collect();
    let (status, body) = harness
        .call(
            "POST",
            "/v1/index/docs/upsert?flush=true",
            json!({ "points": points }),
        )
        .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["metadata_entries"], 200);

    let (status, body) = harness
        .call(
            "POST",
            "/v1/search",
            json!({
                "index": "docs",
                "query": vectors[100],
                "k": 10,
                "filter": "group == even AND score >= 0.2"
            }),
        )
        .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["filter"], "group == even AND score >= 0.2");
    let ids = hit_ids(&body);
    assert_eq!(ids.len(), 10, "a selective filter still fills k");
    assert!(ids.iter().all(|id| id % 2 == 0), "{ids:?}");
    assert!(ids.iter().all(|id| *id >= 40), "{ids:?}");
    // The query is id 100, which matches the filter, so it must come first.
    assert_eq!(ids[0], 100);
}

#[tokio::test]
async fn include_metadata_returns_the_stored_metadata() {
    let harness = Harness::new();
    harness.create("docs").await;
    let vectors = deterministic_vectors(4);
    let points: Vec<Value> = vectors
        .iter()
        .enumerate()
        .map(|(index, vector)| {
            json!({ "id": index as u64, "vector": vector, "metadata": { "label": format!("v{index}") } })
        })
        .collect();
    harness
        .call("POST", "/v1/index/docs/upsert", json!({ "points": points }))
        .await;

    let (_, body) = harness
        .call(
            "POST",
            "/v1/search",
            json!({ "index": "docs", "query": vectors[2], "k": 1, "include_metadata": true }),
        )
        .await;
    assert_eq!(body["results"][0]["hits"][0]["metadata"]["label"], "v2");

    // Without the flag, the field is absent rather than an empty object.
    let (_, body) = harness
        .call(
            "POST",
            "/v1/search",
            json!({ "index": "docs", "query": vectors[2], "k": 1 }),
        )
        .await;
    assert!(
        body["results"][0]["hits"][0].get("metadata").is_none(),
        "{body}"
    );
}

#[tokio::test]
async fn filtering_on_a_value_that_does_not_exist_returns_nothing() {
    let harness = Harness::new();
    harness.create("docs").await;
    let vectors = deterministic_vectors(64);
    let points: Vec<Value> = vectors
        .iter()
        .enumerate()
        .map(|(index, vector)| {
            json!({ "id": index as u64, "vector": vector, "metadata": { "group": "known" } })
        })
        .collect();
    harness
        .call(
            "POST",
            "/v1/index/docs/upsert?flush=true",
            json!({ "points": points }),
        )
        .await;

    // A filter that matches nothing is an empty result, not an error.
    let (status, body) = harness
        .call(
            "POST",
            "/v1/search",
            json!({ "index": "docs", "query": vectors[0], "k": 5, "filter": "group == absent" }),
        )
        .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert!(hit_ids(&body).is_empty(), "{body}");

    // Vectors without metadata never match a comparison, including `!=`.
    let (status, body) = harness
        .call(
            "POST",
            "/v1/search",
            json!({ "index": "docs", "query": vectors[0], "k": 5, "filter": "group != absent" }),
        )
        .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(hit_ids(&body).len(), 5, "{body}");
}

#[tokio::test]
async fn a_broken_filter_expression_is_a_400() {
    let harness = Harness::new();
    harness.seed(8).await;
    for filter in ["group ==", "group = news", "(group == a", "group ~ a"] {
        let (status, body) = harness
            .call(
                "POST",
                "/v1/search",
                json!({ "index": "docs", "query": deterministic_vectors(1)[0], "filter": filter }),
            )
            .await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{filter} -> {body}");
        assert_eq!(error_code(&body), "bad_request");
    }
}

#[tokio::test]
async fn deleting_a_vector_forgets_its_metadata() {
    let harness = Harness::new();
    harness.create("docs").await;
    let vectors = deterministic_vectors(4);
    let points: Vec<Value> = vectors
        .iter()
        .enumerate()
        .map(|(index, vector)| {
            json!({ "id": index as u64, "vector": vector, "metadata": { "group": "a" } })
        })
        .collect();
    harness
        .call("POST", "/v1/index/docs/upsert", json!({ "points": points }))
        .await;

    let (_, body) = harness
        .call("POST", "/v1/index/docs/delete", json!({ "ids": [0, 1] }))
        .await;
    assert_eq!(body["metadata_entries"], 2);

    let (_, stats) = harness.get("/v1/index/docs/stats").await;
    assert_eq!(stats["metadata_entries"], 2);
}

// ---------------------------------------------------------------------------
// maintenance and durability

#[tokio::test]
async fn flush_seals_a_segment_and_stats_describe_it() {
    let harness = Harness::new();
    harness.create("docs").await;
    let vectors = deterministic_vectors(32);
    let points: Vec<Value> = vectors
        .iter()
        .enumerate()
        .map(|(index, vector)| json!({ "id": index as u64, "vector": vector }))
        .collect();
    harness
        .call("POST", "/v1/index/docs/upsert", json!({ "points": points }))
        .await;

    let (_, before) = harness.get("/v1/index/docs/stats").await;
    assert_eq!(before["segments"], 0);
    assert_eq!(before["tail_nodes"], 32);

    let (status, body) = harness
        .call("POST", "/v1/index/docs/flush", Value::Null)
        .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["segments"], 1);
    assert_eq!(body["segment"]["nodes"], 32);
    assert_eq!(body["pending"], 0);
    assert!(body["segment"]["bytes"].as_u64().unwrap() > 0);

    let (_, after) = harness.get("/v1/index/docs/stats").await;
    assert_eq!(after["segments"], 1);
    assert_eq!(after["sealed_nodes"], 32);
    assert_eq!(after["sealed_live"], 32);
    assert!(after["mapped_bytes"].as_u64().unwrap() > 0);
    assert!(["scalar", "neon", "avx2"].contains(&after["kernel"].as_str().unwrap()));

    // Flushing again has nothing to seal.
    let (status, body) = harness
        .call("POST", "/v1/index/docs/flush", Value::Null)
        .await;
    assert_eq!(status, StatusCode::OK);
    assert!(body.get("segment").is_none(), "{body}");
}

#[tokio::test]
async fn upsert_with_the_flush_query_parameter_seals_immediately() {
    let harness = Harness::new();
    harness.create("docs").await;
    let vectors = deterministic_vectors(16);
    let points: Vec<Value> = vectors
        .iter()
        .enumerate()
        .map(|(index, vector)| json!({ "id": index as u64, "vector": vector }))
        .collect();
    let (status, body) = harness
        .call(
            "POST",
            "/v1/index/docs/upsert?flush=true",
            json!({ "points": points }),
        )
        .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["sealed"]["nodes"], 16);
    assert_eq!(body["pending"], 0);

    let (_, stats) = harness.get("/v1/index/docs/stats").await;
    assert_eq!(stats["segments"], 1);
    assert_eq!(stats["wal_bytes"], 0);
}

#[tokio::test]
async fn verify_checks_the_segments_and_the_metadata_log() {
    let harness = Harness::new();
    harness.create("docs").await;
    let vectors = deterministic_vectors(16);
    let points: Vec<Value> = vectors
        .iter()
        .enumerate()
        .map(|(index, vector)| {
            json!({ "id": index as u64, "vector": vector, "metadata": { "group": "a" } })
        })
        .collect();
    harness
        .call(
            "POST",
            "/v1/index/docs/upsert?flush=true",
            json!({ "points": points }),
        )
        .await;

    let (status, body) = harness
        .call("POST", "/v1/index/docs/verify", Value::Null)
        .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["ok"], true);
    assert_eq!(body["nodes"], 16);
    assert_eq!(body["segments"], 1);
    assert_eq!(body["metadata_entries"], 16);
}

#[tokio::test]
async fn compact_collapses_segments_and_keeps_live_vectors() {
    let harness = Harness::new();
    let vectors = harness.seed(24).await;
    harness
        .call(
            "POST",
            "/v1/index/docs/delete",
            json!({ "ids": [0, 1, 2, 3] }),
        )
        .await;
    let mut later = vectors[4].clone();
    later[0] += 0.001;
    harness
        .call(
            "POST",
            "/v1/index/docs/upsert?flush=true",
            json!({ "points": [{ "id": 100, "vector": later }] }),
        )
        .await;
    let (_, before) = harness.get("/v1/index/docs/stats").await;
    assert_eq!(before["segments"], 2);

    let (status, body) = harness
        .call("POST", "/v1/index/docs/compact", Value::Null)
        .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["segments"], 1);
    assert_eq!(body["live"], 21, "24 seeded, 4 deleted, 1 added");

    let (_, after) = harness.get("/v1/index/docs/stats").await;
    assert_eq!(after["sealed_nodes"], 21);
    assert_eq!(after["tombstoned_ids"], 0);

    // Every live vector is still findable after the rewrite.
    for id in [4usize, 10, 23, 100] {
        let query = if id == 100 {
            later.clone()
        } else {
            vectors[id].clone()
        };
        let (_, body) = harness
            .call(
                "POST",
                "/v1/search",
                json!({ "index": "docs", "query": query, "k": 1 }),
            )
            .await;
        assert_eq!(hit_ids(&body), vec![id as u64], "id {id} was lost");
    }
}

#[tokio::test]
async fn collections_survive_a_restart() {
    let mut harness = Harness::new();
    harness.create("docs").await;
    let vectors = deterministic_vectors(48);
    let points: Vec<Value> = vectors
        .iter()
        .enumerate()
        .map(|(index, vector)| {
            json!({ "id": index as u64, "vector": vector, "metadata": { "group": "a" } })
        })
        .collect();
    harness
        .call(
            "POST",
            "/v1/index/docs/upsert",
            json!({ "points": points[..32].to_vec() }),
        )
        .await;
    harness
        .call("POST", "/v1/index/docs/flush", Value::Null)
        .await;
    // The second batch only reaches the write-ahead log before the restart.
    harness
        .call(
            "POST",
            "/v1/index/docs/upsert",
            json!({ "points": points[32..].to_vec() }),
        )
        .await;

    harness.restart();

    let (status, listing) = harness.get("/v1/index").await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(listing[0]["live"], 48, "the log must be replayed");
    assert_eq!(listing[0]["metadata_entries"], 48);

    let (_, body) = harness
        .call(
            "POST",
            "/v1/search",
            json!({ "index": "docs", "query": vectors[40], "k": 3, "include_metadata": true }),
        )
        .await;
    assert_eq!(hit_ids(&body)[0], 40);
    assert_eq!(body["results"][0]["hits"][0]["metadata"]["group"], "a");
}

#[tokio::test]
async fn a_read_only_server_refuses_writes_and_allows_reads() {
    let harness = Harness::new();
    harness.seed(8).await;
    // A second state over the same directory, as a read-only replica would be.
    let state = AppState::open(harness._root.path(), true).unwrap();
    assert!(state.read_only());
    let app = router(state);
    let send = |method: &'static str, uri: String, body: String| {
        let app = app.clone();
        async move {
            let request = Request::builder()
                .method(method)
                .uri(uri)
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(body))
                .unwrap();
            let response = app.oneshot(request).await.unwrap();
            let status = response.status();
            let bytes = to_bytes(response.into_body(), 1 << 20).await.unwrap();
            (
                status,
                serde_json::from_slice::<Value>(&bytes).unwrap_or(Value::Null),
            )
        }
    };

    let upsert = json!({ "points": [{ "id": 99, "vector": vec![0.0f32; DIM] }] }).to_string();
    let (status, body) = send("POST", "/v1/index/docs/upsert".to_string(), upsert).await;
    assert_eq!(status, StatusCode::FORBIDDEN);
    assert_eq!(error_code(&body), "forbidden");

    let (status, _) = send("DELETE", "/v1/index/docs".to_string(), String::new()).await;
    assert_eq!(status, StatusCode::FORBIDDEN);
    let (status, _) = send(
        "PUT",
        "/v1/index/other".to_string(),
        json!({ "dim": 4 }).to_string(),
    )
    .await;
    assert_eq!(status, StatusCode::FORBIDDEN);

    // Reads are unaffected, which is the point of running read-only.
    let search = json!({ "index": "docs", "query": vec![0.0f32; DIM], "k": 1 }).to_string();
    let (status, body) = send("POST", "/v1/search".to_string(), search).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["results"][0]["hits"].as_array().unwrap().len(), 1);
    let (status, body) = send("GET", "/v1/index/docs/stats".to_string(), String::new()).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["dim"], DIM);
    assert_eq!(body["sealed_live"], 8);
}

#[tokio::test]
async fn concurrent_searches_agree_with_a_serial_one() {
    let harness = Harness::new();
    let vectors = harness.seed(256).await;
    let query = vectors[100].clone();

    let (_, reference) = harness
        .call(
            "POST",
            "/v1/search",
            json!({ "index": "docs", "query": query, "k": 10, "ef": 64 }),
        )
        .await;
    let expected: BTreeSet<u64> = hit_ids(&reference).into_iter().collect();

    let mut tasks = Vec::new();
    for _ in 0..8 {
        let app = harness.app.clone();
        let query = query.clone();
        tasks.push(tokio::spawn(async move {
            let mut seen = BTreeSet::new();
            for _ in 0..4 {
                let request = Request::builder()
                    .method("POST")
                    .uri("/v1/search")
                    .header(header::CONTENT_TYPE, "application/json")
                    .body(Body::from(
                        json!({ "index": "docs", "query": query, "k": 10, "ef": 64 }).to_string(),
                    ))
                    .unwrap();
                let response = app.clone().oneshot(request).await.unwrap();
                assert_eq!(response.status(), StatusCode::OK);
                let bytes = to_bytes(response.into_body(), 1 << 20).await.unwrap();
                let body: Value = serde_json::from_slice(&bytes).unwrap();
                seen.extend(hit_ids(&body));
            }
            seen
        }));
    }
    for task in tasks {
        let seen = task.await.unwrap();
        assert_eq!(seen, expected, "a concurrent search disagreed");
    }
}

#[tokio::test]
async fn queries_use_the_configured_ef_when_none_is_given() {
    let harness = Harness::new();
    harness
        .call(
            "PUT",
            "/v1/index/docs",
            json!({ "dim": DIM, "metric": "l2", "ef_search": 111 }),
        )
        .await;
    let vectors = deterministic_vectors(16);
    let points: Vec<Value> = vectors
        .iter()
        .enumerate()
        .map(|(index, vector)| json!({ "id": index as u64, "vector": vector }))
        .collect();
    harness
        .call("POST", "/v1/index/docs/upsert", json!({ "points": points }))
        .await;

    let (_, body) = harness
        .call(
            "POST",
            "/v1/search",
            json!({ "index": "docs", "query": vectors[0], "k": 4 }),
        )
        .await;
    assert_eq!(body["ef"], 111);
    // A candidate list smaller than `k` is raised to `k` rather than honoured.
    let (_, body) = harness
        .call(
            "POST",
            "/v1/search",
            json!({ "index": "docs", "query": vectors[0], "k": 300, "ef": 1 }),
        )
        .await;
    assert_eq!(body["ef"], 300);
}

#[tokio::test]
async fn cosine_collections_rank_by_cosine_distance() {
    let harness = Harness::new();
    let (status, _) = harness
        .call(
            "PUT",
            "/v1/index/unit",
            json!({ "dim": 4, "metric": "cosine" }),
        )
        .await;
    assert_eq!(status, StatusCode::CREATED);
    let points = json!({
        "points": [
            { "id": 1, "vector": [1.0, 0.0, 0.0, 0.0] },
            { "id": 2, "vector": [0.0, 1.0, 0.0, 0.0] },
        ]
    });
    let (status, body) = harness
        .call("POST", "/v1/index/unit/upsert?flush=true", points)
        .await;
    assert_eq!(status, StatusCode::OK, "{body}");

    let (_, body) = harness
        .call(
            "POST",
            "/v1/search",
            json!({ "index": "unit", "query": [3.0, 0.0, 0.0, 0.0], "k": 2 }),
        )
        .await;
    assert_eq!(body["metric"], "cosine");
    let ids = hit_ids(&body);
    assert_eq!(ids[0], 1, "cosine ignores magnitude: {body}");
    let distance = body["results"][0]["hits"][0]["distance"].as_f64().unwrap();
    assert!(distance < 1e-5, "cosine self distance was {distance}");
}

#[tokio::test]
async fn every_documented_route_is_reachable() {
    // A guard against the OpenAPI document describing a path the router does
    // not serve. Each route gets its own collection, so the test does not
    // depend on the order the document lists paths — one of them drops a
    // collection, and another creates one.
    let harness = Harness::new();
    // A collection for the routes that name one in the body rather than the path.
    harness.create("scratch").await;
    let vector = serde_json::to_string(&deterministic_vectors(1)[0]).unwrap();
    let document = lodestar_ann_server::openapi::document();
    let mut index = 0usize;

    for (path, item) in document["paths"].as_object().unwrap() {
        for (method, _) in item.as_object().unwrap() {
            let method = method.to_uppercase();
            let name = format!("route{index}");
            index += 1;
            let uri = path.replace("{name}", &name);
            let creates = method == "PUT" && path == "/v1/index/{name}";
            if !creates && path.contains("{name}") {
                // Sub-resources need a collection to talk about.
                let (status, body) = harness
                    .call("PUT", &format!("/v1/index/{name}"), json!({ "dim": DIM }))
                    .await;
                assert_eq!(status, StatusCode::CREATED, "{body}");
            }
            let body = match (method.as_str(), path.as_str()) {
                ("POST", "/v1/index/{name}/upsert") => {
                    format!("{{\"points\": [{{\"id\": 1, \"vector\": {vector}}}]}}")
                }
                ("POST", "/v1/index/{name}/delete") => "{\"ids\": [1]}".to_string(),
                ("POST", "/v1/search") => {
                    format!("{{\"index\": \"scratch\", \"query\": {vector}, \"k\": 1}}")
                }
                ("PUT", _) => format!("{{\"dim\": {DIM}}}"),
                _ => String::new(),
            };
            let request = Request::builder()
                .method(method.as_str())
                .uri(&uri)
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(body))
                .unwrap();
            let response = harness.app.clone().oneshot(request).await.unwrap();
            assert_ne!(
                response.status(),
                StatusCode::NOT_FOUND,
                "{method} {uri} is documented but not routed"
            );
        }
    }
    assert!(index >= 13, "the document should describe every route");
}
