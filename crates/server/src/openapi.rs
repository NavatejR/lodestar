//! The API description.
//!
//! The document is hand-written rather than derived from the handlers. That is
//! a deliberate trade: a derive macro would make the spec a by-product of the
//! code, but it would also add a dependency, a build-time code generator and a
//! version-compatibility matrix to a service that has thirteen endpoints. A
//! hand-written document is checkable by eye, has zero dependencies, and —
//! because [`document`] is built with `serde_json` — cannot be syntactically
//! invalid JSON.
//!
//! The cost is honest and worth naming: the spec can drift from the handlers.
//! The integration tests in `tests/api.rs` are what keep it honest, by
//! exercising every path the document describes.

use serde_json::{Value, json};

/// The service name reported in the document.
pub const SERVICE: &str = "lodestar";
/// The engine version reported in the document.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Builds the OpenAPI 3.0 document.
#[must_use]
pub fn document() -> Value {
    json!({
        "openapi": "3.0.3",
        "info": {
            "title": "Lodestar",
            "version": VERSION,
            "description": "Vector search from scratch: HNSW over crash-safe, \
                memory-mapped segments, with a batch API, metadata filtering and \
                Prometheus metrics.",
            "license": { "name": "MIT OR Apache-2.0" }
        },
        "servers": [{ "url": "http://localhost:8080" }],
        "tags": [
            { "name": "service", "description": "Liveness and introspection." },
            { "name": "index", "description": "Collections: create, inspect, maintain, drop." },
            { "name": "vectors", "description": "Writing and deleting vectors." },
            { "name": "search", "description": "Nearest-neighbour queries." }
        ],
        "paths": {
            "/healthz": {
                "get": {
                    "tags": ["service"],
                    "summary": "Liveness and build information",
                    "operationId": "healthz",
                    "responses": {
                        "200": {
                            "description": "The service is up.",
                            "content": { "application/json": { "schema": { "$ref": "#/components/schemas/Health" } } }
                        }
                    }
                }
            },
            "/metrics": {
                "get": {
                    "tags": ["service"],
                    "summary": "Prometheus metrics in text exposition format",
                    "operationId": "metrics",
                    "responses": {
                        "200": {
                            "description": "Counters and gauges.",
                            "content": { "text/plain": { "schema": { "type": "string" } } }
                        }
                    }
                }
            },
            "/docs": {
                "get": {
                    "tags": ["service"],
                    "summary": "Human-readable documentation page",
                    "operationId": "docs",
                    "responses": { "200": { "description": "HTML." } }
                }
            },
            "/openapi.json": {
                "get": {
                    "tags": ["service"],
                    "summary": "This document",
                    "operationId": "openapi",
                    "responses": { "200": { "description": "The OpenAPI document." } }
                }
            },
            "/v1/index": {
                "get": {
                    "tags": ["index"],
                    "summary": "List collections",
                    "operationId": "listIndexes",
                    "responses": {
                        "200": {
                            "description": "Every open collection, sorted by name.",
                            "content": {
                                "application/json": {
                                    "schema": { "type": "array", "items": { "$ref": "#/components/schemas/IndexSummary" } }
                                }
                            }
                        }
                    }
                }
            },
            "/v1/index/{name}": {
                "put": {
                    "tags": ["index"],
                    "summary": "Create a collection",
                    "operationId": "createIndex",
                    "parameters": [{ "$ref": "#/components/parameters/Name" }],
                    "requestBody": {
                        "required": true,
                        "content": {
                            "application/json": {
                                "schema": { "$ref": "#/components/schemas/CreateIndexRequest" },
                                "example": { "dim": 384, "metric": "cosine", "m": 16, "ef_construction": 200 }
                            }
                        }
                    },
                    "responses": {
                        "201": { "description": "Created.", "content": { "application/json": { "schema": { "$ref": "#/components/schemas/IndexSummary" } } } },
                        "400": { "$ref": "#/components/responses/BadRequest" },
                        "403": { "$ref": "#/components/responses/Forbidden" },
                        "409": { "$ref": "#/components/responses/Conflict" }
                    }
                },
                "delete": {
                    "tags": ["index"],
                    "summary": "Drop a collection and delete its files",
                    "description": "Irreversible: the collection directory, its segments, its write-ahead log and its metadata log are removed.",
                    "operationId": "dropIndex",
                    "parameters": [{ "$ref": "#/components/parameters/Name" }],
                    "responses": {
                        "204": { "description": "Dropped." },
                        "403": { "$ref": "#/components/responses/Forbidden" },
                        "404": { "$ref": "#/components/responses/NotFound" }
                    }
                }
            },
            "/v1/index/{name}/stats": {
                "get": {
                    "tags": ["index"],
                    "summary": "Counters, sizes and the active SIMD kernel",
                    "operationId": "indexStats",
                    "parameters": [{ "$ref": "#/components/parameters/Name" }],
                    "responses": {
                        "200": { "description": "Statistics.", "content": { "application/json": { "schema": { "$ref": "#/components/schemas/Stats" } } } },
                        "404": { "$ref": "#/components/responses/NotFound" }
                    }
                }
            },
            "/v1/index/{name}/flush": {
                "post": {
                    "tags": ["index"],
                    "summary": "Seal the in-memory tail into a segment",
                    "operationId": "flushIndex",
                    "parameters": [{ "$ref": "#/components/parameters/Name" }],
                    "responses": {
                        "200": { "description": "The tail was sealed, or was already empty.", "content": { "application/json": { "schema": { "$ref": "#/components/schemas/Maintenance" } } } },
                        "403": { "$ref": "#/components/responses/Forbidden" },
                        "404": { "$ref": "#/components/responses/NotFound" }
                    }
                }
            },
            "/v1/index/{name}/compact": {
                "post": {
                    "tags": ["index"],
                    "summary": "Rewrite every live vector into a single segment",
                    "description": "Drops tombstones and collapses the segment list. Also rewrites the metadata log when it is mostly tombstones.",
                    "operationId": "compactIndex",
                    "parameters": [{ "$ref": "#/components/parameters/Name" }],
                    "responses": {
                        "200": { "description": "Compacted.", "content": { "application/json": { "schema": { "$ref": "#/components/schemas/Maintenance" } } } },
                        "403": { "$ref": "#/components/responses/Forbidden" },
                        "404": { "$ref": "#/components/responses/NotFound" }
                    }
                }
            },
            "/v1/index/{name}/verify": {
                "post": {
                    "tags": ["index"],
                    "summary": "Run the full checksum pass",
                    "description": "Re-reads every segment footer, recomputes the graph and vector checksums, and replays the metadata log.",
                    "operationId": "verifyIndex",
                    "parameters": [{ "$ref": "#/components/parameters/Name" }],
                    "responses": {
                        "200": { "description": "Everything checks out.", "content": { "application/json": { "schema": { "$ref": "#/components/schemas/Verify" } } } },
                        "404": { "$ref": "#/components/responses/NotFound" },
                        "500": { "$ref": "#/components/responses/Internal" }
                    }
                }
            },
            "/v1/index/{name}/upsert": {
                "post": {
                    "tags": ["vectors"],
                    "summary": "Insert or replace vectors, with optional metadata",
                    "description": "Re-using an id replaces that vector. Metadata is attached by id and indexed for filtering.",
                    "operationId": "upsert",
                    "parameters": [
                        { "$ref": "#/components/parameters/Name" },
                        { "name": "flush", "in": "query", "required": false, "schema": { "type": "boolean", "default": false }, "description": "Seal the tail into a segment before returning." }
                    ],
                    "requestBody": {
                        "required": true,
                        "content": {
                            "application/json": {
                                "schema": { "$ref": "#/components/schemas/UpsertRequest" },
                                "example": {
                                    "points": [
                                        { "id": 1, "vector": [0.1, 0.2, 0.3], "metadata": { "category": "news", "score": 0.9 } }
                                    ]
                                }
                            }
                        }
                    },
                    "responses": {
                        "200": { "description": "Written.", "content": { "application/json": { "schema": { "$ref": "#/components/schemas/Upsert" } } } },
                        "400": { "$ref": "#/components/responses/BadRequest" },
                        "403": { "$ref": "#/components/responses/Forbidden" },
                        "404": { "$ref": "#/components/responses/NotFound" },
                        "413": { "$ref": "#/components/responses/TooLarge" }
                    }
                }
            },
            "/v1/index/{name}/delete": {
                "post": {
                    "tags": ["vectors"],
                    "summary": "Tombstone ids",
                    "description": "Tombstones hide a vector from search immediately; compaction reclaims the space.",
                    "operationId": "deleteIds",
                    "parameters": [{ "$ref": "#/components/parameters/Name" }],
                    "requestBody": {
                        "required": true,
                        "content": {
                            "application/json": {
                                "schema": { "$ref": "#/components/schemas/DeleteRequest" },
                                "example": { "ids": [1, 2, 3] }
                            }
                        }
                    },
                    "responses": {
                        "200": { "description": "Tombstoned.", "content": { "application/json": { "schema": { "$ref": "#/components/schemas/Delete" } } } },
                        "403": { "$ref": "#/components/responses/Forbidden" },
                        "404": { "$ref": "#/components/responses/NotFound" }
                    }
                }
            },
            "/v1/search": {
                "post": {
                    "tags": ["search"],
                    "summary": "k-nearest-neighbour search",
                    "description": "One `query` or a batch of `queries`. A `filter` expression selects on metadata; \
                        because the engine stores vectors and ids only, a filtered search over-fetches and \
                        expands until it has `k` matches or runs out of live vectors.",
                    "operationId": "search",
                    "requestBody": {
                        "required": true,
                        "content": {
                            "application/json": {
                                "schema": { "$ref": "#/components/schemas/SearchRequest" },
                                "example": { "index": "docs", "query": [0.1, 0.2, 0.3], "k": 10, "ef": 64, "filter": "category == news", "include_metadata": true }
                            }
                        }
                    },
                    "responses": {
                        "200": { "description": "Neighbours, closest first.", "content": { "application/json": { "schema": { "$ref": "#/components/schemas/SearchResponse" } } } },
                        "400": { "$ref": "#/components/responses/BadRequest" },
                        "404": { "$ref": "#/components/responses/NotFound" },
                        "413": { "$ref": "#/components/responses/TooLarge" }
                    }
                }
            }
        },
        "components": {
            "parameters": {
                "Name": {
                    "name": "name",
                    "in": "path",
                    "required": true,
                    "description": "Collection name: letters, digits, `-`, `_` and `.`.",
                    "schema": { "type": "string", "maxLength": 128, "pattern": "^[A-Za-z0-9._-]+$" }
                }
            },
            "responses": {
                "BadRequest": { "description": "Malformed request.", "content": { "application/json": { "schema": { "$ref": "#/components/schemas/Error" } } } },
                "Forbidden": { "description": "The server is read-only.", "content": { "application/json": { "schema": { "$ref": "#/components/schemas/Error" } } } },
                "NotFound": { "description": "No such collection.", "content": { "application/json": { "schema": { "$ref": "#/components/schemas/Error" } } } },
                "Conflict": { "description": "The request contradicts existing state.", "content": { "application/json": { "schema": { "$ref": "#/components/schemas/Error" } } } },
                "TooLarge": { "description": "The request exceeds a documented limit.", "content": { "application/json": { "schema": { "$ref": "#/components/schemas/Error" } } } },
                "Internal": { "description": "The engine failed; the log has the detail.", "content": { "application/json": { "schema": { "$ref": "#/components/schemas/Error" } } } }
            },
            "schemas": {
                "Error": {
                    "type": "object",
                    "required": ["error"],
                    "properties": {
                        "error": {
                            "type": "object",
                            "required": ["code", "message"],
                            "properties": {
                                "code": { "type": "string", "enum": ["bad_request", "forbidden", "not_found", "conflict", "payload_too_large", "internal"] },
                                "message": { "type": "string" }
                            }
                        }
                    }
                },
                "Health": {
                    "type": "object",
                    "properties": {
                        "status": { "type": "string" },
                        "service": { "type": "string" },
                        "version": { "type": "string" },
                        "kernel": { "type": "string", "enum": ["scalar", "neon", "avx2"], "description": "SIMD kernel selected by this process." },
                        "uptime_seconds": { "type": "number" },
                        "collections": { "type": "integer" },
                        "read_only": { "type": "boolean" }
                    }
                },
                "CreateIndexRequest": {
                    "type": "object",
                    "required": ["dim"],
                    "properties": {
                        "dim": { "type": "integer", "minimum": 1 },
                        "metric": { "type": "string", "enum": ["l2", "cosine", "inner_product"], "default": "l2" },
                        "m": { "type": "integer", "minimum": 2, "default": 16 },
                        "m0": { "type": "integer", "minimum": 2, "default": 32 },
                        "ef_construction": { "type": "integer", "minimum": 1, "default": 200 },
                        "ef_search": { "type": "integer", "minimum": 1, "default": 64 },
                        "seed": { "type": "integer" }
                    }
                },
                "Point": {
                    "type": "object",
                    "required": ["id", "vector"],
                    "properties": {
                        "id": { "type": "integer", "format": "int64" },
                        "vector": { "type": "array", "items": { "type": "number", "format": "float" } },
                        "metadata": { "type": "object", "additionalProperties": true }
                    }
                },
                "UpsertRequest": {
                    "type": "object",
                    "required": ["points"],
                    "properties": {
                        "points": { "type": "array", "maxItems": 4096, "items": { "$ref": "#/components/schemas/Point" } }
                    }
                },
                "DeleteRequest": {
                    "type": "object",
                    "required": ["ids"],
                    "properties": {
                        "ids": { "type": "array", "maxItems": 100000, "items": { "type": "integer", "format": "int64" } }
                    }
                },
                "Segment": {
                    "type": "object",
                    "properties": {
                        "file": { "type": "string" },
                        "nodes": { "type": "integer" },
                        "live": { "type": "integer" },
                        "bytes": { "type": "integer" },
                        "created_unix_ms": { "type": "integer" }
                    }
                },
                "Upsert": {
                    "type": "object",
                    "properties": {
                        "index": { "type": "string" },
                        "upserted": { "type": "integer" },
                        "metadata_entries": { "type": "integer" },
                        "live": { "type": "integer" },
                        "pending": { "type": "integer" },
                        "sealed": { "$ref": "#/components/schemas/Segment" }
                    }
                },
                "Delete": {
                    "type": "object",
                    "properties": {
                        "index": { "type": "string" },
                        "deleted": { "type": "integer", "description": "Vectors this request removed from search. Deleting the same id twice removes one vector once." },
                        "metadata_entries": { "type": "integer" },
                        "live": { "type": "integer" }
                    }
                },
                "Maintenance": {
                    "type": "object",
                    "properties": {
                        "index": { "type": "string" },
                        "segment": { "$ref": "#/components/schemas/Segment" },
                        "live": { "type": "integer" },
                        "pending": { "type": "integer" },
                        "segments": { "type": "integer" }
                    }
                },
                "Verify": {
                    "type": "object",
                    "properties": {
                        "index": { "type": "string" },
                        "ok": { "type": "boolean" },
                        "nodes": { "type": "integer" },
                        "segments": { "type": "integer" },
                        "mapped_bytes": { "type": "integer" },
                        "metadata_entries": { "type": "integer" }
                    }
                },
                "IndexSummary": {
                    "type": "object",
                    "properties": {
                        "name": { "type": "string" },
                        "dim": { "type": "integer" },
                        "metric": { "type": "string" },
                        "live": { "type": "integer" },
                        "pending": { "type": "integer" },
                        "segments": { "type": "integer" },
                        "mapped_bytes": { "type": "integer" },
                        "metadata_entries": { "type": "integer" }
                    }
                },
                "Stats": {
                    "type": "object",
                    "properties": {
                        "name": { "type": "string" },
                        "directory": { "type": "string" },
                        "dim": { "type": "integer" },
                        "metric": { "type": "string" },
                        "kernel": { "type": "string" },
                        "live": { "type": "integer", "description": "Searchable vectors." },
                        "segments": { "type": "integer" },
                        "sealed_nodes": { "type": "integer" },
                        "sealed_live": { "type": "integer", "description": "Live, non-tombstoned sealed nodes; a lower bound, because a tombstone is recorded against every segment." },
                        "tail_nodes": { "type": "integer" },
                        "tail_live": { "type": "integer" },
                        "tombstoned_ids": { "type": "integer" },
                        "wal_bytes": { "type": "integer" },
                        "mapped_bytes": { "type": "integer" },
                        "tail_bytes": { "type": "integer" },
                        "tail_max_level": { "type": "integer" },
                        "id_index_bytes": { "type": "integer" },
                        "metadata_entries": { "type": "integer" }
                    }
                },
                "SearchRequest": {
                    "type": "object",
                    "required": ["index"],
                    "properties": {
                        "index": { "type": "string" },
                        "query": { "type": "array", "items": { "type": "number", "format": "float" } },
                        "queries": { "type": "array", "maxItems": 64, "items": { "type": "array", "items": { "type": "number", "format": "float" } } },
                        "k": { "type": "integer", "minimum": 1, "maximum": 1000, "default": 10 },
                        "ef": { "type": "integer", "minimum": 0, "description": "Candidate list size; 0 or absent uses the collection's configured value." },
                        "filter": {
                            "type": "string",
                            "maxLength": 4096,
                            "description": "Filter expression, for example `category == \"news\" AND score >= 0.8`. Comparisons against a missing field are false, including `!=`. Use `NOT field EXISTS` to select vectors with no value.",
                            "example": "category == news AND score >= 0.8"
                        },
                        "include_metadata": { "type": "boolean", "default": false }
                    }
                },
                "Hit": {
                    "type": "object",
                    "properties": {
                        "id": { "type": "integer", "format": "int64" },
                        "distance": { "type": "number", "format": "float", "description": "Smaller is closer, under the collection's metric." },
                        "metadata": { "type": "object", "additionalProperties": true }
                    }
                },
                "QueryResult": {
                    "type": "object",
                    "properties": {
                        "hits": { "type": "array", "items": { "$ref": "#/components/schemas/Hit" } },
                        "took_ms": { "type": "number" }
                    }
                },
                "SearchResponse": {
                    "type": "object",
                    "properties": {
                        "index": { "type": "string" },
                        "dim": { "type": "integer" },
                        "metric": { "type": "string" },
                        "k": { "type": "integer" },
                        "ef": { "type": "integer" },
                        "filter": { "type": "string" },
                        "results": { "type": "array", "items": { "$ref": "#/components/schemas/QueryResult" } },
                        "took_ms": { "type": "number" }
                    }
                }
            }
        }
    })
}

/// A self-contained documentation page.
///
/// No CDN, no bundled Swagger UI: the page fetches the document above and
/// renders the endpoint table in a few lines of vanilla JavaScript, so
/// `docker run` and an air-gapped demo both show the same thing.
pub const DOCS_HTML: &str = r#"<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Lodestar API</title>
<style>
  :root { color-scheme: dark light; --fg: #e6e6e6; --bg: #14161a; --dim: #9aa4b2; --line: #2a2f37; --accent: #7cc4ff; }
  body { margin: 0 auto; max-width: 60rem; padding: 2rem 1.25rem 4rem; background: var(--bg); color: var(--fg);
         font: 15px/1.55 ui-monospace, SFMono-Regular, Menlo, monospace; }
  h1 { font-size: 1.6rem; margin: 0 0 .25rem; }
  h2 { font-size: 1.05rem; margin: 2rem 0 .5rem; color: var(--accent); text-transform: lowercase; }
  p.lead { color: var(--dim); margin: 0 0 1.5rem; }
  table { width: 100%; border-collapse: collapse; }
  td, th { text-align: left; padding: .45rem .5rem; border-bottom: 1px solid var(--line); vertical-align: top; }
  th { color: var(--dim); font-weight: 500; }
  .method { font-weight: 700; white-space: nowrap; }
  .get { color: #7ee787; } .post { color: #ffd479; } .put { color: #d2a8ff; } .delete { color: #ff7b72; }
  code, pre { background: #1b1f26; border-radius: 6px; }
  code { padding: .1rem .35rem; }
  pre { padding: .8rem 1rem; overflow-x: auto; border: 1px solid var(--line); }
  a { color: var(--accent); }
  footer { margin-top: 2.5rem; color: var(--dim); }
</style>
</head>
<body>
<h1>Lodestar</h1>
<p class="lead">Vector search from scratch. <a href="/openapi.json">OpenAPI document</a> &middot; <a href="/metrics">metrics</a> &middot; <a href="/healthz">health</a></p>

<h2>quickstart</h2>
<pre># create a collection
curl -s localhost:8080/v1/index/docs -X PUT -H 'content-type: application/json' \
  -d '{"dim": 3, "metric": "l2"}'

# write two vectors, sealing them into a segment
curl -s 'localhost:8080/v1/index/docs/upsert?flush=true' -X POST -H 'content-type: application/json' \
  -d '{"points": [{"id": 1, "vector": [0, 0, 0], "metadata": {"category": "news"}},
                  {"id": 2, "vector": [1, 1, 1], "metadata": {"category": "blog"}}]}'

# search, filtered on metadata
curl -s localhost:8080/v1/search -X POST -H 'content-type: application/json' \
  -d '{"index": "docs", "query": [0, 0, 0], "k": 2, "filter": "category == news", "include_metadata": true}'</pre>

<h2>endpoints</h2>
<table id="endpoints"><thead><tr><th>method</th><th>path</th><th>summary</th></tr></thead><tbody></tbody></table>

<h2>filter syntax</h2>
<pre>category == "news" AND score &gt;= 0.8
tenant IN ("acme", "globex") AND NOT archived
tags CONTAINS "rust"
author EXISTS
NOT author EXISTS</pre>
<p class="lead">Comparisons against a field a vector does not define are false, including <code>!=</code>.
A filtered search over-fetches and expands until it has <code>k</code> matches or exhausts the live
vectors, so a very selective filter costs more than an unfiltered query.</p>

<footer>lodestar <span id="version"></span></footer>
<script>
fetch('/openapi.json').then(function (response) { return response.json(); }).then(function (spec) {
  var body = document.querySelector('#endpoints tbody');
  var order = ['service', 'index', 'vectors', 'search'];
  var rows = [];
  Object.keys(spec.paths).forEach(function (path) {
    Object.keys(spec.paths[path]).forEach(function (method) {
      var operation = spec.paths[path][method];
      rows.push({
        tag: (operation.tags || ['service'])[0],
        method: method.toUpperCase(),
        path: path,
        summary: operation.summary || ''
      });
    });
  });
  rows.sort(function (a, b) { return order.indexOf(a.tag) - order.indexOf(b.tag); });
  rows.forEach(function (row) {
    var tr = document.createElement('tr');
    var method = document.createElement('td');
    method.className = 'method ' + row.method.toLowerCase();
    method.textContent = row.method;
    var path = document.createElement('td');
    var link = document.createElement('a');
    link.href = '/openapi.json';
    link.textContent = row.path;
    path.appendChild(link);
    var summary = document.createElement('td');
    summary.textContent = row.summary;
    tr.appendChild(method); tr.appendChild(path); tr.appendChild(summary);
    body.appendChild(tr);
  });
  document.getElementById('version').textContent = spec.info.version;
}).catch(function (error) {
  document.querySelector('#endpoints tbody').innerHTML =
    '<tr><td colspan="3">could not load /openapi.json: ' + error + '</td></tr>';
});
</script>
</body>
</html>
"#;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_document_is_valid_and_covers_every_route() {
        let doc = document();
        assert_eq!(doc["openapi"], "3.0.3");
        let paths = doc["paths"].as_object().expect("paths object");
        for expected in [
            "/healthz",
            "/metrics",
            "/docs",
            "/openapi.json",
            "/v1/index",
            "/v1/index/{name}",
            "/v1/index/{name}/stats",
            "/v1/index/{name}/upsert",
            "/v1/index/{name}/delete",
            "/v1/index/{name}/flush",
            "/v1/index/{name}/compact",
            "/v1/index/{name}/verify",
            "/v1/search",
        ] {
            assert!(paths.contains_key(expected), "missing {expected}");
        }
        // Every method listed must be one of the verbs the router registers.
        for (path, item) in paths {
            for (method, _) in item.as_object().expect("path item") {
                assert!(
                    ["get", "put", "post", "delete"].contains(&method.as_str()),
                    "{method} on {path}"
                );
            }
        }
    }

    #[test]
    fn every_reference_resolves() {
        let doc = document();
        let schemas = doc["components"]["schemas"].as_object().unwrap();
        let parameters = doc["components"]["parameters"].as_object().unwrap();
        let responses = doc["components"]["responses"].as_object().unwrap();
        let text = serde_json::to_string(&doc).unwrap();
        for reference in text.split("#/components/schemas/").skip(1) {
            let name = reference.split('"').next().unwrap_or_default();
            assert!(schemas.contains_key(name), "unresolved schema `{name}`");
        }
        for reference in text.split("#/components/parameters/").skip(1) {
            let name = reference.split('"').next().unwrap_or_default();
            assert!(
                parameters.contains_key(name),
                "unresolved parameter `{name}`"
            );
        }
        for reference in text.split("#/components/responses/").skip(1) {
            let name = reference.split('"').next().unwrap_or_default();
            assert!(responses.contains_key(name), "unresolved response `{name}`");
        }
    }

    #[test]
    fn the_docs_page_is_self_contained() {
        assert!(DOCS_HTML.starts_with("<!doctype html>"));
        assert!(DOCS_HTML.contains("fetch('/openapi.json')"));
        // Nothing may be loaded from a network other than the service itself.
        assert!(!DOCS_HTML.contains("http://cdn"));
        assert!(!DOCS_HTML.contains("https://cdn"));
        assert!(!DOCS_HTML.contains("src=\"http"));
    }
}
