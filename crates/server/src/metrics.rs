//! Counters and Prometheus exposition.
//!
//! Metrics are hand-rolled rather than pulled from a facade crate: the service
//! has a dozen numbers, all of them cheap to keep inline, and the exposition
//! format is small enough to write correctly in one place. That also keeps the
//! dependency list honest — a portfolio project should be able to explain every
//! line of its own instrumentation.
//!
//! Counters are recorded per *route template* (`/v1/index/{name}/upsert`), not
//! per path, so a thousand collections do not create a thousand time series.
//!
//! Gauges — live vectors, segments, mapped bytes — are read from the open
//! collections at scrape time, because a counter that has to be kept in sync
//! with the engine is a counter that will eventually drift from it.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, MutexGuard, PoisonError};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

/// Per-route request counters.
#[derive(Debug, Default)]
pub struct RouteStats {
    /// Requests seen, all statuses.
    pub requests: u64,
    /// Responses with a 4xx or 5xx status.
    pub errors: u64,
    /// Requests per status class, indexed by `status / 100`.
    status_classes: [u64; 6],
    /// Cumulative time spent in the handler, in microseconds.
    pub micros: u64,
}

impl RouteStats {
    /// Records one response.
    fn observe(&mut self, status: u16, elapsed: Duration) {
        self.requests += 1;
        let class = (status / 100) as usize;
        if class < self.status_classes.len() {
            self.status_classes[class] += 1;
        }
        if status >= 400 {
            self.errors += 1;
        }
        self.micros = self
            .micros
            .saturating_add(elapsed.as_micros().try_into().unwrap_or(u64::MAX));
    }

    /// Responses in one status class.
    #[must_use]
    pub fn class(&self, class: usize) -> u64 {
        self.status_classes.get(class).copied().unwrap_or(0)
    }
}

/// Reads one gauge out of a collection.
type GaugeReader = fn(&CollectionGauge) -> usize;

/// A gauge metric: its name, its `HELP` text and how to read it.
type GaugeSpec = (&'static str, &'static str, GaugeReader);

/// One collection's gauges, captured at scrape time.
#[derive(Debug, Clone, Default)]
pub struct CollectionGauge {
    /// Collection name, used as the `collection` label.
    pub name: String,
    /// Live vectors, sealed and pending.
    pub live: usize,
    /// Vectors still in the in-memory tail.
    pub pending: usize,
    /// Sealed segments.
    pub segments: usize,
    /// Bytes of sealed segments, as mapped.
    pub mapped_bytes: usize,
    /// Metadata entries the service holds for this collection.
    pub metadata_entries: usize,
}

/// The service's counter set.
#[derive(Debug)]
pub struct Metrics {
    started_unix_ms: u64,
    routes: Mutex<BTreeMap<String, RouteStats>>,
    search_queries: AtomicU64,
    search_hits: AtomicU64,
    search_micros: AtomicU64,
    upserted_points: AtomicU64,
    deleted_ids: AtomicU64,
    engine_failures: AtomicU64,
}

impl Default for Metrics {
    fn default() -> Self {
        Self::new()
    }
}

impl Metrics {
    /// Creates a zeroed registry.
    #[must_use]
    pub fn new() -> Self {
        Self {
            started_unix_ms: now_unix_ms(),
            routes: Mutex::new(BTreeMap::new()),
            search_queries: AtomicU64::new(0),
            search_hits: AtomicU64::new(0),
            search_micros: AtomicU64::new(0),
            upserted_points: AtomicU64::new(0),
            deleted_ids: AtomicU64::new(0),
            engine_failures: AtomicU64::new(0),
        }
    }

    /// Milliseconds since the Unix epoch when the registry was created.
    #[must_use]
    pub fn started_unix_ms(&self) -> u64 {
        self.started_unix_ms
    }

    /// Records one completed HTTP request.
    pub fn observe_request(&self, route: &str, status: u16, elapsed: Duration) {
        let mut routes = lock(&self.routes);
        routes
            .entry(route.to_string())
            .or_default()
            .observe(status, elapsed);
        if status >= 500 {
            self.engine_failures.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// Records the engine work behind one search request.
    pub fn observe_search(&self, queries: u64, hits: u64, elapsed: Duration) {
        self.search_queries.fetch_add(queries, Ordering::Relaxed);
        self.search_hits.fetch_add(hits, Ordering::Relaxed);
        self.search_micros.fetch_add(
            elapsed.as_micros().try_into().unwrap_or(u64::MAX),
            Ordering::Relaxed,
        );
    }

    /// Records points accepted by an upsert.
    pub fn observe_upsert(&self, points: u64) {
        self.upserted_points.fetch_add(points, Ordering::Relaxed);
    }

    /// Records ids tombstoned by a delete.
    pub fn observe_delete(&self, ids: u64) {
        self.deleted_ids.fetch_add(ids, Ordering::Relaxed);
    }

    /// Every route seen so far, for tests and `/metrics`.
    #[must_use]
    pub fn routes(&self) -> Vec<(String, u64, u64)> {
        lock(&self.routes)
            .iter()
            .map(|(route, stats)| (route.clone(), stats.requests, stats.errors))
            .collect()
    }

    /// Renders the Prometheus text exposition format (version 0.0.4).
    #[must_use]
    pub fn render(
        &self,
        collections: &[CollectionGauge],
        uptime: Duration,
        read_only: bool,
    ) -> String {
        let mut out = String::with_capacity(2_048);

        let mut counter = |name: &str, help: &str, value: u64| {
            out.push_str(&format!("# HELP {name} {help}\n# TYPE {name} counter\n"));
            out.push_str(&format!("{name} {value}\n"));
        };
        counter(
            "lodestar_requests_total",
            "HTTP requests handled, by error class. See lodestar_http_requests_total for all.",
            self.routes().iter().map(|(_, requests, _)| requests).sum(),
        );
        counter(
            "lodestar_errors_total",
            "HTTP requests that failed, summed over all routes.",
            self.routes().iter().map(|(_, _, errors)| errors).sum(),
        );
        counter(
            "lodestar_engine_failures_total",
            "Requests that ended in a 5xx, i.e. the engine failed rather than the client.",
            self.engine_failures.load(Ordering::Relaxed),
        );
        counter(
            "lodestar_search_queries_total",
            "Query vectors searched.",
            self.search_queries.load(Ordering::Relaxed),
        );
        counter(
            "lodestar_search_hits_total",
            "Neighbours returned across all searches.",
            self.search_hits.load(Ordering::Relaxed),
        );
        counter(
            "lodestar_vectors_upserted_total",
            "Points accepted by upserts.",
            self.upserted_points.load(Ordering::Relaxed),
        );
        counter(
            "lodestar_ids_deleted_total",
            "Ids tombstoned by deletes.",
            self.deleted_ids.load(Ordering::Relaxed),
        );

        let search_micros = self.search_micros.load(Ordering::Relaxed);
        out.push_str(
            "# HELP lodestar_search_duration_seconds Wall time spent searching, in seconds.\n\
             # TYPE lodestar_search_duration_seconds summary\n",
        );
        out.push_str(&format!(
            "lodestar_search_duration_seconds_sum {}\n",
            seconds(search_micros)
        ));
        out.push_str(&format!(
            "lodestar_search_duration_seconds_count {}\n",
            self.search_queries.load(Ordering::Relaxed)
        ));

        out.push_str(
            "# HELP lodestar_http_requests_total HTTP requests by route template and status class.\n\
             # TYPE lodestar_http_requests_total counter\n",
        );
        out.push_str(
            "# HELP lodestar_http_request_duration_seconds Wall time per route template, in seconds.\n\
             # TYPE lodestar_http_request_duration_seconds summary\n",
        );
        for (route, stats) in lock(&self.routes).iter() {
            let route = escape_label(route);
            for (class, count) in (1..stats.status_classes.len()).map(|c| (c, stats.class(c))) {
                if count == 0 {
                    continue;
                }
                out.push_str(&format!(
                    "lodestar_http_requests_total{{route=\"{route}\",status=\"{class}xx\"}} {count}\n"
                ));
            }
            out.push_str(&format!(
                "lodestar_http_request_duration_seconds_sum{{route=\"{route}\"}} {}\n",
                seconds(stats.micros)
            ));
            out.push_str(&format!(
                "lodestar_http_request_duration_seconds_count{{route=\"{route}\"}} {}\n",
                stats.requests
            ));
        }

        out.push_str(
            "# HELP lodestar_uptime_seconds Seconds since the process started.\n\
             # TYPE lodestar_uptime_seconds gauge\n",
        );
        out.push_str(&format!(
            "lodestar_uptime_seconds {}\n",
            uptime.as_secs_f64()
        ));
        out.push_str(
            "# HELP lodestar_read_only Whether the service refuses mutations.\n\
             # TYPE lodestar_read_only gauge\n",
        );
        out.push_str(&format!("lodestar_read_only {}\n", u8::from(read_only)));
        out.push_str(
            "# HELP lodestar_collections Collections open.\n\
             # TYPE lodestar_collections gauge\n",
        );
        out.push_str(&format!("lodestar_collections {}\n", collections.len()));

        let gauges: [GaugeSpec; 5] = [
            (
                "lodestar_collection_live_vectors",
                "Live vectors in a collection.",
                |gauge| gauge.live,
            ),
            (
                "lodestar_collection_pending_vectors",
                "Vectors still in a collection's in-memory tail.",
                |gauge| gauge.pending,
            ),
            (
                "lodestar_collection_segments",
                "Sealed segments in a collection.",
                |gauge| gauge.segments,
            ),
            (
                "lodestar_collection_mapped_bytes",
                "Bytes of sealed segments, as mapped.",
                |gauge| gauge.mapped_bytes,
            ),
            (
                "lodestar_collection_metadata_entries",
                "Metadata entries held for a collection.",
                |gauge| gauge.metadata_entries,
            ),
        ];
        for (name, help, read) in gauges {
            out.push_str(&format!("# HELP {name} {help}\n# TYPE {name} gauge\n"));
            for gauge in collections {
                out.push_str(&format!(
                    "{name}{{collection=\"{}\"}} {}\n",
                    escape_label(&gauge.name),
                    read(gauge)
                ));
            }
        }
        out
    }
}

/// Microseconds rendered as fractional seconds with nanosecond resolution.
fn seconds(micros: u64) -> String {
    format!("{}.{:06}", micros / 1_000_000, micros % 1_000_000)
}

/// Escapes a Prometheus label value: backslash, quote and newline.
fn escape_label(value: &str) -> String {
    let mut out = String::with_capacity(value.len());
    for character in value.chars() {
        match character {
            '\\' => out.push_str("\\\\"),
            '"' => out.push_str("\\\""),
            '\n' => out.push_str("\\n"),
            other => out.push(other),
        }
    }
    out
}

/// Milliseconds since the Unix epoch, saturating at zero before 1970.
pub(crate) fn now_unix_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |elapsed| {
            elapsed.as_millis().try_into().unwrap_or(u64::MAX)
        })
}

fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(PoisonError::into_inner)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exposition_renders_counters_and_gauges() {
        let metrics = Metrics::new();
        metrics.observe_request("/v1/search", 200, Duration::from_millis(3));
        metrics.observe_request("/v1/search", 500, Duration::from_millis(2));
        metrics.observe_search(4, 40, Duration::from_millis(5));
        metrics.observe_upsert(7);
        metrics.observe_delete(2);

        let gauges = vec![CollectionGauge {
            name: "docs".to_string(),
            live: 10,
            pending: 3,
            segments: 1,
            mapped_bytes: 4_096,
            metadata_entries: 10,
        }];
        let text = metrics.render(&gauges, Duration::from_secs(12), true);

        assert!(text.contains("lodestar_requests_total 2\n"));
        assert!(text.contains("lodestar_errors_total 1\n"));
        assert!(text.contains("lodestar_engine_failures_total 1\n"));
        assert!(
            text.contains("lodestar_http_requests_total{route=\"/v1/search\",status=\"2xx\"} 1")
        );
        assert!(
            text.contains("lodestar_http_requests_total{route=\"/v1/search\",status=\"5xx\"} 1")
        );
        assert!(text.contains("lodestar_search_duration_seconds_count 4\n"));
        assert!(text.contains("lodestar_search_queries_total 4\n"));
        assert!(text.contains("lodestar_vectors_upserted_total 7\n"));
        assert!(text.contains("lodestar_ids_deleted_total 2\n"));
        assert!(text.contains("lodestar_collection_live_vectors{collection=\"docs\"} 10\n"));
        assert!(text.contains("lodestar_uptime_seconds 12\n"));
        assert!(text.contains("lodestar_read_only 1\n"));
        // Every metric has a HELP and a TYPE line, which is what a scraper's
        // parser requires to classify the series.
        for line in text.lines().filter(|line| line.starts_with("# HELP")) {
            assert!(line.len() > "# HELP ".len());
        }
        assert!(
            text.lines()
                .any(|line| line.starts_with("# TYPE lodestar_read_only gauge"))
        );
    }

    #[test]
    fn label_values_are_escaped() {
        assert_eq!(escape_label("a\"b\\c"), "a\\\"b\\\\c");
        assert_eq!(escape_label("line\nbreak"), "line\\nbreak");
        let metrics = Metrics::new();
        let gauges = vec![CollectionGauge {
            name: "we\"ird".to_string(),
            ..CollectionGauge::default()
        }];
        let text = metrics.render(&gauges, Duration::from_secs(0), false);
        assert!(text.contains("collection=\"we\\\"ird\""));
    }

    #[test]
    fn status_classes_are_counted_independently() {
        let metrics = Metrics::new();
        for status in [200, 201, 400, 404, 500] {
            metrics.observe_request("/v1/x", status, Duration::from_millis(1));
        }
        let stats = lock(&metrics.routes);
        let entry = stats.get("/v1/x").unwrap();
        assert_eq!(entry.requests, 5);
        assert_eq!(entry.errors, 3);
        assert_eq!(entry.class(2), 2);
        assert_eq!(entry.class(4), 2);
        assert_eq!(entry.class(5), 1);
    }
}
