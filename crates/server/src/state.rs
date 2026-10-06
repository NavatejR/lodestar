//! Shared service state.
//!
//! One [`AppState`] lives behind every request. It owns
//!
//! * the data root and whether the service accepts writes,
//! * the collections that are open, each with its own locking, and
//! * the metric counters.
//!
//! ## Locking
//!
//! A [`Collection`] is `Send` but not `Sync`, and the engine's own docs point
//! at the reason: a search needs a mutable scratch buffer, which is why
//! [`AppState`] hands out one handle per collection holding
//!
//! * `RwLock<Collection>` — many concurrent searches, one writer. The lock is
//!   a `std` lock, not a `tokio` one, because every operation that touches it
//!   runs on the blocking pool inside `spawn_blocking`; holding an async lock
//!   across that work would only add overhead.
//! * `RwLock<MetadataStore>` — the sidecar, guarded separately so a filtered
//!   search does not block behind an unrelated metadata write.
//! * `Mutex<Vec<SearchScratch>>` — a small pool of reusable visited buffers, so
//!   a steady-state server allocates nothing per query.
//!
//! A poisoned lock is recovered rather than propagated: a panic in one request
//! must not turn the whole collection into a 500 for every later request, and
//! the state behind these locks is plain data.

use std::collections::{BTreeMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, MutexGuard, PoisonError, RwLock, RwLockReadGuard, RwLockWriteGuard};
use std::time::{Duration, Instant};

use lodestar_ann_core::{Candidate, Metric};
use lodestar_ann_index::hnsw::HnswConfig;
use lodestar_ann_index::{Expr, Metadata, SearchScratch};
use lodestar_ann_store::{Collection, CollectionStats, SegmentInfo};

use crate::dto::{IndexSummary, SegmentSummary};
use crate::error::ApiError;
use crate::metadata::MetadataStore;
use crate::metrics::{CollectionGauge, Metrics};

/// Longest accepted collection name.
pub const MAX_NAME_LEN: usize = 128;

/// How many idle scratch buffers one collection keeps.
const MAX_POOLED_SCRATCHES: usize = 64;

/// Rejects names that are empty, too long, or could escape the data root.
///
/// The store joins the name onto the root directory, so a name of `../..`
/// would write outside the data root. Nothing that arrives over HTTP gets to
/// choose a path.
///
/// # Errors
///
/// [`ApiError::BadRequest`] when the name is unusable.
pub fn validate_name(name: &str) -> Result<(), ApiError> {
    if name.is_empty() || name.len() > MAX_NAME_LEN {
        return Err(ApiError::BadRequest(format!(
            "collection name must be 1 to {MAX_NAME_LEN} characters"
        )));
    }
    if name == "." || name == ".." {
        return Err(ApiError::BadRequest(
            "collection name may not be `.` or `..`".to_string(),
        ));
    }
    if !name
        .chars()
        .all(|character| character.is_ascii_alphanumeric() || matches!(character, '-' | '_' | '.'))
    {
        return Err(ApiError::BadRequest(
            "collection name may contain only letters, digits, `-`, `_` and `.`".to_string(),
        ));
    }
    Ok(())
}

/// One open collection: the engine handle plus the service's own state.
#[derive(Debug)]
pub struct CollectionHandle {
    name: String,
    directory: PathBuf,
    collection: RwLock<Collection>,
    metadata: RwLock<MetadataStore>,
    scratches: Mutex<Vec<SearchScratch>>,
}

impl CollectionHandle {
    /// Opens an existing collection that lives under `root`.
    ///
    /// # Errors
    ///
    /// As [`Collection::open`], plus [`MetadataStore::open`] for a damaged
    /// metadata log.
    pub fn open(root: &Path, name: &str) -> lodestar_ann_store::Result<Self> {
        let collection = Collection::open(root, name)?;
        let directory = collection.directory().to_path_buf();
        let metadata = MetadataStore::open(&directory)?;
        Ok(Self {
            name: name.to_string(),
            collection: RwLock::new(collection),
            metadata: RwLock::new(metadata),
            scratches: Mutex::new(Vec::new()),
            directory,
        })
    }

    /// Creates a collection and opens it.
    ///
    /// # Errors
    ///
    /// As [`Collection::create`].
    pub fn create(
        root: &Path,
        name: &str,
        dim: usize,
        metric: Metric,
        config: HnswConfig,
    ) -> lodestar_ann_store::Result<Self> {
        let collection = Collection::create(root, name, dim, metric, config)?;
        let directory = collection.directory().to_path_buf();
        let metadata = MetadataStore::open(&directory)?;
        Ok(Self {
            name: name.to_string(),
            collection: RwLock::new(collection),
            metadata: RwLock::new(metadata),
            scratches: Mutex::new(Vec::new()),
            directory,
        })
    }

    /// Collection name.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Directory holding the collection.
    #[must_use]
    pub fn directory(&self) -> &Path {
        &self.directory
    }

    /// Vector dimensionality.
    #[must_use]
    pub fn dim(&self) -> usize {
        read_lock(&self.collection).dim()
    }

    /// Ranking metric.
    #[must_use]
    pub fn metric(&self) -> Metric {
        read_lock(&self.collection).metric()
    }

    /// Graph configuration the collection was built with.
    #[must_use]
    pub fn config(&self) -> HnswConfig {
        read_lock(&self.collection).config()
    }

    /// Counters for this collection.
    #[must_use]
    pub fn stats(&self) -> CollectionStats {
        read_lock(&self.collection).stats()
    }

    /// Searchable vectors: live tail nodes plus sealed nodes that are not
    /// tombstoned.
    #[must_use]
    pub fn live_len(&self) -> usize {
        read_lock(&self.collection).live_len()
    }

    /// Live metadata entries.
    #[must_use]
    pub fn metadata_len(&self) -> usize {
        read_lock(&self.metadata).len()
    }

    /// Runs `work` against the collection under a read lock.
    pub fn with_collection<T>(&self, work: impl FnOnce(&Collection) -> T) -> T {
        work(&read_lock(&self.collection))
    }

    /// Runs `work` against the collection under a write lock.
    pub fn with_collection_mut<T>(&self, work: impl FnOnce(&mut Collection) -> T) -> T {
        work(&mut write_lock(&self.collection))
    }

    /// Runs `work` against the metadata sidecar under a write lock.
    pub fn with_metadata_mut<T>(&self, work: impl FnOnce(&mut MetadataStore) -> T) -> T {
        work(&mut write_lock(&self.metadata))
    }

    /// Looks up metadata for the given ids, skipping ids that have none.
    #[must_use]
    pub fn metadata_for(&self, ids: &[u64]) -> BTreeMap<u64, Metadata> {
        let metadata = read_lock(&self.metadata);
        let mut found = BTreeMap::new();
        for id in ids {
            if let Some(entry) = metadata.get(*id) {
                found.insert(*id, entry.clone());
            }
        }
        found
    }

    /// Resolves a filter expression to the set of matching ids.
    #[must_use]
    pub fn matching_ids(&self, expr: &Expr) -> HashSet<u64> {
        read_lock(&self.metadata).matching(expr)
    }

    /// Searches for `k` neighbours, optionally restricted to `filter`.
    ///
    /// The engine cannot push a metadata predicate into traversal, so a
    /// filtered search over-fetches: it asks for `k`, and when the filter drops
    /// too many candidates it asks for four times as many, up to the number of
    /// live vectors. A selective filter therefore costs more than an unfiltered
    /// search, which the API documents rather than hides.
    ///
    /// # Errors
    ///
    /// [`ApiError::BadRequest`] if the query has the wrong length, or a store
    /// error if the search fails.
    pub fn search(
        &self,
        query: &[f32],
        k: usize,
        ef: usize,
        filter: Option<&HashSet<u64>>,
    ) -> Result<Vec<Candidate>, ApiError> {
        let collection = read_lock(&self.collection);
        let live = collection.live_len();
        let mut scratch = self.take_scratch(collection.total_nodes());
        let outcome = (|| -> lodestar_ann_store::Result<Vec<Candidate>> {
            let mut fetch = k.max(1);
            loop {
                let candidates =
                    collection.search_with_scratch(query, fetch, ef.max(fetch), &mut scratch)?;
                let kept: Vec<Candidate> = match filter {
                    None => return Ok(candidates),
                    Some(allowed) => candidates
                        .into_iter()
                        .filter(|hit| allowed.contains(&hit.id))
                        .collect(),
                };
                if kept.len() >= k || fetch >= live {
                    return Ok(kept);
                }
                // Four times the ask, never more than the collection holds, and
                // never a step that fails to grow: the loop must terminate even
                // if the collection is larger than `usize` arithmetic allows.
                let next = fetch
                    .saturating_mul(4)
                    .max(fetch.saturating_add(1))
                    .min(live);
                if next <= fetch {
                    return Ok(kept);
                }
                fetch = next;
            }
        })();
        self.give_scratch(scratch);
        let mut hits = outcome.map_err(ApiError::from)?;
        hits.truncate(k);
        Ok(hits)
    }

    /// Seals the tail into a segment.
    ///
    /// # Errors
    ///
    /// As [`Collection::flush`].
    pub fn flush(&self) -> lodestar_ann_store::Result<Option<SegmentInfo>> {
        write_lock(&self.collection).flush()
    }

    /// Rewrites every live vector into one segment.
    ///
    /// # Errors
    ///
    /// As [`Collection::compact`].
    pub fn compact(&self) -> lodestar_ann_store::Result<Option<SegmentInfo>> {
        write_lock(&self.collection).compact()
    }

    /// Runs the full checksum pass.
    ///
    /// # Errors
    ///
    /// As [`Collection::verify`].
    pub fn verify(&self) -> lodestar_ann_store::Result<()> {
        read_lock(&self.collection).verify()
    }

    /// Summary for `GET /v1/index`.
    #[must_use]
    pub fn summary(&self) -> IndexSummary {
        let stats = self.stats();
        IndexSummary {
            name: stats.name,
            dim: stats.dim,
            metric: stats.metric,
            live: stats.sealed_live + stats.tail_live,
            pending: stats.tail_nodes,
            segments: stats.segments,
            mapped_bytes: stats.mapped_bytes,
            metadata_entries: self.metadata_len(),
        }
    }

    /// Gauges for `/metrics`.
    #[must_use]
    pub fn gauge(&self) -> CollectionGauge {
        let stats = self.stats();
        CollectionGauge {
            name: stats.name,
            live: stats.sealed_live + stats.tail_live,
            pending: stats.tail_nodes,
            segments: stats.segments,
            mapped_bytes: stats.mapped_bytes,
            metadata_entries: self.metadata_len(),
        }
    }

    /// Takes a scratch buffer from the pool, growing it if the collection has
    /// gained nodes since it was last used.
    fn take_scratch(&self, nodes: usize) -> SearchScratch {
        let mut pool = mutex_lock(&self.scratches);
        let mut scratch = pool.pop().unwrap_or_default();
        scratch.ensure_capacity(nodes);
        scratch
    }

    /// Returns a scratch buffer to the pool, which stays bounded.
    fn give_scratch(&self, scratch: SearchScratch) {
        let mut pool = mutex_lock(&self.scratches);
        if pool.len() < MAX_POOLED_SCRATCHES {
            pool.push(scratch);
        }
    }
}

/// The state every request shares.
#[derive(Clone, Debug)]
pub struct AppState {
    inner: Arc<Inner>,
}

/// The non-`Clone` half of [`AppState`].
#[derive(Debug)]
struct Inner {
    root: PathBuf,
    read_only: bool,
    collections: RwLock<BTreeMap<String, Arc<CollectionHandle>>>,
    metrics: Metrics,
    started: Instant,
}

impl AppState {
    /// Opens every collection under `root`, creating the directory if needed.
    ///
    /// Collections are opened eagerly so that a damaged segment or manifest is
    /// reported at start-up, when an operator is watching, rather than on the
    /// first request that happens to touch it.
    ///
    /// # Errors
    ///
    /// A store error if the root cannot be created or any collection under it
    /// cannot be opened.
    pub fn open(root: impl Into<PathBuf>, read_only: bool) -> lodestar_ann_store::Result<Self> {
        let root = root.into();
        std::fs::create_dir_all(&root).map_err(|source| lodestar_ann_store::Error::Io {
            path: root.clone(),
            source,
        })?;
        let mut collections = BTreeMap::new();
        let entries = std::fs::read_dir(&root).map_err(|source| lodestar_ann_store::Error::Io {
            path: root.clone(),
            source,
        })?;
        for entry in entries {
            let entry = entry.map_err(|source| lodestar_ann_store::Error::Io {
                path: root.clone(),
                source,
            })?;
            let path = entry.path();
            if !path.is_dir() {
                continue;
            }
            let manifest = path.join(lodestar_ann_store::collection::MANIFEST_FILE);
            if !manifest.is_file() {
                continue;
            }
            let Some(name) = path.file_name().and_then(|name| name.to_str()) else {
                tracing::warn!(path = %path.display(), "skipping a collection whose name is not UTF-8");
                continue;
            };
            let handle = Arc::new(CollectionHandle::open(&root, name)?);
            tracing::info!(
                collection = name,
                dim = handle.dim(),
                metric = ?handle.metric(),
                live = handle.live_len(),
                "opened collection"
            );
            collections.insert(name.to_string(), handle);
        }
        Ok(Self {
            inner: Arc::new(Inner {
                root,
                read_only,
                collections: RwLock::new(collections),
                metrics: Metrics::new(),
                started: Instant::now(),
            }),
        })
    }

    /// Directory holding the collections.
    #[must_use]
    pub fn root(&self) -> &Path {
        &self.inner.root
    }

    /// Whether mutations are refused.
    #[must_use]
    pub fn read_only(&self) -> bool {
        self.inner.read_only
    }

    /// The metric registry.
    #[must_use]
    pub fn metrics(&self) -> &Metrics {
        &self.inner.metrics
    }

    /// Seconds since start-up.
    #[must_use]
    pub fn uptime(&self) -> Duration {
        self.inner.started.elapsed()
    }

    /// Number of collections currently open.
    #[must_use]
    pub fn len(&self) -> usize {
        read_lock(&self.inner.collections).len()
    }

    /// Whether no collection is open.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Fails when the service is read-only.
    ///
    /// # Errors
    ///
    /// [`ApiError::Forbidden`].
    pub fn require_writable(&self) -> Result<(), ApiError> {
        if self.inner.read_only {
            return Err(ApiError::Forbidden(
                "this server is read-only; restart it without --read-only to write".to_string(),
            ));
        }
        Ok(())
    }

    /// Returns the handle for `name`, opening it if the collection exists on
    /// disk but not in memory (which happens when the CLI created it while the
    /// service was running).
    ///
    /// # Errors
    ///
    /// [`ApiError::BadRequest`] for an unusable name, [`ApiError::NotFound`] if
    /// no such collection exists, or a store error if it cannot be opened.
    pub fn handle(&self, name: &str) -> Result<Arc<CollectionHandle>, ApiError> {
        validate_name(name)?;
        if let Some(handle) = read_lock(&self.inner.collections).get(name) {
            return Ok(Arc::clone(handle));
        }
        let directory = self.inner.root.join(name);
        if !directory
            .join(lodestar_ann_store::collection::MANIFEST_FILE)
            .is_file()
        {
            return Err(ApiError::NotFound(format!(
                "collection `{name}` does not exist"
            )));
        }
        let opened = Arc::new(CollectionHandle::open(&self.inner.root, name)?);
        let mut collections = write_lock(&self.inner.collections);
        Ok(Arc::clone(
            collections.entry(name.to_string()).or_insert(opened),
        ))
    }

    /// Registers a newly created collection.
    ///
    /// # Errors
    ///
    /// [`ApiError::Conflict`] if the name is already open.
    pub fn register(&self, handle: Arc<CollectionHandle>) -> Result<(), ApiError> {
        let mut collections = write_lock(&self.inner.collections);
        if collections.contains_key(handle.name()) {
            return Err(ApiError::Conflict(format!(
                "collection `{}` is already open",
                handle.name()
            )));
        }
        collections.insert(handle.name().to_string(), handle);
        Ok(())
    }

    /// Removes a collection from the open set, returning it.
    pub fn unregister(&self, name: &str) -> Option<Arc<CollectionHandle>> {
        write_lock(&self.inner.collections).remove(name)
    }

    /// Every open collection.
    #[must_use]
    pub fn handles(&self) -> Vec<Arc<CollectionHandle>> {
        read_lock(&self.inner.collections)
            .values()
            .cloned()
            .collect()
    }

    /// Summaries of every open collection, sorted by name.
    #[must_use]
    pub fn summaries(&self) -> Vec<IndexSummary> {
        self.handles()
            .iter()
            .map(|handle| handle.summary())
            .collect()
    }

    /// Gauges of every open collection, for the metrics scrape.
    #[must_use]
    pub fn gauges(&self) -> Vec<CollectionGauge> {
        self.handles().iter().map(|handle| handle.gauge()).collect()
    }
}

/// Converts a sealed segment into its wire form.
#[must_use]
pub fn segment_summary(info: &SegmentInfo) -> SegmentSummary {
    SegmentSummary {
        file: info
            .path
            .file_name()
            .map(|name| name.to_string_lossy().into_owned())
            .unwrap_or_else(|| info.path.to_string_lossy().into_owned()),
        nodes: info.nodes,
        live: info.live,
        bytes: info.bytes,
        created_unix_ms: info.created_unix_ms,
    }
}

pub(crate) fn read_lock<T>(lock: &RwLock<T>) -> RwLockReadGuard<'_, T> {
    lock.read().unwrap_or_else(PoisonError::into_inner)
}

pub(crate) fn write_lock<T>(lock: &RwLock<T>) -> RwLockWriteGuard<'_, T> {
    lock.write().unwrap_or_else(PoisonError::into_inner)
}

pub(crate) fn mutex_lock<T>(lock: &Mutex<T>) -> MutexGuard<'_, T> {
    lock.lock().unwrap_or_else(PoisonError::into_inner)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_shared_state_can_cross_threads() {
        // Axum requires the state to be `Send + Sync`. Asserting it here means
        // a future field that breaks the invariant fails in this test rather
        // than in a handler that only compiles for the author's machine.
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<AppState>();
        assert_send_sync::<Arc<CollectionHandle>>();
        assert_send_sync::<CollectionHandle>();
    }

    #[test]
    fn names_that_could_escape_the_root_are_rejected() {
        for bad in [
            "",
            ".",
            "..",
            "../etc",
            "a/b",
            "a\\b",
            "a b",
            "a*",
            &"x".repeat(129),
        ] {
            assert!(validate_name(bad).is_err(), "`{bad}` should be rejected");
        }
        for good in ["docs", "my-index", "my_index.v2", "a", "A9"] {
            assert!(validate_name(good).is_ok(), "`{good}` should be accepted");
        }
    }

    #[test]
    fn an_empty_root_opens_with_no_collections() {
        let directory = tempfile::tempdir().unwrap();
        let state = AppState::open(directory.path().join("data"), false).unwrap();
        assert!(state.is_empty());
        assert!(!state.read_only());
        assert!(state.root().is_dir());
        assert!(state.handles().is_empty());
        assert!(matches!(
            state.handle("missing"),
            Err(ApiError::NotFound(_))
        ));
    }

    #[test]
    fn a_collection_round_trips_through_create_and_open() {
        let directory = tempfile::tempdir().unwrap();
        let state = AppState::open(directory.path(), false).unwrap();
        let handle = Arc::new(
            CollectionHandle::create(
                directory.path(),
                "docs",
                4,
                Metric::L2,
                HnswConfig::default(),
            )
            .unwrap(),
        );
        state.register(Arc::clone(&handle)).unwrap();
        assert!(state.register(Arc::clone(&handle)).is_err());

        handle
            .with_collection_mut(|collection| {
                collection.upsert_batch(&[1, 2], &[0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0])
            })
            .unwrap();
        handle.flush().unwrap();

        // A second state over the same root finds the collection on disk.
        let reopened = AppState::open(directory.path(), true).unwrap();
        assert_eq!(reopened.len(), 1);
        let found = reopened.handle("docs").unwrap();
        assert_eq!(found.dim(), 4);
        assert_eq!(found.stats().sealed_live, 2);
        assert!(reopened.require_writable().is_err());

        let hits = found.search(&[0.0, 0.0, 0.0, 0.0], 2, 32, None).unwrap();
        assert_eq!(hits[0].id, 1);
    }

    #[test]
    fn scratch_buffers_are_reused_and_bounded() {
        // The scratch the first statement below takes is not returned before
        // the second statement asks for one, so the pool has room for it.
        let directory = tempfile::tempdir().unwrap();
        let handle = CollectionHandle::create(
            directory.path(),
            "docs",
            2,
            Metric::L2,
            HnswConfig::default(),
        )
        .unwrap();
        // A buffer that was grown once is reused at that size.
        let scratch = handle.take_scratch(100);
        assert!(scratch.capacity() >= 100);
        handle.give_scratch(scratch);
        assert!(handle.take_scratch(0).capacity() >= 100);

        // Handing back more buffers than the cap must not grow the pool.
        let held: Vec<_> = (0..MAX_POOLED_SCRATCHES * 2)
            .map(|_| handle.take_scratch(0))
            .collect();
        for scratch in held {
            handle.give_scratch(scratch);
        }
        assert_eq!(mutex_lock(&handle.scratches).len(), MAX_POOLED_SCRATCHES);
    }
}
