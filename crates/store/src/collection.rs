//! The collection handle: open, write, search, flush, verify.
//!
//! A collection is a directory holding a manifest, zero or more immutable
//! segments, and a write-ahead log. Writes land in an in-memory HNSW *tail* and
//! are logged before they are acknowledged; [`Collection::flush`] seals the tail
//! into a new segment, which makes it searchable from the mapping and empties
//! the log. This is the log-structured merge shape, applied to vectors: the tail
//! absorbs writes, sealed segments are immutable, and a query merges both.
//!
//! # Deletes and overwrites
//!
//! A sealed segment cannot be edited, so a delete or an overwrite records a
//! *tombstone* against every segment that existed at the time. The tombstone
//! lists live in the manifest next to the segments they hide, and disappear with
//! those segments the next time the collection is compacted. A newer copy of an
//! id therefore never hides itself: a tombstone only ever applies to segments
//! older than the write that created it.
//!
//! # Id lookup
//!
//! The tail has an id map; sealed segments do not. Answering "does id X exist?"
//! exactly would need an index over every sealed segment, and this build does
//! not keep one. `insert`-style writes do not need it (they tombstone
//! unconditionally), and neither does search, so the cost is a missing
//! convenience rather than a missing capability — a limitation recorded here
//! rather than papered over.

use std::collections::HashSet;
use std::path::{Path, PathBuf};

use lodestar_ann_core::{Candidate, Error as CoreError, Metric};
use lodestar_ann_index::graph::{GraphView, SearchScratch};
use lodestar_ann_index::hnsw::{Hnsw, HnswConfig};

use crate::error::{Error, Result};
use crate::manifest::{Manifest, SegmentRef};
use crate::mapped::Segment;
use crate::segment::{self, SegmentInfo, now_unix_ms};
use crate::wal::{Wal, WalOp};

/// File name of the manifest inside a collection directory.
pub const MANIFEST_FILE: &str = "manifest.json";
/// File name of the write-ahead log inside a collection directory.
pub const WAL_FILE: &str = "wal.log";

/// Counters describing a collection.
#[derive(Debug, Clone, PartialEq)]
pub struct CollectionStats {
    /// Collection name.
    pub name: String,
    /// Vector dimensionality.
    pub dim: usize,
    /// Ranking metric.
    pub metric: Metric,
    /// Number of sealed segments.
    pub segments: usize,
    /// Nodes across sealed segments, tombstones included.
    pub sealed_nodes: usize,
    /// Live nodes across sealed segments, tombstones excluded.
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
}

/// A durable, searchable collection of vectors.
#[derive(Debug)]
pub struct Collection {
    directory: PathBuf,
    manifest_path: PathBuf,
    manifest: Manifest,
    config: HnswConfig,
    segments: Vec<Segment>,
    /// Ids hidden in each segment, parallel to `segments`.
    deleted: Vec<HashSet<u64>>,
    /// Writes that are not part of a sealed segment yet.
    tail: Hnsw,
    wal: Wal,
}

impl Collection {
    /// Creates a new collection directory.
    ///
    /// # Errors
    ///
    /// * [`Error::CollectionExists`] if the directory already holds a manifest.
    /// * [`Error::Io`] for filesystem failures.
    /// * [`Error::Core`] if the graph configuration is invalid.
    pub fn create(
        root: impl AsRef<Path>,
        name: &str,
        dim: usize,
        metric: Metric,
        config: HnswConfig,
    ) -> Result<Self> {
        config.validate()?;
        let directory = root.as_ref().join(name);
        let manifest_path = directory.join(MANIFEST_FILE);
        if manifest_path.exists() {
            return Err(Error::CollectionExists(name.to_string()));
        }
        std::fs::create_dir_all(&directory).map_err(|source| Error::Io {
            path: directory.clone(),
            source,
        })?;
        let manifest = Manifest::new(name, dim, metric, config);
        manifest.validate()?;
        manifest.store(&manifest_path)?;
        let tail = Hnsw::new(dim, metric, config)?;
        let wal = Wal::open(directory.join(WAL_FILE))?;
        Ok(Self {
            directory,
            manifest_path,
            manifest,
            config,
            segments: Vec::new(),
            deleted: Vec::new(),
            tail,
            wal,
        })
    }

    /// Opens an existing collection, replaying the log.
    ///
    /// # Errors
    ///
    /// * [`Error::NoSuchCollection`] if the directory has no manifest.
    /// * [`Error::Corrupt`] if a manifest, segment or log record fails
    ///   validation. A torn log tail is repaired rather than reported.
    /// * [`Error::Io`] for filesystem failures.
    pub fn open(root: impl AsRef<Path>, name: &str) -> Result<Self> {
        let directory = root.as_ref().join(name);
        let manifest_path = directory.join(MANIFEST_FILE);
        if !manifest_path.exists() {
            return Err(Error::NoSuchCollection(name.to_string()));
        }
        // A crash can leave a partially written segment or manifest behind.
        // Neither is reachable from the manifest, so removing them is safe and
        // they are the only thing that would otherwise accumulate.
        segment::remove_temporaries(&directory)?;
        let manifest = Manifest::load(&manifest_path)?;
        remove_orphan_segments(&directory, &manifest)?;

        let mut segments = Vec::with_capacity(manifest.segments.len());
        let mut deleted = Vec::with_capacity(manifest.segments.len());
        for reference in &manifest.segments {
            let segment = Segment::open(manifest.segment_path(&directory, reference))?;
            if segment.dim() != manifest.dim || segment.metric() != manifest.metric {
                return Err(Error::corrupt(
                    "collection",
                    format!(
                        "segment `{}` is {}d {:?} but the manifest says {}d {:?}",
                        reference.file,
                        segment.dim(),
                        segment.metric(),
                        manifest.dim,
                        manifest.metric
                    ),
                ));
            }
            segments.push(segment);
            deleted.push(reference.deleted.iter().copied().collect::<HashSet<_>>());
        }

        let config: HnswConfig = manifest.config.into();
        config.validate()?;
        let mut tail = Hnsw::new(manifest.dim, manifest.metric, config)?;
        let wal_path = directory.join(WAL_FILE);
        let entries = crate::wal::replay_and_repair(&wal_path, manifest.dim)?;
        for entry in &entries {
            // Both operations hide any older copy: the new value lives in the
            // tail, and the delete is a tombstone for everything sealed.
            for set in &mut deleted {
                set.insert(entry.id);
            }
            match &entry.op {
                WalOp::Upsert(vector) => tail.insert(entry.id, vector)?,
                WalOp::Delete => {
                    tail.delete(entry.id);
                }
            }
        }
        let wal = Wal::open(&wal_path)?;
        Ok(Self {
            directory,
            manifest_path,
            manifest,
            config,
            segments,
            deleted,
            tail,
            wal,
        })
    }

    /// Opens a collection if it exists, otherwise creates it.
    ///
    /// # Errors
    ///
    /// As [`Collection::create`] and [`Collection::open`].
    pub fn open_or_create(
        root: impl AsRef<Path>,
        name: &str,
        dim: usize,
        metric: Metric,
        config: HnswConfig,
    ) -> Result<Self> {
        match Self::open(&root, name) {
            Err(Error::NoSuchCollection(_)) => Self::create(root, name, dim, metric, config),
            other => other,
        }
    }

    /// Collection name.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.manifest.name
    }

    /// Directory holding the collection.
    #[must_use]
    pub fn directory(&self) -> &Path {
        &self.directory
    }

    /// Vector dimensionality.
    #[must_use]
    pub fn dim(&self) -> usize {
        self.manifest.dim
    }

    /// Ranking metric.
    #[must_use]
    pub fn metric(&self) -> Metric {
        self.manifest.metric
    }

    /// Graph parameters.
    #[must_use]
    pub fn config(&self) -> HnswConfig {
        self.config
    }

    /// Manifest as currently held in memory.
    #[must_use]
    pub fn manifest(&self) -> &Manifest {
        &self.manifest
    }

    /// Number of searchable vectors: live nodes in the tail plus live,
    /// non-tombstoned nodes in sealed segments.
    #[must_use]
    pub fn live_len(&self) -> usize {
        let sealed: usize = self
            .segments
            .iter()
            .enumerate()
            .map(|(index, segment)| {
                segment
                    .live_count()
                    .saturating_sub(self.deleted[index].len())
            })
            .sum();
        sealed + self.tail.len()
    }

    /// Number of vectors buffered in the tail and not yet sealed.
    #[must_use]
    pub fn pending_len(&self) -> usize {
        self.tail.node_count()
    }

    /// Writes a vector, durably, then returns.
    ///
    /// The log is synced before this returns, so an acknowledged write survives
    /// a crash. Bulk loads should use [`Collection::upsert_batch`], which syncs
    /// once per batch instead of once per vector.
    ///
    /// # Errors
    ///
    /// * [`Error::DimensionMismatch`](lodestar_ann_core::Error::DimensionMismatch)
    ///   if the vector length is wrong.
    /// * [`Error::Io`] if the log cannot be written.
    pub fn upsert(&mut self, id: u64, vector: &[f32]) -> Result<()> {
        self.check_dim(vector.len())?;
        self.wal.append_upsert(id, vector)?;
        self.wal.sync()?;
        self.apply_upsert(id, vector)
    }

    /// Writes many vectors, syncing the log once at the end.
    ///
    /// # Errors
    ///
    /// As [`Collection::upsert`], plus
    /// [`Error::DimensionMismatch`](lodestar_ann_core::Error::DimensionMismatch)
    /// if `ids.len()` does not match the number of rows.
    pub fn upsert_batch(&mut self, ids: &[u64], vectors: &[f32]) -> Result<usize> {
        let dim = self.dim();
        if vectors.len() % dim.max(1) != 0 {
            return Err(Error::Core(CoreError::DimensionMismatch {
                expected: (vectors.len() / dim.max(1)) * dim,
                actual: vectors.len(),
            }));
        }
        let rows = vectors.len() / dim.max(1);
        if ids.len() != rows {
            return Err(Error::Core(CoreError::DimensionMismatch {
                expected: rows,
                actual: ids.len(),
            }));
        }
        for (row, &id) in ids.iter().enumerate() {
            let vector = &vectors[row * dim..(row + 1) * dim];
            self.wal.append_upsert(id, vector)?;
        }
        self.wal.sync()?;
        for (row, &id) in ids.iter().enumerate() {
            let vector = &vectors[row * dim..(row + 1) * dim];
            self.apply_upsert(id, vector)?;
        }
        Ok(rows)
    }

    /// Deletes a vector.
    ///
    /// Deleting an unknown id is not an error: the tombstone is what matters,
    /// and it makes an id that arrives later through a stale segment stay hidden.
    ///
    /// # Errors
    ///
    /// [`Error::Io`] if the log cannot be written.
    pub fn delete(&mut self, id: u64) -> Result<()> {
        self.wal.append_delete(id)?;
        self.wal.sync()?;
        self.apply_delete(id);
        Ok(())
    }

    /// Deletes many ids, syncing once.
    ///
    /// # Errors
    ///
    /// As [`Collection::delete`].
    pub fn delete_batch(&mut self, ids: &[u64]) -> Result<usize> {
        for &id in ids {
            self.wal.append_delete(id)?;
        }
        self.wal.sync()?;
        for &id in ids {
            self.apply_delete(id);
        }
        Ok(ids.len())
    }

    /// Applies an upsert that is already in the log.
    fn apply_upsert(&mut self, id: u64, vector: &[f32]) -> Result<()> {
        // Every existing segment may hold an older copy of this id.
        for set in &mut self.deleted {
            set.insert(id);
        }
        self.tail.insert(id, vector)?;
        Ok(())
    }

    /// Applies a delete that is already in the log.
    fn apply_delete(&mut self, id: u64) {
        for set in &mut self.deleted {
            set.insert(id);
        }
        self.tail.delete(id);
    }

    fn check_dim(&self, actual: usize) -> Result<()> {
        if actual != self.dim() {
            return Err(Error::Core(CoreError::DimensionMismatch {
                expected: self.dim(),
                actual,
            }));
        }
        Ok(())
    }

    /// Searches the collection, allocating a scratch buffer.
    ///
    /// # Errors
    ///
    /// [`Error::DimensionMismatch`](lodestar_ann_core::Error::DimensionMismatch)
    /// if the query length is wrong.
    pub fn search(&self, query: &[f32], k: usize, ef: usize) -> Result<Vec<Candidate>> {
        let mut scratch = SearchScratch::new(self.total_nodes());
        self.search_with_scratch(query, k, ef, &mut scratch)
    }

    /// Searches the collection, reusing a caller-owned scratch buffer.
    ///
    /// A server should keep one scratch per worker: the buffer is one `u32` per
    /// node, and allocating it per query is pure overhead at scale.
    ///
    /// # Errors
    ///
    /// [`Error::DimensionMismatch`](lodestar_ann_core::Error::DimensionMismatch)
    /// if the query length is wrong.
    pub fn search_with_scratch(
        &self,
        query: &[f32],
        k: usize,
        ef: usize,
        scratch: &mut SearchScratch,
    ) -> Result<Vec<Candidate>> {
        self.check_dim(query.len())?;
        if k == 0 {
            return Ok(Vec::new());
        }
        // Each source must return enough candidates that, after tombstoned ids
        // are dropped, `k` live ones remain. Asking for `k + tombstones` per
        // source is exact rather than probabilistic: at most `tombstones` hits
        // can be discarded in total.
        let headroom = self.deleted.iter().map(HashSet::len).max().unwrap_or(0);
        let ask = k.saturating_add(headroom);

        let mut merged: Vec<Candidate> = Vec::with_capacity(ask + 1);
        for (index, segment) in self.segments.iter().enumerate() {
            if segment.live_count() == 0 {
                continue;
            }
            let hidden = &self.deleted[index];
            let hits = search_segment(segment, query, ask, ef, scratch)?;
            merged.extend(hits.into_iter().filter(|hit| !hidden.contains(&hit.id)));
        }
        merged.extend(
            self.tail
                .search_with_scratch(query, ask, ef, None, scratch)?,
        );

        // The same id can exist in more than one segment; the closest copy wins.
        merged.sort_unstable();
        let mut seen = HashSet::with_capacity(merged.len());
        merged.retain(|hit| seen.insert(hit.id));
        merged.truncate(k);
        Ok(merged)
    }

    /// Seals the tail into a new immutable segment.
    ///
    /// Returns the new segment's description, or `None` when the tail held
    /// nothing to seal.
    ///
    /// # Errors
    ///
    /// * [`Error::Io`] if the segment, the manifest or the log cannot be
    ///   written.
    /// * [`Error::Core`] if the tail cannot be compacted, which cannot happen for
    ///   a tail built through this type.
    pub fn flush(&mut self) -> Result<Option<SegmentInfo>> {
        let compacted = self.tail.compact()?;
        if compacted.is_empty() {
            // Nothing to seal. The log may still carry delete records whose
            // tombstones have no other durable record than the manifest's
            // per-segment deleted lists, so those lists are written before the
            // log is emptied. (A log of zero records means the manifest already
            // names every tombstone.) Doing it in the other order would lose
            // every delete of an id that lives only in sealed segments.
            if !self.wal.is_empty() {
                let mut manifest = self.manifest.clone();
                manifest.write_deleted(&self.deleted);
                manifest.updated_unix_ms = now_unix_ms();
                manifest.store(&self.manifest_path)?;
                self.manifest = manifest;
            }
            self.wal.reset()?;
            return Ok(None);
        }
        let file = format!("segment-{:06}.seg", self.manifest.next_segment);
        let path = self.directory.join(&file);
        let info = segment::write_segment(&path, &compacted.parts())?;

        // Ordering: the segment is complete before the manifest names it, and
        // the log is emptied only after the manifest is durable. A crash at any
        // point leaves a state that replays correctly.
        let mut manifest = self.manifest.clone();
        manifest.next_segment += 1;
        manifest.segments.push(SegmentRef {
            file,
            nodes: info.nodes,
            live: info.live,
            created_unix_ms: info.created_unix_ms,
            deleted: Vec::new(),
        });
        manifest.write_deleted(&self.deleted);
        manifest.updated_unix_ms = now_unix_ms();
        manifest.store(&self.manifest_path)?;

        let segment = Segment::open(&path)?;
        self.segments.push(segment);
        self.deleted.push(HashSet::new());
        self.manifest = manifest;
        self.tail = Hnsw::new(self.dim(), self.metric(), self.config)?;
        self.wal.reset()?;
        Ok(Some(info))
    }

    /// Rewrites every segment as one, dropping tombstones and deleted ids.
    ///
    /// Compaction is what bounds the manifest's tombstone lists and the number
    /// of segments a query has to merge.
    ///
    /// # Errors
    ///
    /// * [`Error::Io`] if a file cannot be written or removed.
    /// * [`Error::Core`] if the rebuilt index refuses a vector, which cannot
    ///   happen for vectors that came out of a valid collection.
    pub fn compact(&mut self) -> Result<Option<SegmentInfo>> {
        let mut rebuilt = Hnsw::new(self.dim(), self.metric(), self.config)?;
        // Rebuild in segment order, then the tail, so the newest copy of an id
        // is the one that survives. `insert` replaces an existing id in place,
        // which is what makes "newest wins" hold across sources.
        for (index, segment) in self.segments.iter().enumerate() {
            let hidden = &self.deleted[index];
            for node in 0..segment.len() as u32 {
                let id = segment.external_id(node);
                if !segment.is_live(node) || hidden.contains(&id) {
                    continue;
                }
                rebuilt.insert(id, segment.vector(node))?;
            }
        }
        for node in 0..self.tail.node_count() as u32 {
            if self.tail.is_live(node) {
                rebuilt.insert(self.tail.external_id(node), self.tail.vector(node))?;
            }
        }
        if rebuilt.is_empty() {
            // Everything was deleted: drop the segments and keep the manifest
            // honest about it.
            let mut manifest = self.manifest.clone();
            for reference in &manifest.segments {
                segment::remove_file_if_present(
                    &manifest.segment_path(&self.directory, reference),
                )?;
            }
            manifest.segments.clear();
            manifest.deleted.clear();
            manifest.updated_unix_ms = now_unix_ms();
            manifest.store(&self.manifest_path)?;
            self.segments.clear();
            self.deleted.clear();
            self.manifest = manifest;
            self.tail = rebuilt;
            self.wal.reset()?;
            return Ok(None);
        }
        let compacted = rebuilt.compact()?;
        let file = format!("segment-{:06}.seg", self.manifest.next_segment);
        let path = self.directory.join(&file);
        let info = segment::write_segment(&path, &compacted.parts())?;

        let mut manifest = self.manifest.clone();
        manifest.next_segment += 1;
        let replaced = std::mem::take(&mut manifest.segments);
        manifest.segments.push(SegmentRef {
            file,
            nodes: info.nodes,
            live: info.live,
            created_unix_ms: info.created_unix_ms,
            deleted: Vec::new(),
        });
        manifest.deleted.clear();
        manifest.updated_unix_ms = now_unix_ms();
        manifest.store(&self.manifest_path)?;

        // Only now is it safe to remove the files the manifest stopped naming.
        for reference in &replaced {
            segment::remove_file_if_present(&self.directory.join(&reference.file))?;
        }
        let segment = Segment::open(&path)?;
        self.segments = vec![segment];
        self.deleted = vec![HashSet::new()];
        self.manifest = manifest;
        self.tail = Hnsw::new(self.dim(), self.metric(), self.config)?;
        self.wal.reset()?;
        Ok(Some(info))
    }

    /// Runs the full checksum pass over every sealed segment.
    ///
    /// # Errors
    ///
    /// [`Error::Corrupt`] naming the first segment that fails.
    pub fn verify(&self) -> Result<()> {
        for segment in &self.segments {
            segment.verify()?;
        }
        Ok(())
    }

    /// Counters describing the collection.
    #[must_use]
    pub fn stats(&self) -> CollectionStats {
        let mut tombstoned = HashSet::new();
        for set in &self.deleted {
            tombstoned.extend(set.iter().copied());
        }
        CollectionStats {
            name: self.manifest.name.clone(),
            dim: self.dim(),
            metric: self.metric(),
            segments: self.segments.len(),
            sealed_nodes: self.segments.iter().map(Segment::len).sum(),
            sealed_live: self.segments.iter().map(Segment::live_count).sum(),
            tail_nodes: self.tail.node_count(),
            tail_live: self.tail.len(),
            tombstoned_ids: tombstoned.len(),
            wal_bytes: self.wal.len(),
            mapped_bytes: self.segments.iter().map(Segment::mapped_bytes).sum(),
            tail_bytes: self.tail.memory_bytes(),
            tail_max_level: self.tail.max_level(),
        }
    }

    /// Total nodes across segments and tail, which sizes a scratch buffer.
    #[must_use]
    pub fn total_nodes(&self) -> usize {
        self.segments.iter().map(Segment::len).sum::<usize>() + self.tail.node_count()
    }

    /// Descriptions of the sealed segments, oldest first.
    #[must_use]
    pub fn segments(&self) -> Vec<SegmentInfo> {
        self.segments.iter().map(Segment::info).collect()
    }
}

/// Searches one mapped segment with a caller-owned scratch buffer.
fn search_segment(
    segment: &Segment,
    query: &[f32],
    k: usize,
    ef: usize,
    scratch: &mut SearchScratch,
) -> Result<Vec<Candidate>> {
    if query.len() != segment.dim() {
        return Err(Error::Core(CoreError::DimensionMismatch {
            expected: segment.dim(),
            actual: query.len(),
        }));
    }
    let mut prepared = query.to_vec();
    segment.metric().prepare(&mut prepared);
    Ok(lodestar_ann_index::graph::search_with_scratch(
        segment, &prepared, k, ef, None, scratch,
    ))
}

/// Removes segment files the manifest does not name.
///
/// These are leftovers from a crash between "manifest written" and "old
/// segments deleted" during compaction, or from a killed flush.
fn remove_orphan_segments(directory: &Path, manifest: &Manifest) -> Result<usize> {
    let named: HashSet<&str> = manifest
        .segments
        .iter()
        .map(|reference| reference.file.as_str())
        .collect();
    let entries = std::fs::read_dir(directory).map_err(|source| Error::Io {
        path: directory.to_path_buf(),
        source,
    })?;
    let mut removed = 0;
    for entry in entries {
        let entry = entry.map_err(|source| Error::Io {
            path: directory.to_path_buf(),
            source,
        })?;
        let name = entry.file_name();
        let name = name.to_string_lossy().into_owned();
        if name.starts_with("segment-") && name.ends_with(".seg") && !named.contains(name.as_str())
        {
            std::fs::remove_file(entry.path()).map_err(|source| Error::Io {
                path: entry.path(),
                source,
            })?;
            removed += 1;
        }
    }
    Ok(removed)
}
