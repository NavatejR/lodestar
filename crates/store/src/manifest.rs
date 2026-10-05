//! The collection manifest.
//!
//! The manifest is the authority on what a collection contains: its dimensions,
//! its metric, the graph parameters, the segments that make up the index, and
//! the ids deleted since the last compaction. It is a small JSON document so
//! that a human can read it, diff it and repair it, which matters more for a
//! file whose corruption means "the index is gone" than a few bytes of compact
//! encoding would.
//!
//! It is replaced by writing a temporary file, syncing it, and renaming it over
//! the old one. Rename is atomic on every filesystem this project targets, so a
//! reader sees either the previous manifest or the next one and never a mixture.

use std::path::{Path, PathBuf};

use lodestar_ann_core::Metric;
use lodestar_ann_index::hnsw::HnswConfig;
use serde::{Deserialize, Serialize};

use crate::error::{Error, Result};
use crate::segment::now_unix_ms;

/// Manifest format version this build writes and reads.
pub const MANIFEST_VERSION: u32 = 1;

/// Reference to one segment file.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SegmentRef {
    /// File name, relative to the collection directory.
    pub file: String,
    /// Number of nodes, including tombstones.
    pub nodes: usize,
    /// Number of live nodes.
    pub live: usize,
    /// Creation time, milliseconds since the Unix epoch.
    pub created_unix_ms: u64,
    /// Ids hidden in this segment, sorted and unique.
    ///
    /// A tombstone is stored next to the segments it applies to, which is why it
    /// lives here rather than in a single collection-wide set: an id deleted now
    /// must not hide a copy written later, and segment order is what encodes
    /// "later".
    #[serde(default)]
    pub deleted: Vec<u64>,
}

/// Graph parameters, stored so that a reopened tail behaves like the one that
/// was written.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct GraphConfig {
    /// HNSW `m`.
    pub m: usize,
    /// HNSW `m0`.
    pub m0: usize,
    /// HNSW `ef_construction`.
    pub ef_construction: usize,
    /// HNSW `ef_search` default.
    pub ef_search: usize,
    /// Level-generation seed.
    pub seed: u64,
}

impl From<HnswConfig> for GraphConfig {
    fn from(config: HnswConfig) -> Self {
        Self {
            m: config.m,
            m0: config.m0,
            ef_construction: config.ef_construction,
            ef_search: config.ef_search,
            seed: config.seed,
        }
    }
}

impl From<GraphConfig> for HnswConfig {
    fn from(config: GraphConfig) -> Self {
        Self {
            m: config.m,
            m0: config.m0,
            ef_construction: config.ef_construction,
            ef_search: config.ef_search,
            seed: config.seed,
        }
    }
}

/// The collection manifest.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Manifest {
    /// Format version.
    pub format: u32,
    /// Collection name.
    pub name: String,
    /// Vector dimensionality.
    pub dim: usize,
    /// Ranking metric.
    pub metric: Metric,
    /// Graph parameters.
    pub config: GraphConfig,
    /// Sealed segments, oldest first.
    pub segments: Vec<SegmentRef>,
    /// Ids deleted since the last compaction, sorted and deduplicated.
    pub deleted: Vec<u64>,
    /// Number to give the next segment file.
    pub next_segment: u64,
    /// Last update time, milliseconds since the Unix epoch.
    pub updated_unix_ms: u64,
}

impl Manifest {
    /// Builds the manifest for a new, empty collection.
    #[must_use]
    pub fn new(name: &str, dim: usize, metric: Metric, config: HnswConfig) -> Self {
        Self {
            format: MANIFEST_VERSION,
            name: name.to_string(),
            dim,
            metric,
            config: config.into(),
            segments: Vec::new(),
            deleted: Vec::new(),
            next_segment: 1,
            updated_unix_ms: now_unix_ms(),
        }
    }

    /// Reads a manifest and checks that this build understands it.
    ///
    /// # Errors
    ///
    /// * [`Error::Io`] if the file cannot be read.
    /// * [`Error::Corrupt`] if it is not valid JSON or a field is missing.
    /// * [`Error::UnsupportedVersion`] if the format is newer.
    /// * [`Error::InvalidManifest`] if a field is present but unusable.
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let bytes = std::fs::read(path).map_err(|source| Error::Io {
            path: path.to_path_buf(),
            source,
        })?;
        let manifest: Self = serde_json::from_slice(&bytes).map_err(|source| {
            Error::corrupt(
                "manifest",
                format!("{} is not a valid manifest ({source})", path.display()),
            )
        })?;
        if manifest.format != MANIFEST_VERSION {
            return Err(Error::UnsupportedVersion {
                found: manifest.format,
                supported: MANIFEST_VERSION,
            });
        }
        manifest.validate()?;
        Ok(manifest)
    }

    /// Checks invariants the rest of the crate relies on.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidManifest`] describing the first problem found.
    pub fn validate(&self) -> Result<()> {
        if self.name.is_empty() || self.name.contains('/') || self.name.starts_with('.') {
            return Err(Error::InvalidManifest(format!(
                "`{}` is not a usable collection name",
                self.name
            )));
        }
        if self.dim == 0 || self.dim > 65_536 {
            return Err(Error::InvalidManifest(format!(
                "dimensionality {} is out of range",
                self.dim
            )));
        }
        if self.next_segment == 0 {
            return Err(Error::InvalidManifest(
                "next segment number must be positive".to_string(),
            ));
        }
        let mut previous: Option<u64> = None;
        for deleted in &self.deleted {
            if let Some(previous) = previous {
                if *deleted <= previous {
                    return Err(Error::InvalidManifest(
                        "deleted ids must be sorted and unique".to_string(),
                    ));
                }
            }
            previous = Some(*deleted);
        }
        for segment in &self.segments {
            if segment.file.is_empty()
                || segment.file.contains('/')
                || segment.file.starts_with('.')
            {
                return Err(Error::InvalidManifest(format!(
                    "`{}` is not a usable segment name",
                    segment.file
                )));
            }
            if segment.live > segment.nodes {
                return Err(Error::InvalidManifest(format!(
                    "segment `{}` claims {} live of {} nodes",
                    segment.file, segment.live, segment.nodes
                )));
            }
            let mut previous: Option<u64> = None;
            for deleted in &segment.deleted {
                if let Some(previous) = previous {
                    if *deleted <= previous {
                        return Err(Error::InvalidManifest(format!(
                            "segment `{}` has unsorted or repeated tombstones",
                            segment.file
                        )));
                    }
                }
                previous = Some(*deleted);
            }
        }
        let config: HnswConfig = self.config.into();
        config.validate()?;
        Ok(())
    }

    /// Writes the manifest atomically.
    ///
    /// # Errors
    ///
    /// * [`Error::Io`] if the temporary file cannot be written or renamed.
    /// * [`Error::InvalidManifest`] if the manifest does not validate.
    pub fn store(&self, path: impl AsRef<Path>) -> Result<()> {
        self.validate()?;
        let path = path.as_ref();
        let directory = path.parent().unwrap_or_else(|| Path::new("."));
        let temporary = directory.join(format!(
            ".{}.tmp-{}",
            path.file_name()
                .map(|name| name.to_string_lossy().into_owned())
                .unwrap_or_else(|| "manifest".to_string()),
            std::process::id()
        ));
        let bytes = serde_json::to_vec_pretty(self).map_err(|source| {
            Error::InvalidManifest(format!("manifest cannot be encoded: {source}"))
        })?;
        {
            use std::io::Write;
            let mut file = std::fs::File::create(&temporary).map_err(|source| Error::Io {
                path: temporary.clone(),
                source,
            })?;
            file.write_all(&bytes).map_err(|source| Error::Io {
                path: temporary.clone(),
                source,
            })?;
            file.write_all(b"\n").map_err(|source| Error::Io {
                path: temporary.clone(),
                source,
            })?;
            // The contents must be durable before the name is, otherwise a crash
            // could leave a manifest that names segments but is empty itself.
            file.sync_all().map_err(|source| Error::Io {
                path: temporary.clone(),
                source,
            })?;
        }
        std::fs::rename(&temporary, path).map_err(|source| Error::Io {
            path: path.to_path_buf(),
            source,
        })?;
        if let Ok(handle) = std::fs::File::open(directory) {
            let _ = handle.sync_all();
        }
        Ok(())
    }

    /// Records a segment and bumps the segment counter.
    pub fn push_segment(&mut self, reference: SegmentRef) {
        self.segments.push(reference);
    }

    /// Sorts and deduplicates the deleted-id list.
    pub fn normalize_deleted(&mut self) {
        self.deleted.sort_unstable();
        self.deleted.dedup();
    }

    /// Copies in-memory tombstone sets into the segment references.
    ///
    /// `sets` is parallel to `segments`, which is how the collection holds them:
    /// one set per sealed segment, in the same order.
    pub fn write_deleted(&mut self, sets: &[std::collections::HashSet<u64>]) {
        for (reference, set) in self.segments.iter_mut().zip(sets) {
            reference.deleted = set.iter().copied().collect();
            reference.deleted.sort_unstable();
        }
    }

    /// Path of a segment file named by this manifest, given the collection
    /// directory.
    #[must_use]
    pub fn segment_path(&self, directory: &Path, reference: &SegmentRef) -> PathBuf {
        directory.join(&reference.file)
    }
}
