//! The metadata sidecar.
//!
//! The storage engine persists ids and vectors and nothing else: an ANN index
//! that also has to own a document store is a different project. What the
//! engine *does* ship is a metadata filter language ([`lodestar_ann_index::filter`]),
//! and a filter is only useful if something remembers the metadata.
//!
//! So the service keeps it, in one append-only JSONL file per collection:
//!
//! ```text
//! {"id":7,"metadata":{"category":"news","score":0.9}}
//! {"id":7,"deleted":true}
//! ```
//!
//! ## Durability
//!
//! The file is the same shape as the engine's write-ahead log, and is treated
//! the same way:
//!
//! * records are appended and flushed per request, so they survive a process
//!   crash;
//! * a torn tail — a half-written record from a crash mid-append — is truncated
//!   on open and the file is rewritten, so the log is always complete and
//!   parseable;
//! * anything *before* a torn tail that fails to parse is
//!   [`Error::Corrupt`], never a panic;
//! * `flush` and `compact` call [`MetadataStore::sync`] before sealing vectors,
//!   so a segment is never durable while the metadata describing it is not;
//! * the log is rewritten atomically (temp file, `fsync`, rename, directory
//!   `fsync`) once it has more than twice as many records as live entries, so
//!   it cannot grow without bound.
//!
//! ## What it does not do
//!
//! Metadata for a deleted id is never reclaimed, because the engine cannot
//! enumerate the ids in a sealed segment. It is bounded by the log compaction
//! above, and filtering happens against live search results anyway.

use std::collections::{HashMap, HashSet};
use std::fs::{File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};

use lodestar_ann_index::{Expr, Metadata};
use lodestar_ann_store::{Error, Result};
use serde::{Deserialize, Serialize};

/// File name of the metadata log inside a collection directory.
pub const METADATA_FILE: &str = "metadata.jsonl";

/// Rewrite the log once it holds at least this many append-only records *and*
/// would shrink by half.
const COMPACT_MIN_RECORDS: usize = 1_024;

/// One line of the log.
#[derive(Debug, Serialize, Deserialize)]
struct Record {
    id: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    metadata: Option<Metadata>,
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    deleted: bool,
}

/// Append-only metadata for one collection, with an in-memory index.
#[derive(Debug)]
pub struct MetadataStore {
    /// Path of the log this store appends to.
    path: PathBuf,
    /// Live metadata, keyed by vector id.
    entries: HashMap<u64, Metadata>,
    /// Records in the log, live or not.
    appended: usize,
    /// Open append handle, created on first write.
    sink: Option<File>,
}

impl MetadataStore {
    /// Opens (or starts) the sidecar for the collection stored in `directory`.
    ///
    /// A missing file is an empty store; a torn tail is repaired.
    ///
    /// # Errors
    ///
    /// [`Error::Corrupt`] if a complete record cannot be parsed, or
    /// [`Error::Io`] if the file cannot be read or repaired.
    pub fn open(directory: &Path) -> Result<Self> {
        let path = directory.join(METADATA_FILE);
        let mut store = Self {
            path,
            entries: HashMap::new(),
            appended: 0,
            sink: None,
        };
        if !store.path.exists() {
            return Ok(store);
        }
        let bytes = std::fs::read(&store.path).map_err(|source| Error::Io {
            path: store.path.clone(),
            source,
        })?;
        let repaired = store.replay(&bytes)?;
        if repaired {
            // The tail was garbage from a crash: drop it now so the next open
            // does not have to reason about it again.
            tracing::warn!(path = %store.path.display(), "truncated a torn metadata record");
            store.rewrite()?;
        } else {
            store.appended = store.entries.len();
            // Count the records that a compacting rewrite would remove, so a
            // log inherited from a previous process still gets rewritten once
            // it is mostly tombstones.
            store.appended = count_records(&bytes)?;
        }
        Ok(store)
    }

    /// Number of live metadata entries.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the collection has no metadata at all.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Path of the log file.
    #[must_use]
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Looks up one id.
    #[must_use]
    pub fn get(&self, id: u64) -> Option<&Metadata> {
        self.entries.get(&id)
    }

    /// Every `(id, metadata)` pair, in unspecified order.
    pub fn entries(&self) -> impl Iterator<Item = (&u64, &Metadata)> {
        self.entries.iter()
    }

    /// The ids whose metadata satisfies `expr`.
    ///
    /// A vector with no metadata never matches, which is the same rule the
    /// filter language applies to a missing field.
    #[must_use]
    pub fn matching(&self, expr: &Expr) -> HashSet<u64> {
        self.entries
            .iter()
            .filter(|(_, metadata)| expr.matches(Some(metadata)))
            .map(|(id, _)| *id)
            .collect()
    }

    /// Attaches `metadata` to `id`, replacing anything already there.
    ///
    /// # Errors
    ///
    /// [`Error::Io`] if the record cannot be appended.
    pub fn set(&mut self, id: u64, metadata: Metadata) -> Result<()> {
        self.append(&Record {
            id,
            metadata: Some(metadata.clone()),
            deleted: false,
        })?;
        self.entries.insert(id, metadata);
        Ok(())
    }

    /// Drops the metadata of every id in `ids`, returning how many were held.
    ///
    /// # Errors
    ///
    /// [`Error::Io`] if a record cannot be appended.
    pub fn clear(&mut self, ids: &[u64]) -> Result<usize> {
        let mut cleared = 0;
        for id in ids {
            if self.entries.remove(id).is_some() {
                self.append(&Record {
                    id: *id,
                    metadata: None,
                    deleted: true,
                })?;
                cleared += 1;
            }
        }
        Ok(cleared)
    }

    /// Flushes pending bytes and asks the filesystem to make them durable.
    ///
    /// # Errors
    ///
    /// [`Error::Io`] if the sync fails.
    pub fn sync(&mut self) -> Result<()> {
        if let Some(sink) = self.sink.as_mut() {
            sink.flush().map_err(|source| Error::Io {
                path: self.path.clone(),
                source,
            })?;
            sink.sync_data().map_err(|source| Error::Io {
                path: self.path.clone(),
                source,
            })?;
        }
        Ok(())
    }

    /// Rewrites the log when it is mostly tombstones.
    ///
    /// # Errors
    ///
    /// [`Error::Io`] if the rewrite fails.
    pub fn compact_if_needed(&mut self) -> Result<bool> {
        let live = self.entries.len().max(1);
        if self.appended >= COMPACT_MIN_RECORDS && self.appended >= live * 2 {
            self.rewrite()?;
            return Ok(true);
        }
        Ok(false)
    }

    /// Rewrites the log so that it holds exactly one record per live entry.
    ///
    /// The new file is built beside the old one and renamed into place, so a
    /// crash leaves either the old log or the new one, never a mixture.
    ///
    /// # Errors
    ///
    /// [`Error::Io`] if the rewrite fails.
    pub fn rewrite(&mut self) -> Result<()> {
        let directory = self
            .path
            .parent()
            .map_or_else(|| PathBuf::from("."), Path::to_path_buf);
        let temporary = directory.join(format!(".{METADATA_FILE}.tmp-{}", std::process::id()));

        let mut buffer = Vec::new();
        let mut ids: Vec<u64> = self.entries.keys().copied().collect();
        // A stable order makes the rewrite reproducible, which makes a diff of
        // two collections meaningful.
        ids.sort_unstable();
        for id in ids {
            let Some(metadata) = self.entries.get(&id) else {
                continue;
            };
            let record = Record {
                id,
                metadata: Some(metadata.clone()),
                deleted: false,
            };
            serde_json::to_writer(&mut buffer, &record)
                .map_err(|error| corrupt("metadata record", error.to_string()))?;
            buffer.push(b'\n');
        }

        {
            let mut file = File::create(&temporary).map_err(|source| Error::Io {
                path: temporary.clone(),
                source,
            })?;
            file.write_all(&buffer).map_err(|source| Error::Io {
                path: temporary.clone(),
                source,
            })?;
            file.sync_all().map_err(|source| Error::Io {
                path: temporary.clone(),
                source,
            })?;
        }
        std::fs::rename(&temporary, &self.path).map_err(|source| Error::Io {
            path: self.path.clone(),
            source,
        })?;
        sync_directory(&directory)?;

        // The old handle points at a replaced file; drop it so the next append
        // opens the new one.
        self.sink = None;
        self.appended = self.entries.len();
        Ok(())
    }

    /// Re-reads the log and checks that it describes exactly what is in memory.
    ///
    /// # Errors
    ///
    /// [`Error::Corrupt`] if the file and the in-memory index disagree, or
    /// [`Error::Io`] if the file cannot be read.
    pub fn verify(&self) -> Result<()> {
        let bytes = std::fs::read(&self.path).map_err(|source| Error::Io {
            path: self.path.clone(),
            source,
        })?;
        let mut replayed: HashMap<u64, Metadata> = HashMap::new();
        replay_into(&bytes, &mut replayed, true)?;
        if replayed != self.entries {
            return Err(corrupt(
                "metadata log",
                format!(
                    "log holds {} live entries, memory holds {}",
                    replayed.len(),
                    self.entries.len()
                ),
            ));
        }
        Ok(())
    }

    /// Replays `bytes` into a fresh store, returning whether a torn tail was
    /// dropped.
    fn replay(&mut self, bytes: &[u8]) -> Result<bool> {
        let mut repaired = false;
        let mut offset = 0usize;
        for line in bytes.split_inclusive(|byte| *byte == b'\n') {
            let is_last = offset + line.len() == bytes.len();
            offset += line.len();
            let text = std::str::from_utf8(line)
                .map_err(|error| corrupt("metadata log", error.to_string()))?;
            let text = text.trim();
            if text.is_empty() {
                continue;
            }
            match serde_json::from_str::<Record>(text) {
                Ok(record) => {
                    if record.deleted {
                        self.entries.remove(&record.id);
                    } else if let Some(metadata) = record.metadata {
                        self.entries.insert(record.id, metadata);
                    } else {
                        self.entries.remove(&record.id);
                    }
                }
                Err(error) => {
                    // Only an unterminated final line can be a torn write; a
                    // broken record anywhere else means the file is damaged.
                    if is_last && !bytes.ends_with(b"\n") {
                        repaired = true;
                        break;
                    }
                    return Err(corrupt(
                        "metadata log",
                        format!("line at byte {offset}: {error}"),
                    ));
                }
            }
        }
        Ok(repaired)
    }

    /// Appends one record and advances the counters.
    fn append(&mut self, record: &Record) -> Result<()> {
        if self.sink.is_none() {
            let sink = OpenOptions::new()
                .create(true)
                .append(true)
                .open(&self.path)
                .map_err(|source| Error::Io {
                    path: self.path.clone(),
                    source,
                })?;
            self.sink = Some(sink);
        }
        let mut line = serde_json::to_vec(record)
            .map_err(|error| corrupt("metadata record", error.to_string()))?;
        line.push(b'\n');
        let sink = self.sink.as_mut().expect("sink was just opened");
        sink.write_all(&line).map_err(|source| Error::Io {
            path: self.path.clone(),
            source,
        })?;
        sink.flush().map_err(|source| Error::Io {
            path: self.path.clone(),
            source,
        })?;
        self.appended += 1;
        Ok(())
    }
}

/// Builds a [`Error::Corrupt`] without spelling the struct out every time.
fn corrupt(what: &str, detail: impl Into<String>) -> Error {
    Error::Corrupt {
        what: what.to_string(),
        detail: detail.into(),
    }
}

/// Replays a log into `into`, optionally ignoring a torn final line.
fn replay_into(bytes: &[u8], into: &mut HashMap<u64, Metadata>, allow_torn: bool) -> Result<()> {
    let mut offset = 0usize;
    for line in bytes.split_inclusive(|byte| *byte == b'\n') {
        let is_last = offset + line.len() == bytes.len();
        offset += line.len();
        let text = std::str::from_utf8(line)
            .map_err(|error| corrupt("metadata log", error.to_string()))?;
        let text = text.trim();
        if text.is_empty() {
            continue;
        }
        match serde_json::from_str::<Record>(text) {
            Ok(record) => {
                if record.deleted || record.metadata.is_none() {
                    into.remove(&record.id);
                } else if let Some(metadata) = record.metadata {
                    into.insert(record.id, metadata);
                }
            }
            Err(error) => {
                if allow_torn && is_last && !bytes.ends_with(b"\n") {
                    return Ok(());
                }
                return Err(corrupt(
                    "metadata log",
                    format!("line at byte {offset}: {error}"),
                ));
            }
        }
    }
    Ok(())
}

/// Counts the non-empty lines in a log.
fn count_records(bytes: &[u8]) -> Result<usize> {
    let mut count = 0usize;
    for line in bytes.split_inclusive(|byte| *byte == b'\n') {
        if std::str::from_utf8(line)
            .map_err(|error| corrupt("metadata log", error.to_string()))?
            .trim()
            .is_empty()
        {
            continue;
        }
        count += 1;
    }
    Ok(count)
}

/// Makes a rename durable. A directory cannot be opened as a file on Windows,
/// where the rename is atomic through the filesystem anyway.
#[cfg(unix)]
fn sync_directory(directory: &Path) -> Result<()> {
    let handle = File::open(directory).map_err(|source| Error::Io {
        path: directory.to_path_buf(),
        source,
    })?;
    handle.sync_all().map_err(|source| Error::Io {
        path: directory.to_path_buf(),
        source,
    })
}

#[cfg(not(unix))]
fn sync_directory(directory: &Path) -> Result<()> {
    let _ = directory;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use lodestar_ann_index::Value;

    fn metadata(pairs: &[(&str, Value)]) -> Metadata {
        pairs
            .iter()
            .map(|(key, value)| ((*key).to_string(), value.clone()))
            .collect()
    }

    #[test]
    fn records_round_trip_through_the_log() {
        let directory = tempfile::tempdir().unwrap();
        let mut store = MetadataStore::open(directory.path()).unwrap();
        assert!(store.is_empty());
        store
            .set(
                1,
                metadata(&[("category", "news".into()), ("score", 0.9f64.into())]),
            )
            .unwrap();
        store
            .set(2, metadata(&[("category", "blog".into())]))
            .unwrap();
        assert_eq!(store.len(), 2);

        let reopened = MetadataStore::open(directory.path()).unwrap();
        assert_eq!(reopened.len(), 2);
        assert_eq!(
            reopened.get(1).unwrap().get("category"),
            Some(&Value::Str("news".to_string()))
        );
    }

    #[test]
    fn clear_removes_entries_and_survives_a_reopen() {
        let directory = tempfile::tempdir().unwrap();
        let mut store = MetadataStore::open(directory.path()).unwrap();
        store.set(1, metadata(&[("a", 1f64.into())])).unwrap();
        store.set(2, metadata(&[("a", 2f64.into())])).unwrap();
        assert_eq!(store.clear(&[1, 9]).unwrap(), 1);
        assert_eq!(store.clear(&[1]).unwrap(), 0);

        let reopened = MetadataStore::open(directory.path()).unwrap();
        assert_eq!(reopened.len(), 1);
        assert!(reopened.get(1).is_none());
    }

    #[test]
    fn a_torn_final_record_is_repaired_not_reported() {
        let directory = tempfile::tempdir().unwrap();
        let mut store = MetadataStore::open(directory.path()).unwrap();
        store.set(1, metadata(&[("a", 1f64.into())])).unwrap();
        store.set(2, metadata(&[("a", 2f64.into())])).unwrap();
        let path = store.path().to_path_buf();
        drop(store);
        // Simulate a crash in the middle of appending a third record.
        let mut bytes = std::fs::read(&path).unwrap();
        bytes.extend_from_slice(br#"{"id":3,"metada"#);
        std::fs::write(&path, &bytes).unwrap();

        let repaired = MetadataStore::open(directory.path()).unwrap();
        assert_eq!(repaired.len(), 2);
        let text = std::fs::read_to_string(&path).unwrap();
        assert!(!text.contains("metada\""), "torn tail should be gone");
    }

    #[test]
    fn a_damaged_record_before_the_tail_is_corruption() {
        let directory = tempfile::tempdir().unwrap();
        let mut store = MetadataStore::open(directory.path()).unwrap();
        store.set(1, metadata(&[("a", 1f64.into())])).unwrap();
        let path = store.path().to_path_buf();
        drop(store);
        let mut bytes = b"not a record\n".to_vec();
        bytes.extend_from_slice(&std::fs::read(&path).unwrap());
        std::fs::write(&path, &bytes).unwrap();

        let error = MetadataStore::open(directory.path()).unwrap_err();
        assert!(matches!(error, Error::Corrupt { .. }), "{error}");
    }

    #[test]
    fn rewriting_bounds_the_log() {
        let directory = tempfile::tempdir().unwrap();
        let mut store = MetadataStore::open(directory.path()).unwrap();
        for round in 0..64u64 {
            store
                .set(1, metadata(&[("a", (round as f64).into())]))
                .unwrap();
        }
        assert_eq!(store.len(), 1);
        assert!(store.appended > COMPACT_MIN_RECORDS / 64, "appended grows");
        store.rewrite().unwrap();
        let text = std::fs::read_to_string(store.path()).unwrap();
        assert_eq!(text.lines().count(), 1);
        let reopened = MetadataStore::open(directory.path()).unwrap();
        assert_eq!(reopened.get(1).unwrap().get("a"), Some(&Value::Num(63.0)));
    }

    #[test]
    fn matching_applies_the_filter_language() {
        let directory = tempfile::tempdir().unwrap();
        let mut store = MetadataStore::open(directory.path()).unwrap();
        store
            .set(
                1,
                metadata(&[("category", "news".into()), ("score", 0.9f64.into())]),
            )
            .unwrap();
        store
            .set(
                2,
                metadata(&[("category", "blog".into()), ("score", 0.4f64.into())]),
            )
            .unwrap();
        store.set(3, metadata(&[("score", 0.8f64.into())])).unwrap();

        let news = lodestar_ann_index::filter::parse("category == news").unwrap();
        let matched = store.matching(&news);
        assert_eq!(matched, HashSet::from([1]));

        let high = lodestar_ann_index::filter::parse("score >= 0.8").unwrap();
        assert_eq!(store.matching(&high), HashSet::from([1, 3]));
    }

    #[test]
    fn verify_detects_a_log_that_disagrees_with_memory() {
        let directory = tempfile::tempdir().unwrap();
        let mut store = MetadataStore::open(directory.path()).unwrap();
        store.set(1, metadata(&[("a", 1f64.into())])).unwrap();
        store.verify().unwrap();
        // A record appended behind the store's back is a real disagreement.
        let path = store.path().to_path_buf();
        let mut bytes = std::fs::read(&path).unwrap();
        bytes.extend_from_slice(b"{\"id\":2,\"metadata\":{\"a\":2.0}}\n");
        std::fs::write(&path, bytes).unwrap();
        let error = store.verify().unwrap_err();
        assert!(matches!(error, Error::Corrupt { .. }), "{error}");
    }
}
