//! The write-ahead log.
//!
//! Every mutation that is not yet part of a sealed segment is appended here
//! before it is acknowledged, which is what makes an acknowledged write survive
//! a `SIGKILL`. The log is replayed on open, then truncated once the tail it
//! describes has been sealed into a segment.
//!
//! # Record framing
//!
//! ```text
//! magic      u32   b"LODW"
//! tag        u8    1 = upsert, 2 = delete
//! flags      u8    0
//! reserved   u16   0
//! id         u64
//! body_len   u32   bytes of body that follow
//! body       ...   upsert: dim f32 values, little-endian
//! crc        u32   crc32 of everything above
//! ```
//!
//! # Torn tails versus corruption
//!
//! A crash can leave a partially written record at the end of the file, and the
//! two cases must be handled differently:
//!
//! * A record that is *incomplete* (a truncated body, or a header area that was
//!   never written) is a torn tail. It is truncated away and the log is used up
//!   to that point. Losing it is safe: the write it describes was never
//!   acknowledged, because acknowledgement happens after the flush.
//! * A record that is *complete but fails its checksum* is corruption. It is
//!   reported as an error rather than truncated, because silently dropping it
//!   would drop an acknowledged write.

use std::fs::{File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};

use bytemuck::cast_slice;

use crate::error::{Error, Result};

/// Magic number at the start of every record: `LODW` in little-endian order.
pub const RECORD_MAGIC: u32 = u32::from_le_bytes(*b"LODW");
/// Size of a record header, checksum excluded.
pub const RECORD_HEADER_LEN: usize = 20;
/// Size of the trailing checksum.
pub const RECORD_CRC_LEN: usize = 4;
/// Tag byte for an upsert.
pub const TAG_UPSERT: u8 = 1;
/// Tag byte for a delete.
pub const TAG_DELETE: u8 = 2;

/// One replayed log entry.
#[derive(Debug, Clone, PartialEq)]
pub enum WalOp {
    /// Insert or replace a vector.
    Upsert(Vec<f32>),
    /// Remove a vector.
    Delete,
}

/// A log entry: an operation and the id it applies to.
#[derive(Debug, Clone, PartialEq)]
pub struct WalEntry {
    /// External id.
    pub id: u64,
    /// Operation.
    pub op: WalOp,
}

/// An append-only log of mutations.
#[derive(Debug)]
pub struct Wal {
    path: PathBuf,
    file: File,
    len: u64,
    records: u64,
    unsynced: u64,
}

impl Wal {
    /// Opens (creating if needed) the log at `path` for appending.
    ///
    /// # Errors
    ///
    /// [`Error::Io`] if the file cannot be opened.
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        let file = OpenOptions::new()
            .create(true)
            .read(true)
            .append(true)
            .open(&path)
            .map_err(|source| Error::Io {
                path: path.clone(),
                source,
            })?;
        let len = file
            .metadata()
            .map_err(|source| Error::Io {
                path: path.clone(),
                source,
            })?
            .len();
        Ok(Self {
            path,
            file,
            len,
            records: 0,
            unsynced: 0,
        })
    }

    /// Path of the log file.
    #[must_use]
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Bytes currently in the log.
    #[must_use]
    pub fn len(&self) -> u64 {
        self.len
    }

    /// Whether the log is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Number of records appended by this handle.
    #[must_use]
    pub fn records(&self) -> u64 {
        self.records
    }

    /// Appends an upsert.
    ///
    /// # Errors
    ///
    /// [`Error::Io`] if the record cannot be written.
    pub fn append_upsert(&mut self, id: u64, vector: &[f32]) -> Result<()> {
        let body = cast_slice::<f32, u8>(vector);
        self.append(id, TAG_UPSERT, body)
    }

    /// Appends a delete.
    ///
    /// # Errors
    ///
    /// [`Error::Io`] if the record cannot be written.
    pub fn append_delete(&mut self, id: u64) -> Result<()> {
        self.append(id, TAG_DELETE, &[])
    }

    fn append(&mut self, id: u64, tag: u8, body: &[u8]) -> Result<()> {
        let mut record = Vec::with_capacity(RECORD_HEADER_LEN + body.len() + RECORD_CRC_LEN);
        record.extend_from_slice(&RECORD_MAGIC.to_le_bytes());
        record.push(tag);
        record.push(0);
        record.extend_from_slice(&0u16.to_le_bytes());
        record.extend_from_slice(&id.to_le_bytes());
        record.extend_from_slice(&(body.len() as u32).to_le_bytes());
        record.extend_from_slice(body);
        let crc = crate::segment::crc32(&record);
        record.extend_from_slice(&crc.to_le_bytes());
        self.file.write_all(&record).map_err(|source| Error::Io {
            path: self.path.clone(),
            source,
        })?;
        self.len += record.len() as u64;
        self.records += 1;
        self.unsynced += 1;
        Ok(())
    }

    /// Flushes and syncs the log; this is the durability point.
    ///
    /// Batching matters here: a sync per record costs a disk flush per write,
    /// while a sync per batch costs one per batch. Both are correct.
    ///
    /// # Errors
    ///
    /// [`Error::Io`] if the flush or sync fails.
    pub fn sync(&mut self) -> Result<()> {
        if self.unsynced == 0 {
            return Ok(());
        }
        self.file.flush().map_err(|source| Error::Io {
            path: self.path.clone(),
            source,
        })?;
        self.file.sync_data().map_err(|source| Error::Io {
            path: self.path.clone(),
            source,
        })?;
        self.unsynced = 0;
        Ok(())
    }

    /// Empties the log, which is what happens after the tail it described has
    /// been sealed into a segment.
    ///
    /// # Errors
    ///
    /// [`Error::Io`] if the file cannot be truncated.
    pub fn reset(&mut self) -> Result<()> {
        self.file.flush().map_err(|source| Error::Io {
            path: self.path.clone(),
            source,
        })?;
        self.file.set_len(0).map_err(|source| Error::Io {
            path: self.path.clone(),
            source,
        })?;
        self.file.sync_all().map_err(|source| Error::Io {
            path: self.path.clone(),
            source,
        })?;
        self.len = 0;
        self.records = 0;
        self.unsynced = 0;
        Ok(())
    }
}

/// Replays a log file.
///
/// Returns the entries in append order and the number of bytes that formed
/// complete, checksummed records. The caller truncates the file to that length,
/// which is what discards a torn tail.
///
/// # Errors
///
/// * [`Error::Io`] if the file cannot be read.
/// * [`Error::Corrupt`] for a complete record whose checksum fails, or a body
///   length that cannot belong to the log's dimensionality.
pub fn read_all(path: impl AsRef<Path>, dim: usize) -> Result<(Vec<WalEntry>, u64)> {
    let path = path.as_ref();
    let bytes = match std::fs::read(path) {
        Ok(bytes) => bytes,
        Err(source) if source.kind() == std::io::ErrorKind::NotFound => return Ok((Vec::new(), 0)),
        Err(source) => {
            return Err(Error::Io {
                path: path.to_path_buf(),
                source,
            });
        }
    };
    let mut entries = Vec::new();
    let mut cursor = 0usize;
    while cursor < bytes.len() {
        let remaining = bytes.len() - cursor;
        if remaining < RECORD_HEADER_LEN {
            // Not enough for a header: a torn append.
            break;
        }
        let header = &bytes[cursor..cursor + RECORD_HEADER_LEN];
        let magic = u32::from_le_bytes(header[0..4].try_into().expect("four bytes"));
        if magic != RECORD_MAGIC {
            if header.iter().all(|byte| *byte == 0) {
                // A pre-allocated but never written region.
                break;
            }
            return Err(Error::corrupt(
                "write-ahead log",
                format!("no record magic at offset {cursor}"),
            ));
        }
        let tag = header[4];
        let id = u64::from_le_bytes(header[8..16].try_into().expect("eight bytes"));
        let body_len = u32::from_le_bytes(header[16..20].try_into().expect("four bytes")) as usize;
        let record_len = RECORD_HEADER_LEN + body_len + RECORD_CRC_LEN;
        if remaining < record_len {
            // The record started but did not finish: a torn append.
            break;
        }
        let record = &bytes[cursor..cursor + record_len];
        let stored = u32::from_le_bytes(
            record[RECORD_HEADER_LEN + body_len..record_len]
                .try_into()
                .expect("four bytes"),
        );
        if crate::segment::crc32(&record[..RECORD_HEADER_LEN + body_len]) != stored {
            return Err(Error::corrupt(
                "write-ahead log",
                format!("checksum mismatch in the record at offset {cursor}"),
            ));
        }
        let op = match tag {
            TAG_DELETE => {
                if body_len != 0 {
                    return Err(Error::corrupt(
                        "write-ahead log",
                        format!("delete record for id {id} carries {body_len} bytes"),
                    ));
                }
                WalOp::Delete
            }
            TAG_UPSERT => {
                if body_len != dim * 4 {
                    return Err(Error::corrupt(
                        "write-ahead log",
                        format!(
                            "upsert record for id {id} carries {body_len} bytes, expected {}",
                            dim * 4
                        ),
                    ));
                }
                let body = &record[RECORD_HEADER_LEN..RECORD_HEADER_LEN + body_len];
                let values: &[f32] = bytemuck::try_cast_slice(body).map_err(|_| {
                    Error::corrupt(
                        "write-ahead log",
                        format!("upsert body for id {id} is not aligned for f32"),
                    )
                })?;
                WalOp::Upsert(values.to_vec())
            }
            other => {
                return Err(Error::corrupt(
                    "write-ahead log",
                    format!("unknown record tag {other} at offset {cursor}"),
                ));
            }
        };
        entries.push(WalEntry { id, op });
        cursor += record_len;
    }
    Ok((entries, cursor as u64))
}

/// Replays a log and truncates any torn tail, in that order.
///
/// # Errors
///
/// As [`read_all`], plus [`Error::Io`] if the truncation fails.
pub fn replay_and_repair(path: impl AsRef<Path>, dim: usize) -> Result<Vec<WalEntry>> {
    let path = path.as_ref();
    let (entries, valid) = read_all(path, dim)?;
    let actual = std::fs::metadata(path).map(|meta| meta.len()).unwrap_or(0);
    if actual > valid {
        crate::segment::truncate_file(path, valid)?;
    }
    Ok(entries)
}
