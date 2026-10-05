//! Durable storage for Lodestar indexes.
//!
//! An index that cannot be reopened is a demo, not a component, so persistence
//! is part of the engine rather than a serialisation afterthought. The storage
//! layer is deliberately small and boring: an immutable segment format that is
//! mapped read-only, a write-ahead log for the writes that arrive after a
//! segment is sealed, and a manifest that defines what "open" means.
//!
//! * [`segment`] — the on-disk format: a fixed header, aligned blocks, per-level
//!   CSR adjacency and a checksummed footer.
//! * [`mapped`] — the reader: maps a segment and implements `GraphView` over it,
//!   so search runs the same traversal as the in-memory index.
//! * [`wal`] — the write-ahead log of inserts and deletes.
//! * [`manifest`] — the small JSON document that names the live segments and is
//!   committed by an atomic rename.
//! * [`collection`] — the handle that ties them together: open, upsert, delete,
//!   search, flush, verify.
//!
//! # Crash consistency
//!
//! The ordering rules are the whole design:
//!
//! 1. A write-ahead log record is appended and flushed before the write is
//!    acknowledged, so an acknowledged write survives a crash.
//! 2. A segment is written to a temporary file, flushed, and only then renamed
//!    into place, so a segment name always refers to a complete file.
//! 3. The manifest is replaced by writing a temporary file and renaming it, so a
//!    reader sees either the old set of segments or the new one.
//! 4. A segment becomes unreferenced only after the manifest that stopped
//!    naming it is durable, so a crash cannot leave a manifest pointing at a
//!    deleted file.
//!
//! `tests/crash.rs` kills a child process at random points during writes and
//! reopens the collection, which is the only way to find out whether those
//! rules actually hold.

#![forbid(unsafe_op_in_unsafe_fn)]
#![warn(missing_docs)]
#![warn(clippy::all)]

pub mod collection;
pub mod error;
pub mod manifest;
pub mod mapped;
pub mod segment;
pub mod wal;

pub use collection::{Collection, CollectionStats};
pub use error::{Error, Result};
pub use manifest::{Manifest, SegmentRef};
pub use mapped::Segment;
pub use segment::{SegmentInfo, write_segment};
