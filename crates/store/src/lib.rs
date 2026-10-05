//! Durable storage for Lodestar indexes.
//!
//! An index that cannot be reopened is a demo, not a component, so persistence
//! is a first-class part of Lodestar rather than a serialisation afterthought.
//! The planned contents of this crate:
//!
//! * [`segment`] — an append-only file per index, laid out so that the vector
//!   block can be mapped read-only and searched without copying.
//! * [`wal`] — a write-ahead log of inserts and deletes, replayed on open.
//! * [`manifest`] — a small JSON document committed atomically by rename, which
//!   names the live segments and therefore defines what "open" means.
//! * [`store`] — the public handle: open, upsert, delete, search, compact.
//!
//! Crash consistency is the point of the design: a `SIGKILL` in the middle of a
//! write must leave an index that either has the write or does not, and that
//! invariant is covered by a test that kills a child process mid-write.
//!
//! This module is a placeholder: the crate currently contains no implementation,
//! and its `Cargo.toml` declares the dependencies the real code needs. It exists
//! so that the workspace resolves and builds while the storage layer is written.

#![forbid(unsafe_op_in_unsafe_fn)]
#![warn(missing_docs)]
#![warn(clippy::all)]
