//! # lodestar-ann-index
//!
//! Approximate-nearest-neighbour indexes built on top of `lodestar-ann-core`:
//!
//! * [`hnsw`] — Hierarchical Navigable Small World graphs. The default index:
//!   excellent recall-per-byte, incremental inserts, no training step.
//! * [`ivfpq`] — Inverted-file indexes with flat or product-quantized residues.
//!   Compact and fast to scan when the collection is large and the working set
//!   does not fit in the page cache.
//! * [`filter`] — A small metadata query language (`category == "books" and
//!   year > 2019`) compiled to a bitset so that traversal can skip
//!   non-matching nodes without a callback into user code.
//! * [`graph`] — The one best-first traversal shared by every graph-based
//!   backend, expressed over [`graph::GraphView`] so that in-memory and
//!   memory-mapped indexes execute identical search code.
//!
//! ## Choosing an index
//!
//! | Situation | Use |
//! | --- | --- |
//! | Up to a few million vectors, recall matters most | [`hnsw::Hnsw`] |
//! | Tens of millions of vectors, memory is the constraint | [`ivfpq::IvfPq`] |
//! | Exact ground truth for tests and gates | `lodestar_ann_core::brute` |
//!
//! All types are deterministic: the same seed, data and configuration produce
//! the same graph, and therefore identical results across runs and platforms.
//!
//! ```
//! use lodestar_ann_core::Metric;
//! use lodestar_ann_index::hnsw::{Hnsw, HnswConfig};
//!
//! let mut index = Hnsw::new(4, Metric::L2, HnswConfig::default()).unwrap();
//! index
//!     .insert_batch(&[0, 1, 2], &[0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 5.0, 5.0, 5.0, 5.0])
//!     .unwrap();
//! let hits = index.search(&[0.1, 0.1, 0.1, 0.1], 1).unwrap();
//! assert_eq!(hits[0].id, 0);
//! ```

#![forbid(unsafe_op_in_unsafe_fn)]
#![warn(missing_docs)]
#![warn(missing_debug_implementations)]
#![warn(clippy::all)]

pub mod filter;
pub mod graph;
pub mod hnsw;
pub mod ivfpq;

pub use filter::{Expr, Metadata, Value};
pub use graph::{GraphView, SearchScratch};
pub use hnsw::{Hnsw, HnswConfig, HnswStats};
pub use ivfpq::{IvfPq, IvfPqConfig, IvfPqStats};
