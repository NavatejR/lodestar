//! Inverted-file index with flat or product-quantized postings.
//!
//! An IVF index trades a linear scan for two smaller scans. Training learns
//! `nlist` coarse centroids; every vector is filed under its nearest centroid;
//! a query ranks the centroids, scans only the `nprobe` closest lists, and
//! keeps the best `k` candidates it sees. The saving is roughly
//! `nprobe / nlist` of the work, at the cost of a recall loss that shrinks as
//! `nprobe` grows.
//!
//! Two storage modes share that structure:
//!
//! * **Flat** ([`IvfPqConfig::pq_m`]` == 0`): each posting stores the vector
//!   itself, so distances inside the probed lists are exact. With
//!   `nprobe == nlist` the search is exhaustive and *identically* matches brute
//!   force, which is what makes this mode a testable reference implementation.
//! * **Product-quantized** ([`IvfPqConfig::pq_m`]` > 0`): each posting stores an
//!   `m`-byte code. Distances are computed from a lookup table (asymmetric
//!   distance computation), so both the scan and the memory footprint shrink by
//!   a factor of about `4 * dim / m`.
//!
//! With [`IvfPqConfig::residual`] enabled, PQ encodes `vector - centroid(list)`
//! rather than the raw vector. Residuals are much smaller than the vectors they
//! came from, so a codebook spends its centroids on local detail instead of
//! re-learning global structure that the coarse quantizer already captured.
//! This is the standard Faiss configuration for L2 data, and the one
//! [`IvfPqConfig::with_pq`] selects by default.
//!
//! # Metrics
//!
//! Flat postings support every metric. PQ postings are checked against the data
//! rather than assumed:
//!
//! * `L2` — direct.
//! * `Cosine` — vectors are normalised on insert, and for unit vectors squared
//!   euclidean distance is a strictly monotone function of cosine distance
//!   (`||a - b||² = 2 - 2·cos`), so the same codes rank correctly. Reported
//!   distances are converted back to the cosine scale.
//! * `InnerProduct` — **rejected** with [`Error::InvalidParameter`]. ADC scores
//!   `||q - c||²`, which contains a `||c||²` term that is not constant across
//!   candidates, so the ranking would silently differ from the metric the caller
//!   asked for. Use flat postings or an HNSW index for inner product.
//!
//! # Persistence
//!
//! There is deliberately no `parts`/`from_parts` pair here. Durable segments in
//! this project are built around the HNSW graph layout, and IVF indexes are
//! trained and served from the vector source in one process. Adding a second
//! segment layout before the first one has a crash-consistency test would mean
//! maintaining two unverified formats, so the format wait is deliberate.

use std::collections::{BinaryHeap, HashMap};

use lodestar_ann_core::brute::RowFilter;
use lodestar_ann_core::pq::{MAX_CENTROIDS, ProductQuantizer, kmeans};
use lodestar_ann_core::{Candidate, Error, Metric, Result, Rng};
use serde::{Deserialize, Serialize};

/// Seed domain separator for the coarse quantizer.
const CENTROID_SEED: u64 = 0xC0DE_0FF0_1234_0001;
/// Seed domain separator for the product quantizer.
const PQ_SEED: u64 = 0x00B0_0B5E_0000_0001;

/// Configuration for an [`IvfPq`] index.
///
/// [`IvfPqConfig::default`] describes a flat index, which is valid for any
/// dimensionality. [`IvfPqConfig::with_pq`] switches on product quantization,
/// and with it residual encoding, the recommended PQ configuration.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct IvfPqConfig {
    /// Number of coarse centroids, at most
    /// [`MAX_CENTROIDS`].
    ///
    /// The literature calls this `nlist`. As a rule of thumb it should be
    /// between `sqrt(n)` and `16 * sqrt(n)` for `n` vectors.
    pub nlist: usize,
    /// How many lists a query scans by default.
    ///
    /// Larger means better recall and more work. Values above `nlist` are
    /// clamped rather than rejected, so `usize::MAX` is a valid way to ask for
    /// an exhaustive scan.
    pub nprobe: usize,
    /// Number of PQ subspaces, or `0` for exact flat postings.
    ///
    /// Must divide the dimensionality. Each code is `pq_m` bytes, so
    /// `pq_m = 8` on 128-dimensional data is a 64x reduction against `f32`.
    pub pq_m: usize,
    /// Centroids per PQ subspace, at most 256.
    ///
    /// 256 is the usual choice: it uses the full range of a byte and keeps ADC
    /// tables small.
    pub pq_ksub: usize,
    /// Lloyd iterations when training the coarse quantizer.
    pub kmeans_iters: usize,
    /// Lloyd iterations per PQ subspace.
    pub pq_iters: usize,
    /// Cap on the number of vectors used for training.
    ///
    /// Training k-means on a whole corpus costs more than it buys. Sampling is
    /// deterministic (seeded shuffle), so the resulting index is reproducible.
    /// Zero means "use everything". Peak training memory is roughly
    /// `train_limit * dim * 4` bytes for the coarse step, and another
    /// `train_limit * dim * 4` when PQ residuals are trained.
    pub train_limit: usize,
    /// Master seed. Every derived seed comes from it, so changing it changes
    /// the whole index.
    pub seed: u64,
    /// Whether PQ encodes residuals instead of raw vectors.
    ///
    /// Rejected when `pq_m == 0`, so that a configuration cannot silently mean
    /// something other than what it says. [`IvfPqConfig::with_pq`] turns this on
    /// and [`IvfPqConfig::without_residuals`] turns it off.
    pub residual: bool,
}

impl Default for IvfPqConfig {
    fn default() -> Self {
        Self {
            nlist: 256,
            nprobe: 16,
            pq_m: 0,
            pq_ksub: 256,
            kmeans_iters: 25,
            pq_iters: 25,
            train_limit: 100_000,
            seed: 0x10DA_7A5E_ED00_0001,
            // Flat by default: a PQ configuration depends on the
            // dimensionality, so it cannot be chosen before the data is known.
            residual: false,
        }
    }
}

impl IvfPqConfig {
    /// Sets [`IvfPqConfig::nlist`].
    #[must_use]
    pub const fn with_nlist(mut self, nlist: usize) -> Self {
        self.nlist = nlist;
        self
    }

    /// Sets [`IvfPqConfig::nprobe`].
    #[must_use]
    pub const fn with_nprobe(mut self, nprobe: usize) -> Self {
        self.nprobe = nprobe;
        self
    }

    /// Switches to product-quantized postings with `m` subspaces and `ksub`
    /// centroids per subspace, and enables residual encoding.
    ///
    /// Call [`IvfPqConfig::without_residuals`] afterwards to quantize the raw
    /// vectors instead.
    #[must_use]
    pub const fn with_pq(mut self, m: usize, ksub: usize) -> Self {
        self.pq_m = m;
        self.pq_ksub = ksub;
        self.residual = true;
        self
    }

    /// Uses exact flat postings.
    #[must_use]
    pub const fn flat(mut self) -> Self {
        self.pq_m = 0;
        self.residual = false;
        self
    }

    /// Sets [`IvfPqConfig::seed`].
    #[must_use]
    pub const fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// Encodes raw vectors instead of residuals.
    #[must_use]
    pub const fn without_residuals(mut self) -> Self {
        self.residual = false;
        self
    }

    /// Checks the parts of the configuration that do not depend on the data.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidParameter`] for any out-of-range or inconsistent field.
    pub fn validate(&self, dim: usize, metric: Metric) -> Result<()> {
        if dim == 0 {
            return Err(Error::InvalidParameter {
                name: "dim",
                reason: "must be non-zero".to_string(),
            });
        }
        if self.nlist == 0 || self.nlist > MAX_CENTROIDS {
            return Err(Error::InvalidParameter {
                name: "ivf.nlist",
                reason: format!("{} must be in 1..={MAX_CENTROIDS}", self.nlist),
            });
        }
        if self.nprobe == 0 {
            return Err(Error::InvalidParameter {
                name: "ivf.nprobe",
                reason: "must be at least 1".to_string(),
            });
        }
        if self.kmeans_iters == 0 {
            return Err(Error::InvalidParameter {
                name: "ivf.kmeans_iters",
                reason: "must be at least 1".to_string(),
            });
        }
        if self.pq_m == 0 {
            if self.residual {
                return Err(Error::InvalidParameter {
                    name: "ivf.residual",
                    reason: "requires pq_m > 0; flat postings are never residuals".to_string(),
                });
            }
            return Ok(());
        }
        if dim % self.pq_m != 0 {
            return Err(Error::InvalidParameter {
                name: "ivf.pq_m",
                reason: format!("dim={dim} must be a multiple of pq_m={}", self.pq_m),
            });
        }
        if self.pq_ksub == 0 || self.pq_ksub > MAX_CENTROIDS {
            return Err(Error::InvalidParameter {
                name: "ivf.pq_ksub",
                reason: format!("{} must be in 1..={MAX_CENTROIDS}", self.pq_ksub),
            });
        }
        if self.pq_iters == 0 {
            return Err(Error::InvalidParameter {
                name: "ivf.pq_iters",
                reason: "must be at least 1".to_string(),
            });
        }
        if metric == Metric::InnerProduct {
            return Err(Error::InvalidParameter {
                name: "ivf.pq_m",
                reason: "product quantization cannot rank inner product; use flat postings \
                         (pq_m = 0) or an HNSW index"
                    .to_string(),
            });
        }
        Ok(())
    }

    /// Bytes per stored vector in each mode, given the dimensionality.
    ///
    /// Flat is `4 * dim`; PQ is `pq_m`. Useful for capacity planning and for
    /// the memory figures reported by [`IvfPq::memory_bytes`].
    #[must_use]
    pub const fn bytes_per_vector(&self, dim: usize) -> usize {
        if self.pq_m == 0 { dim * 4 } else { self.pq_m }
    }
}

/// Counters describing the shape of a trained index.
#[derive(Debug, Clone, PartialEq)]
pub struct IvfPqStats {
    /// Dimensionality.
    pub dim: usize,
    /// Number of coarse centroids.
    pub nlist: usize,
    /// Lists scanned per query by default.
    pub nprobe: usize,
    /// Number of live vectors.
    pub live: usize,
    /// Number of tombstoned vectors waiting for [`IvfPq::compact`].
    pub deleted: usize,
    /// Lists with no live members.
    pub empty_lists: usize,
    /// Smallest live count across the lists, counting empty ones as zero.
    pub list_len_min: usize,
    /// Largest live count across the lists.
    pub list_len_max: usize,
    /// Mean live count across the lists.
    pub list_len_mean: f64,
    /// Bytes per stored vector.
    pub bytes_per_vector: usize,
    /// Bytes occupied by vectors or codes, excluding ids and lists.
    pub posting_bytes: usize,
}

/// An inverted-file index with flat or product-quantized postings.
///
/// Construct one with [`IvfPq::train`], which doubles as the builder: the coarse
/// quantizer cannot exist without data, so there is no untrained state to
/// represent.
///
/// ```
/// use lodestar_ann_core::Metric;
/// use lodestar_ann_index::ivfpq::{IvfPq, IvfPqConfig};
///
/// // Two well-separated triples; nlist == 2 clusters them cleanly.
/// let data = [
///     0.0f32, 0.0, 0.1, 0.0, 0.0, 0.1, // cluster A: ids 0, 1, 2
///     5.0, 5.0, 5.1, 5.0, 5.0, 5.1, // cluster B: ids 3, 4, 5
/// ];
/// let config = IvfPqConfig::default().with_nlist(2).with_nprobe(1).flat();
/// let mut index = IvfPq::train(&data, 6, 2, Metric::L2, config).unwrap();
/// index.insert(6, &[5.0, 5.05]).unwrap();
///
/// let hits = index.search(&[0.05, 0.0], 1).unwrap();
/// assert!(hits[0].id < 3);
/// assert_eq!(index.len(), 7);
/// ```
#[derive(Debug, Clone)]
pub struct IvfPq {
    dim: usize,
    metric: Metric,
    config: IvfPqConfig,
    /// `nlist * dim`, laid out one centroid per row.
    centroids: Vec<f32>,
    /// Present exactly when `config.pq_m > 0`.
    quantizer: Option<ProductQuantizer>,
    /// External id per slot, in insertion order.
    ids: Vec<u64>,
    /// External id to slot.
    slots: HashMap<u64, u32>,
    /// Tombstone flag per slot.
    live: Vec<bool>,
    /// Coarse list owning each slot.
    assignment: Vec<u32>,
    /// Slot indices per list.
    lists: Vec<Vec<u32>>,
    /// `count * dim` raw vectors; empty in PQ mode.
    flat: Vec<f32>,
    /// `count * pq_m` codes; empty in flat mode.
    codes: Vec<u8>,
    /// Live vectors, so that `len` does not have to scan the tombstones.
    live_count: usize,
}

impl IvfPq {
    /// Trains an index on `count` vectors laid out contiguously, then inserts
    /// all of them.
    ///
    /// Training never mutates the caller's buffer: vectors are copied through a
    /// prepared row buffer, so `Cosine` indexes normalise their copies and the
    /// source keeps whatever values it had.
    ///
    /// # Errors
    ///
    /// * [`Error::DimensionMismatch`] if `vectors.len() != count * dim`.
    /// * [`Error::InvalidParameter`] for a rejected configuration, including
    ///   product quantization with [`Metric::InnerProduct`].
    /// * [`Error::InsufficientTrainingData`] if fewer vectors are available
    ///   than `nlist` (or than `pq_ksub`, for the PQ step).
    pub fn train(
        vectors: &[f32],
        count: usize,
        dim: usize,
        metric: Metric,
        config: IvfPqConfig,
    ) -> Result<Self> {
        config.validate(dim, metric)?;
        if dim != 0 && vectors.len() != count * dim {
            return Err(Error::DimensionMismatch {
                expected: count * dim,
                actual: vectors.len(),
            });
        }

        let limit = if config.train_limit == 0 {
            count
        } else {
            config.train_limit.min(count)
        };
        if limit < config.nlist {
            return Err(Error::InsufficientTrainingData {
                needed: config.nlist,
                got: limit,
            });
        }

        let rows = sample_rows(count, limit, config.seed);
        let sample = prepared_sample(vectors, &rows, dim, metric);
        let centroids = kmeans(
            &sample,
            limit,
            dim,
            config.nlist,
            config.kmeans_iters,
            config.seed ^ CENTROID_SEED,
        )?;

        let quantizer = if config.pq_m > 0 {
            let mut residuals = sample;
            if config.residual {
                subtract_centroids(&mut residuals, limit, dim, &centroids, config.nlist);
            }
            Some(ProductQuantizer::train(
                &residuals,
                limit,
                dim,
                config.pq_m,
                config.pq_ksub,
                config.pq_iters,
                config.seed ^ PQ_SEED,
                config.train_limit,
            )?)
        } else {
            None
        };

        let mut index = Self::with_parts(
            dim,
            metric,
            config,
            centroids,
            quantizer,
            Vec::new(),
            Vec::new(),
            Vec::new(),
            Vec::new(),
        );
        index.insert_batch_all(vectors, count)?;
        Ok(index)
    }

    /// Empty index sharing the given trained state.
    ///
    /// Only used internally: an index without its trained centroids cannot
    /// assign anything to a list, and an untrained index is useless.
    #[allow(clippy::too_many_arguments)]
    fn with_parts(
        dim: usize,
        metric: Metric,
        config: IvfPqConfig,
        centroids: Vec<f32>,
        quantizer: Option<ProductQuantizer>,
        ids: Vec<u64>,
        live: Vec<bool>,
        assignment: Vec<u32>,
        lists: Vec<Vec<u32>>,
    ) -> Self {
        let mut slots = HashMap::with_capacity(ids.len());
        for (slot, &id) in ids.iter().enumerate() {
            slots.entry(id).or_insert(slot as u32);
        }
        let mut index = Self {
            dim,
            metric,
            config,
            centroids,
            quantizer,
            ids,
            slots,
            live,
            assignment,
            lists,
            flat: Vec::new(),
            codes: Vec::new(),
            live_count: 0,
        };
        index.rebuild_lists();
        index
    }

    /// Recomputes list membership from `assignment` and `live`.
    fn rebuild_lists(&mut self) {
        self.lists = vec![Vec::new(); self.config.nlist];
        self.live_count = 0;
        for slot in 0..self.ids.len() {
            if !self.live[slot] {
                continue;
            }
            self.live_count += 1;
            let list = self.assignment[slot] as usize;
            self.lists[list].push(slot as u32);
        }
    }

    /// Dimensionality of the indexed vectors.
    #[must_use]
    pub const fn dim(&self) -> usize {
        self.dim
    }

    /// Metric used for ranking.
    #[must_use]
    pub const fn metric(&self) -> Metric {
        self.metric
    }

    /// Configuration this index was trained with.
    #[must_use]
    pub const fn config(&self) -> &IvfPqConfig {
        &self.config
    }

    /// The product quantizer, if postings are quantized.
    #[must_use]
    pub const fn quantizer(&self) -> Option<&ProductQuantizer> {
        self.quantizer.as_ref()
    }

    /// Coarse centroids, laid out `nlist x dim`.
    #[must_use]
    pub fn centroids(&self) -> &[f32] {
        &self.centroids
    }

    /// Number of live vectors.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.live_count
    }

    /// Whether the index holds no live vectors.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.live_count == 0
    }

    /// Total number of slots, including tombstoned ones.
    #[must_use]
    pub fn slot_count(&self) -> usize {
        self.ids.len()
    }

    /// Whether `id` is present and live.
    #[must_use]
    pub fn contains(&self, id: u64) -> bool {
        self.slots
            .get(&id)
            .is_some_and(|&slot| self.live[slot as usize])
    }

    /// Slot backing `id`, live or not.
    ///
    /// Slot indices are stable across inserts and deletes, which makes them the
    /// right currency for row predicates: a caller can compile a bitset once and
    /// have it stay valid while the index is mutated.
    #[must_use]
    pub fn slot_of(&self, id: u64) -> Option<usize> {
        self.slots.get(&id).map(|&slot| slot as usize)
    }

    /// External id stored in `slot`.
    #[must_use]
    pub fn id_at(&self, slot: usize) -> Option<u64> {
        self.ids.get(slot).copied()
    }

    /// Coarse list owning `slot`.
    #[must_use]
    pub fn list_of_slot(&self, slot: usize) -> Option<usize> {
        self.assignment.get(slot).map(|&list| list as usize)
    }

    /// Number of live members in `list`.
    #[must_use]
    pub fn list_len(&self, list: usize) -> usize {
        self.lists.get(list).map_or(0, |members| {
            members.iter().filter(|&&s| self.live[s as usize]).count()
        })
    }

    /// Inserts or replaces one vector.
    ///
    /// Replacing an id keeps its slot, so any compiled row predicate stays
    /// valid; only list membership changes.
    ///
    /// # Errors
    ///
    /// [`Error::DimensionMismatch`] if `vector.len() != dim`.
    pub fn insert(&mut self, id: u64, vector: &[f32]) -> Result<()> {
        let mut row = Vec::with_capacity(self.dim);
        let mut code = Vec::with_capacity(self.config.pq_m);
        self.insert_prepared(id, vector, &mut row, &mut code)
    }

    /// Inserts many vectors from a contiguous `count x dim` matrix.
    ///
    /// # Errors
    ///
    /// * [`Error::DimensionMismatch`] if `ids.len() != count` or the vector
    ///   buffer is not `count x dim`.
    pub fn insert_batch(&mut self, ids: &[u64], vectors: &[f32]) -> Result<usize> {
        if self.dim != 0 && vectors.len() % self.dim != 0 {
            return Err(Error::DimensionMismatch {
                expected: (vectors.len() / self.dim) * self.dim,
                actual: vectors.len(),
            });
        }
        let count = vectors.len() / self.dim.max(1);
        if ids.len() != count {
            return Err(Error::DimensionMismatch {
                expected: count,
                actual: ids.len(),
            });
        }
        let mut row = Vec::with_capacity(self.dim);
        let mut code = Vec::with_capacity(self.config.pq_m);
        for (i, &id) in ids.iter().enumerate() {
            self.insert_prepared(
                id,
                &vectors[i * self.dim..(i + 1) * self.dim],
                &mut row,
                &mut code,
            )?;
        }
        Ok(count)
    }

    /// Inserts a whole corpus that has already been validated by
    /// [`IvfPq::train`].
    fn insert_batch_all(&mut self, vectors: &[f32], count: usize) -> Result<()> {
        let mut row = Vec::with_capacity(self.dim);
        let mut code = Vec::with_capacity(self.config.pq_m);
        for index in 0..count {
            self.insert_prepared(
                index as u64,
                &vectors[index * self.dim..(index + 1) * self.dim],
                &mut row,
                &mut code,
            )?;
        }
        Ok(())
    }

    /// Inserts one row, reusing `row` and `code` as scratch buffers.
    ///
    /// The caller owns both buffers, so encoding never touches the index's own
    /// storage until the final copy. Flat postings are written straight from the
    /// prepared row, with no byte-level round trip.
    fn insert_prepared(
        &mut self,
        id: u64,
        vector: &[f32],
        row: &mut Vec<f32>,
        code: &mut Vec<u8>,
    ) -> Result<()> {
        if vector.len() != self.dim {
            return Err(Error::DimensionMismatch {
                expected: self.dim,
                actual: vector.len(),
            });
        }
        row.clear();
        row.extend_from_slice(vector);
        self.metric.prepare(row);
        let (list, _) = nearest_centroid(row, &self.centroids, self.dim, self.config.nlist);

        if let Some(pq) = self.quantizer.as_ref() {
            code.resize(pq.code_len(), 0);
            if self.config.residual {
                let centroid = &self.centroids[list * self.dim..(list + 1) * self.dim];
                for (value, center) in row.iter_mut().zip(centroid.iter()) {
                    *value -= *center;
                }
            }
            // Shapes were validated when the quantizer was trained.
            let _ = pq.encode_into(row, code);
        }

        let existing = self.slots.get(&id).copied();
        let slot = existing.map_or(self.ids.len(), |slot| slot as usize);

        // Storage first: `slot` is either the next free index or an existing one,
        // and both encodings are fixed-size per slot.
        if self.quantizer.is_some() {
            let end = slot * self.config.pq_m + self.config.pq_m;
            if self.codes.len() < end {
                self.codes.resize(end, 0);
            }
            self.codes[slot * self.config.pq_m..end].copy_from_slice(&code[..self.config.pq_m]);
        } else {
            let end = slot * self.dim + self.dim;
            if self.flat.len() < end {
                self.flat.resize(end, 0.0);
            }
            self.flat[slot * self.dim..end].copy_from_slice(row);
        }

        let Some(previous) = existing.map(|slot| self.assignment[slot as usize] as usize) else {
            self.ids.push(id);
            self.slots.insert(id, slot as u32);
            self.live.push(true);
            self.assignment.push(list as u32);
            self.lists[list].push(slot as u32);
            self.live_count += 1;
            return Ok(());
        };

        // Existing id: the slot is kept (so compiled row predicates stay valid),
        // only its list membership and encoding change.
        if previous != list {
            let members = &mut self.lists[previous];
            if let Some(position) = members.iter().position(|&s| s as usize == slot) {
                members.swap_remove(position);
            }
            self.lists[list].push(slot as u32);
            self.assignment[slot] = list as u32;
        }
        if !self.live[slot] {
            self.live[slot] = true;
            self.live_count += 1;
        }
        Ok(())
    }

    /// The stored vector for `id`, if postings are flat.
    ///
    /// Returns `None` in PQ mode, where the exact vector is not retained, and
    /// for tombstoned ids. Note that the returned vector is *prepared*: cosine
    /// indexes store normalised vectors.
    #[must_use]
    pub fn get(&self, id: u64) -> Option<&[f32]> {
        if self.quantizer.is_some() {
            return None;
        }
        let slot = *self.slots.get(&id)? as usize;
        if !self.live[slot] {
            return None;
        }
        Some(&self.flat[slot * self.dim..(slot + 1) * self.dim])
    }

    /// Marks `id` as deleted, returning whether it was live.
    ///
    /// The vector stays in place as a tombstone so that slot indices remain
    /// stable; space is reclaimed by [`IvfPq::compact`].
    pub fn delete(&mut self, id: u64) -> bool {
        let Some(&slot) = self.slots.get(&id) else {
            return false;
        };
        let slot = slot as usize;
        if !self.live[slot] {
            return false;
        }
        self.live[slot] = false;
        self.live_count -= 1;
        true
    }

    /// Rebuilds the index with tombstones removed.
    ///
    /// Encoded vectors and list assignments are copied verbatim: the trained
    /// state is unchanged, so nothing needs re-encoding and the rebuild is
    /// lossless.
    #[must_use]
    pub fn compact(&self) -> Self {
        let mut out = Self {
            dim: self.dim,
            metric: self.metric,
            config: self.config,
            centroids: self.centroids.clone(),
            quantizer: self.quantizer.clone(),
            ids: Vec::with_capacity(self.live_count),
            slots: HashMap::with_capacity(self.live_count),
            live: Vec::with_capacity(self.live_count),
            assignment: Vec::with_capacity(self.live_count),
            lists: vec![Vec::new(); self.config.nlist],
            flat: Vec::new(),
            codes: Vec::new(),
            live_count: self.live_count,
        };
        for slot in 0..self.ids.len() {
            if !self.live[slot] {
                continue;
            }
            let new_slot = out.ids.len();
            out.ids.push(self.ids[slot]);
            out.slots.insert(self.ids[slot], new_slot as u32);
            out.live.push(true);
            out.assignment.push(self.assignment[slot]);
            out.lists[self.assignment[slot] as usize].push(new_slot as u32);
            out.store_raw(new_slot, self.raw_row(slot));
        }
        out
    }

    /// Raw encoded row for `slot` (flat floats or PQ bytes), as a byte slice
    /// for flat mode and a borrowed code for PQ mode.
    fn raw_row(&self, slot: usize) -> RawRow<'_> {
        if self.quantizer.is_some() {
            RawRow::Code(&self.codes[slot * self.config.pq_m..(slot + 1) * self.config.pq_m])
        } else {
            RawRow::Flat(&self.flat[slot * self.dim..(slot + 1) * self.dim])
        }
    }

    /// Appends a raw row to the tail of the storage arrays.
    fn store_raw(&mut self, slot: usize, row: RawRow<'_>) {
        match row {
            RawRow::Flat(values) => {
                self.flat.resize(slot * self.dim + self.dim, 0.0);
                self.flat[slot * self.dim..(slot + 1) * self.dim].copy_from_slice(values);
            }
            RawRow::Code(code) => {
                self.codes
                    .resize(slot * self.config.pq_m + self.config.pq_m, 0);
                self.codes[slot * self.config.pq_m..(slot + 1) * self.config.pq_m]
                    .copy_from_slice(code);
            }
        }
    }

    /// Searches with the configured [`IvfPqConfig::nprobe`].
    ///
    /// # Errors
    ///
    /// [`Error::DimensionMismatch`] if `query.len() != dim`.
    pub fn search(&self, query: &[f32], k: usize) -> Result<Vec<Candidate>> {
        self.search_with(query, k, self.config.nprobe, None)
    }

    /// Searches with an explicit probe count.
    ///
    /// # Errors
    ///
    /// As [`IvfPq::search`].
    pub fn search_with_probes(
        &self,
        query: &[f32],
        k: usize,
        nprobe: usize,
    ) -> Result<Vec<Candidate>> {
        self.search_with(query, k, nprobe, None)
    }

    /// Searches a subset of the lists, optionally skipping rows.
    ///
    /// The predicate receives a **slot index** in the same space as
    /// [`IvfPq::slot_of`], so a bitset compiled over slots can be applied here
    /// without rebuilding it per query.
    ///
    /// # Errors
    ///
    /// [`Error::DimensionMismatch`] if `query.len() != dim`.
    pub fn search_with(
        &self,
        query: &[f32],
        k: usize,
        nprobe: usize,
        filter: Option<RowFilter<'_>>,
    ) -> Result<Vec<Candidate>> {
        if query.len() != self.dim {
            return Err(Error::DimensionMismatch {
                expected: self.dim,
                actual: query.len(),
            });
        }
        if k == 0 || self.live_count == 0 || self.dim == 0 {
            return Ok(Vec::new());
        }

        let mut prepared = query.to_vec();
        self.metric.prepare(&mut prepared);
        let probes = self.probe_order(&prepared, nprobe);

        let mut heap: BinaryHeap<Candidate> = BinaryHeap::with_capacity(k + 1);
        let mut lut: Vec<f32> = Vec::new();
        let mut adjusted: Vec<f32> = Vec::new();

        for list in probes {
            let members = &self.lists[list];
            match &self.quantizer {
                Some(pq) => {
                    lut.resize(pq.m() * pq.ksub(), 0.0);
                    let query_for_lut = if self.config.residual {
                        adjusted.clear();
                        adjusted.extend_from_slice(&prepared);
                        let centroid = &self.centroids[list * self.dim..(list + 1) * self.dim];
                        for (value, center) in adjusted.iter_mut().zip(centroid.iter()) {
                            *value -= *center;
                        }
                        adjusted.as_slice()
                    } else {
                        prepared.as_slice()
                    };
                    // The quantizer was trained on this dimensionality.
                    let _ = pq.adc_lut(query_for_lut, &mut lut);
                    for &slot in members {
                        let slot = slot as usize;
                        if !self.live[slot] || filter.is_some_and(|allow| !allow(slot)) {
                            continue;
                        }
                        let code = &self.codes[slot * pq.code_len()..(slot + 1) * pq.code_len()];
                        let adc = pq.distance_from_lut(&lut, code);
                        let distance = if self.metric == Metric::Cosine {
                            adc * 0.5
                        } else {
                            adc
                        };
                        push_bounded(
                            &mut heap,
                            Candidate {
                                distance,
                                id: self.ids[slot],
                            },
                            k,
                        );
                    }
                }
                None => {
                    for &slot in members {
                        let slot = slot as usize;
                        if !self.live[slot] || filter.is_some_and(|allow| !allow(slot)) {
                            continue;
                        }
                        let vector = &self.flat[slot * self.dim..(slot + 1) * self.dim];
                        push_bounded(
                            &mut heap,
                            Candidate {
                                distance: self.metric.distance(&prepared, vector),
                                id: self.ids[slot],
                            },
                            k,
                        );
                    }
                }
            }
        }

        let mut out: Vec<Candidate> = heap.into_vec();
        out.sort_unstable();
        Ok(out)
    }

    /// List indices ordered by increasing centroid distance from `query`.
    ///
    /// `nprobe` is clamped to `1..=nlist`.
    fn probe_order(&self, query: &[f32], nprobe: usize) -> Vec<usize> {
        let mut scored: Vec<(f32, usize)> = (0..self.config.nlist)
            .map(|list| {
                let centroid = &self.centroids[list * self.dim..(list + 1) * self.dim];
                (
                    lodestar_ann_core::distance::l2_squared(query, centroid),
                    list,
                )
            })
            .collect();
        scored.sort_unstable_by(|a, b| a.0.total_cmp(&b.0).then_with(|| a.1.cmp(&b.1)));
        let limit = nprobe.clamp(1, self.config.nlist);
        scored.truncate(limit);
        scored.into_iter().map(|(_, list)| list).collect()
    }

    /// Counters describing list shapes and storage sizes.
    #[must_use]
    pub fn stats(&self) -> IvfPqStats {
        let lengths: Vec<usize> = (0..self.config.nlist).map(|l| self.list_len(l)).collect();
        let total: usize = lengths.iter().sum();
        let empty_lists = lengths.iter().filter(|&&len| len == 0).count();
        IvfPqStats {
            dim: self.dim,
            nlist: self.config.nlist,
            nprobe: self.config.nprobe,
            live: self.live_count,
            deleted: self.ids.len() - self.live_count,
            empty_lists,
            list_len_min: lengths.iter().copied().min().unwrap_or(0),
            list_len_max: lengths.iter().copied().max().unwrap_or(0),
            list_len_mean: if lengths.is_empty() {
                0.0
            } else {
                total as f64 / lengths.len() as f64
            },
            bytes_per_vector: self.config.bytes_per_vector(self.dim),
            posting_bytes: self.flat.len() * 4 + self.codes.len(),
        }
    }

    /// Estimated heap footprint, in bytes.
    ///
    /// This is an estimate, not a measured allocation: it counts the containers
    /// an index owns plus a per-slot allowance for the id map, and deliberately
    /// does not include the allocator's own overhead.
    #[must_use]
    pub fn memory_bytes(&self) -> usize {
        let mut bytes = self.centroids.len() * 4;
        bytes += self.quantizer.as_ref().map_or(0, |pq| pq.codebook_bytes());
        bytes += self.ids.len() * std::mem::size_of::<u64>();
        bytes += self.live.len() * std::mem::size_of::<bool>();
        bytes += self.assignment.len() * std::mem::size_of::<u32>();
        bytes += self.flat.len() * 4 + self.codes.len();
        bytes += self.lists.capacity() * std::mem::size_of::<Vec<u32>>();
        for members in &self.lists {
            bytes += members.capacity() * std::mem::size_of::<u32>();
        }
        bytes += self.slots.capacity() * (std::mem::size_of::<u64>() + std::mem::size_of::<u32>());
        bytes
    }

    /// Checks the internal invariants that make search meaningful.
    ///
    /// # Errors
    ///
    /// [`Error::Numeric`] describing the first inconsistency found. A healthy
    /// index always returns `Ok(())`; this exists for tests and for debugging a
    /// persisted index.
    pub fn verify_integrity(&self) -> Result<()> {
        let expected = if self.config.pq_m == 0 {
            self.ids.len() * self.dim
        } else {
            self.ids.len() * self.config.pq_m
        };
        let actual = if self.config.pq_m == 0 {
            self.flat.len()
        } else {
            self.codes.len()
        };
        if actual != expected {
            return Err(Error::Numeric(format!(
                "storage holds {actual} values for {} slots, expected {expected}",
                self.ids.len()
            )));
        }
        if self.centroids.len() != self.config.nlist * self.dim {
            return Err(Error::Numeric(format!(
                "centroids hold {} values, expected {}",
                self.centroids.len(),
                self.config.nlist * self.dim
            )));
        }
        if self.lists.len() != self.config.nlist {
            return Err(Error::Numeric(format!(
                "{} lists for nlist={}",
                self.lists.len(),
                self.config.nlist
            )));
        }
        if self.slots.len() != self.ids.len() {
            return Err(Error::Numeric(format!(
                "id map holds {} entries for {} slots",
                self.slots.len(),
                self.ids.len()
            )));
        }
        let mut seen = vec![false; self.ids.len()];
        let mut live_members = 0usize;
        for (list, members) in self.lists.iter().enumerate() {
            for &slot in members {
                let slot = slot as usize;
                if slot >= self.ids.len() {
                    return Err(Error::Numeric(format!(
                        "list {list} holds out-of-range slot {slot}"
                    )));
                }
                if seen[slot] {
                    return Err(Error::Numeric(format!("slot {slot} appears in two lists")));
                }
                seen[slot] = true;
                // Tombstones stay in their list until compaction; they are
                // skipped by search, not removed from the postings.
                if self.live[slot] {
                    live_members += 1;
                }
                if self.assignment[slot] as usize != list {
                    return Err(Error::Numeric(format!(
                        "slot {slot} is in list {list} but assigned to {}",
                        self.assignment[slot]
                    )));
                }
            }
        }
        if live_members != self.live_count {
            return Err(Error::Numeric(format!(
                "{live_members} live list members for {} live vectors",
                self.live_count
            )));
        }
        for (id, &slot) in &self.slots {
            if self.ids[slot as usize] != *id {
                return Err(Error::Numeric(format!("id {id} maps to slot {slot}")));
            }
        }
        Ok(())
    }
}

/// A borrowed row of encoded storage.
enum RawRow<'a> {
    /// Raw `f32` values.
    Flat(&'a [f32]),
    /// A product-quantization code.
    Code(&'a [u8]),
}

/// Keeps the `k` best candidates seen so far.
fn push_bounded(heap: &mut BinaryHeap<Candidate>, candidate: Candidate, k: usize) {
    if heap.len() < k {
        heap.push(candidate);
        return;
    }
    if let Some(worst) = heap.peek() {
        if candidate < *worst {
            heap.pop();
            heap.push(candidate);
        }
    }
}

/// Nearest coarse centroid to `vector`, and its squared distance.
fn nearest_centroid(vector: &[f32], centroids: &[f32], dim: usize, nlist: usize) -> (usize, f32) {
    let mut best = 0usize;
    let mut best_distance = f32::INFINITY;
    for list in 0..nlist {
        let distance = lodestar_ann_core::distance::l2_squared(
            vector,
            &centroids[list * dim..(list + 1) * dim],
        );
        if distance < best_distance {
            best_distance = distance;
            best = list;
        }
    }
    (best, best_distance)
}

/// Deterministic `limit`-of-`count` row selection, returned in ascending order.
fn sample_rows(count: usize, limit: usize, seed: u64) -> Vec<usize> {
    if limit >= count {
        return (0..count).collect();
    }
    let mut all: Vec<usize> = (0..count).collect();
    let mut rng = Rng::new(seed ^ 0x5A45_1234_ABCD_0002);
    rng.shuffle(&mut all);
    all.truncate(limit);
    all.sort_unstable();
    all
}

/// Copies `rows` out of `vectors` through the metric's preparation step.
fn prepared_sample(vectors: &[f32], rows: &[usize], dim: usize, metric: Metric) -> Vec<f32> {
    let mut out = Vec::with_capacity(rows.len() * dim);
    for &row in rows {
        let start = out.len();
        out.extend_from_slice(&vectors[row * dim..(row + 1) * dim]);
        metric.prepare(&mut out[start..]);
    }
    out
}

/// Replaces every row by its residual against its nearest centroid.
fn subtract_centroids(
    residuals: &mut [f32],
    count: usize,
    dim: usize,
    centroids: &[f32],
    nlist: usize,
) {
    for row in 0..count {
        let values = &mut residuals[row * dim..(row + 1) * dim];
        let (list, _) = nearest_centroid(values, centroids, dim, nlist);
        let centroid = &centroids[list * dim..(list + 1) * dim];
        for (value, center) in values.iter_mut().zip(centroid.iter()) {
            *value -= *center;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use lodestar_ann_core::brute;

    /// Builds `count` points around `clusters` centres on a `[-4, 4]` cube.
    ///
    /// The spread matters: with a tiny noise level every candidate in a cluster
    /// is effectively equidistant from a query, so top-k is an arbitrary choice
    /// among tied points and a recall figure would measure quantization noise
    /// rather than ranking ability. A wide spread makes the true ranking
    /// well-defined, which is what these tests need to be meaningful.
    fn clustered_data(
        count: usize,
        dim: usize,
        clusters: usize,
        noise: f32,
        seed: u64,
    ) -> Vec<f32> {
        let mut rng = Rng::new(seed);
        let mut centres = vec![0.0f32; clusters * dim];
        for value in centres.iter_mut() {
            *value = (rng.next_f64() as f32 - 0.5) * 8.0;
        }
        let mut data = Vec::with_capacity(count * dim);
        for row in 0..count {
            let cluster = row % clusters;
            for d in 0..dim {
                let jitter = (rng.next_f64() as f32 - 0.5) * noise;
                data.push(centres[cluster * dim + d] + jitter);
            }
        }
        data
    }

    fn top_ids(hits: &[Candidate]) -> Vec<u64> {
        hits.iter().map(|hit| hit.id).collect()
    }

    #[test]
    fn flat_index_with_full_probe_matches_brute_force_exactly() {
        let (count, dim) = (400usize, 16usize);
        let data = clustered_data(count, dim, 8, 0.5, 7);
        let config = IvfPqConfig::default().with_nlist(8).with_nprobe(8).flat();
        let index = IvfPq::train(&data, count, dim, Metric::L2, config).unwrap();

        for query_index in 0..10 {
            let query = &data[query_index * dim..(query_index + 1) * dim];
            let truth = brute::knn(&data, count, dim, Metric::L2, query, 10, None);
            let hits = index.search(query, 10).unwrap();
            assert_eq!(top_ids(&hits), top_ids(&truth), "query {query_index}");
        }
    }

    #[test]
    fn recall_grows_with_nprobe() {
        let (count, dim) = (1500usize, 24usize);
        let data = clustered_data(count, dim, 16, 0.5, 11);
        let config = IvfPqConfig::default().with_nlist(16).flat();
        let index = IvfPq::train(&data, count, dim, Metric::L2, config).unwrap();

        let queries = clustered_data(50, dim, 16, 0.5, 12);
        let mut previous = -1.0f64;
        for nprobe in [1usize, 2, 4, 8, 16] {
            let mut hits = Vec::new();
            let mut truth = Vec::new();
            for q in 0..50 {
                let query = &queries[q * dim..(q + 1) * dim];
                hits.push(index.search_with_probes(query, 10, nprobe).unwrap());
                truth.push(brute::knn(&data, count, dim, Metric::L2, query, 10, None));
            }
            let recall = brute::recall(&truth, &hits, 10);
            assert!(
                recall >= previous,
                "recall fell from {previous} to {recall} at nprobe={nprobe}"
            );
            previous = recall;
        }
        assert!(previous >= 0.99, "full probe recall was {previous}");
    }

    #[test]
    fn flat_index_recall_beats_a_single_probe() {
        let (count, dim) = (1500usize, 24usize);
        let data = clustered_data(count, dim, 16, 0.5, 21);
        let config = IvfPqConfig::default().with_nlist(32).flat();
        let index = IvfPq::train(&data, count, dim, Metric::L2, config).unwrap();

        let queries = clustered_data(40, dim, 16, 0.5, 22);
        let truth: Vec<_> = (0..40)
            .map(|q| {
                brute::knn(
                    &data,
                    count,
                    dim,
                    Metric::L2,
                    &queries[q * dim..(q + 1) * dim],
                    10,
                    None,
                )
            })
            .collect();
        let narrow: Vec<_> = (0..40)
            .map(|q| {
                index
                    .search_with_probes(&queries[q * dim..(q + 1) * dim], 10, 1)
                    .unwrap()
            })
            .collect();
        let wide: Vec<_> = (0..40)
            .map(|q| {
                index
                    .search_with_probes(&queries[q * dim..(q + 1) * dim], 10, 8)
                    .unwrap()
            })
            .collect();

        let narrow_recall = brute::recall(&truth, &narrow, 10);
        let wide_recall = brute::recall(&truth, &wide, 10);
        assert!(
            wide_recall > narrow_recall,
            "{wide_recall} vs {narrow_recall}"
        );
        assert!(wide_recall >= 0.95, "recall@10 was {wide_recall}");
    }

    #[test]
    fn pq_shortlist_reranking_recovers_the_true_ranking() {
        let (count, dim) = (1200usize, 32usize);
        let data = clustered_data(count, dim, 12, 0.5, 31);
        let config = IvfPqConfig::default()
            .with_nlist(12)
            .with_nprobe(12)
            .with_pq(4, 32);
        let index = IvfPq::train(&data, count, dim, Metric::L2, config).unwrap();

        let queries = clustered_data(30, dim, 12, 0.5, 32);
        let mut adc_hits = Vec::new();
        let mut reranked = Vec::new();
        let mut truth = Vec::new();
        for q in 0..30 {
            let query = &queries[q * dim..(q + 1) * dim];
            let shortlist = index.search(query, 64).unwrap();
            // Rerank the ADC shortlist with the exact vectors, the standard way
            // to get the best of both: quantization noise only has to keep the
            // right candidates inside the shortlist, not order them.
            let mut exact: Vec<Candidate> = shortlist
                .iter()
                .map(|hit| Candidate {
                    distance: Metric::L2.distance(
                        query,
                        &data[hit.id as usize * dim..(hit.id as usize + 1) * dim],
                    ),
                    id: hit.id,
                })
                .collect();
            exact.sort_unstable();
            exact.truncate(10);
            adc_hits.push(shortlist.into_iter().take(10).collect::<Vec<_>>());
            reranked.push(exact);
            truth.push(brute::knn(&data, count, dim, Metric::L2, query, 10, None));
        }

        let adc_recall = brute::recall(&truth, &adc_hits, 10);
        let reranked_recall = brute::recall(&truth, &reranked, 10);
        assert!(
            reranked_recall >= adc_recall,
            "reranking lowered recall: {reranked_recall} vs {adc_recall}"
        );
        assert!(
            reranked_recall >= 0.95,
            "reranked recall was {reranked_recall}"
        );
    }

    #[test]
    fn residual_encoding_beats_plain_at_equal_size_and_both_rerank_well() {
        // 4 bytes per vector for 32-dimensional data is a 32x compression, so
        // this is close to the worst case for product quantization. Recall is
        // measured two ways: straight from the ADC ranking, and after reranking
        // the 64-candidate shortlist with exact distances.
        //
        // Measured on this dataset and seed, with 4 subspaces of 256 centroids:
        //   residual  adc@10 0.73  reranked 1.00
        //   plain     adc@10 0.49  reranked 0.97
        // Residuals win at equal size, which is why `with_pq` turns them on.
        let (count, dim) = (1200usize, 32usize);
        let data = clustered_data(count, dim, 12, 0.5, 41);
        let queries = clustered_data(30, dim, 12, 0.5, 42);
        let truth: Vec<_> = (0..30)
            .map(|q| {
                brute::knn(
                    &data,
                    count,
                    dim,
                    Metric::L2,
                    &queries[q * dim..(q + 1) * dim],
                    10,
                    None,
                )
            })
            .collect();

        let mut measured = Vec::new();
        for residual in [true, false] {
            let mut config = IvfPqConfig::default()
                .with_nlist(12)
                .with_nprobe(12)
                .with_pq(4, 256)
                .with_seed(41);
            config.residual = residual;
            let index = IvfPq::train(&data, count, dim, Metric::L2, config).unwrap();

            let mut adc = Vec::new();
            let mut reranked = Vec::new();
            for q in 0..30 {
                let query = &queries[q * dim..(q + 1) * dim];
                let shortlist = index.search(query, 64).unwrap();
                let mut exact: Vec<Candidate> = shortlist
                    .iter()
                    .map(|hit| Candidate {
                        distance: Metric::L2.distance(
                            query,
                            &data[hit.id as usize * dim..(hit.id as usize + 1) * dim],
                        ),
                        id: hit.id,
                    })
                    .collect();
                exact.sort_unstable();
                exact.truncate(10);
                adc.push(shortlist.into_iter().take(10).collect::<Vec<_>>());
                reranked.push(exact);
            }

            let adc_recall = brute::recall(&truth, &adc, 10);
            let reranked_recall = brute::recall(&truth, &reranked, 10);
            assert!(
                reranked_recall >= 0.95,
                "residual={residual} reranked recall was {reranked_recall}"
            );
            measured.push((residual, adc_recall));
        }

        let residual_recall = measured[0].1;
        let plain_recall = measured[1].1;
        assert!(
            residual_recall > plain_recall,
            "residuals should rank better at equal size: {residual_recall} vs {plain_recall}"
        );
    }

    #[test]
    fn rejects_inner_product_with_product_quantization() {
        let data = clustered_data(64, 8, 4, 0.5, 3);
        let config = IvfPqConfig::default().with_nlist(4).with_pq(2, 16);
        let err = IvfPq::train(&data, 64, 8, Metric::InnerProduct, config).unwrap_err();
        assert!(err.to_string().contains("inner product"), "{err}");
    }

    #[test]
    fn rejects_inconsistent_configurations() {
        let data = clustered_data(64, 8, 4, 0.5, 4);
        let cases = [
            IvfPqConfig::default().with_nlist(0),
            IvfPqConfig::default().with_nlist(300),
            IvfPqConfig::default().with_nprobe(0),
            IvfPqConfig::default().with_pq(3, 16),
            IvfPqConfig::default().with_pq(4, 0),
            IvfPqConfig {
                residual: true,
                ..IvfPqConfig::default()
            },
        ];
        for config in cases {
            assert!(
                IvfPq::train(&data, 64, 8, Metric::L2, config).is_err(),
                "config {config:?} was accepted"
            );
        }
    }

    #[test]
    fn rejects_shape_and_training_data_mismatches() {
        let data = clustered_data(64, 8, 4, 0.5, 5);
        let short = IvfPqConfig::default().with_nlist(8);
        assert!(matches!(
            IvfPq::train(&data, 65, 8, Metric::L2, short),
            Err(Error::DimensionMismatch { .. })
        ));
        let greedy = IvfPqConfig::default().with_nlist(256);
        assert!(matches!(
            IvfPq::train(&data, 64, 8, Metric::L2, greedy),
            Err(Error::InsufficientTrainingData { .. })
        ));
    }

    #[test]
    fn deletes_hide_vectors_and_compaction_reclaims_them() {
        let (count, dim) = (200usize, 8usize);
        let data = clustered_data(count, dim, 8, 0.5, 6);
        let config = IvfPqConfig::default().with_nlist(8).flat();
        let mut index = IvfPq::train(&data, count, dim, Metric::L2, config).unwrap();

        assert!(index.delete(3));
        assert!(index.delete(150));
        assert!(!index.delete(3));
        assert_eq!(index.len(), count - 2);
        assert_eq!(index.slot_count(), count);
        assert!(!index.contains(3));
        assert!(index.get(3).is_none());
        let hits = index.search(&data[3 * dim..(3 + 1) * dim], 5).unwrap();
        assert!(hits.iter().all(|hit| hit.id != 3));
        index.verify_integrity().unwrap();

        let compacted = index.compact();
        assert_eq!(compacted.len(), count - 2);
        assert_eq!(compacted.slot_count(), count - 2);
        compacted.verify_integrity().unwrap();
        let before = index.search(&data[7 * dim..(7 + 1) * dim], 5).unwrap();
        let after = compacted.search(&data[7 * dim..(7 + 1) * dim], 5).unwrap();
        assert_eq!(top_ids(&before), top_ids(&after));
    }

    #[test]
    fn inserting_an_existing_id_replaces_the_vector() {
        let data = clustered_data(300, 8, 6, 0.5, 61);
        let config = IvfPqConfig::default().with_nlist(6).flat();
        let mut index = IvfPq::train(&data, 300, 8, Metric::L2, config).unwrap();
        let slot = index.slot_of(10).unwrap();
        let before = index.len();

        // Move id 10 onto a completely different vector.
        let replacement = [40.0f32, 40.0, 40.0, 40.0, 40.0, 40.0, 40.0, 40.0];
        index.insert(10, &replacement).unwrap();

        assert_eq!(index.len(), before);
        assert_eq!(index.slot_of(10), Some(slot));
        assert_eq!(
            index.get(10).unwrap(),
            &replacement[..],
            "the stored vector must be the new one"
        );
        let hits = index.search(&replacement, 1).unwrap();
        assert_eq!(hits[0].id, 10);
        assert!(hits[0].distance.abs() < 1e-6);
        index.verify_integrity().unwrap();
    }

    #[test]
    fn filters_restrict_results_to_allowed_slots() {
        let (count, dim) = (400usize, 8usize);
        let data = clustered_data(count, dim, 8, 0.5, 71);
        let config = IvfPqConfig::default().with_nlist(8).flat();
        let index = IvfPq::train(&data, count, dim, Metric::L2, config).unwrap();

        // Only even ids are allowed, spelled as slots so the predicate is a row
        // predicate as documented.
        let allowed: Vec<bool> = (0..count)
            .map(|slot| index.id_at(slot).is_some_and(|id| id % 2 == 0))
            .collect();
        let predicate = |slot: usize| allowed[slot];
        let query = &data[0..dim];
        let hits = index.search_with(query, 5, 8, Some(&predicate)).unwrap();
        assert_eq!(hits.len(), 5);
        assert!(hits.iter().all(|hit| hit.id % 2 == 0));
    }

    #[test]
    fn dimension_mismatches_are_rejected() {
        let data = clustered_data(64, 8, 4, 0.5, 81);
        let config = IvfPqConfig::default().with_nlist(4).flat();
        let mut index = IvfPq::train(&data, 64, 8, Metric::L2, config).unwrap();
        assert!(matches!(
            index.insert(1, &[0.0, 1.0]),
            Err(Error::DimensionMismatch { .. })
        ));
        assert!(matches!(
            index.search(&[0.0, 1.0], 3),
            Err(Error::DimensionMismatch { .. })
        ));
        assert!(matches!(
            index.insert_batch(&[1, 2, 3], &[0.0; 8]),
            Err(Error::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn empty_queries_and_zero_k_return_nothing() {
        let data = clustered_data(64, 8, 4, 0.5, 91);
        let config = IvfPqConfig::default().with_nlist(4).flat();
        let index = IvfPq::train(&data, 64, 8, Metric::L2, config).unwrap();
        assert!(index.search(&data[0..8], 0).unwrap().is_empty());

        let mut empty = index.compact();
        for id in 0..64 {
            empty.delete(id);
        }
        assert!(empty.is_empty());
        assert!(empty.search(&data[0..8], 5).unwrap().is_empty());
        empty.verify_integrity().unwrap();
    }

    #[test]
    fn cosine_distance_is_reported_on_the_cosine_scale() {
        let data = [1.0f32, 0.0, 3.0, 4.0, 0.0, 1.0, -1.0, 0.0];
        let config = IvfPqConfig::default().with_nlist(2).flat();
        let index = IvfPq::train(&data, 4, 2, Metric::Cosine, config).unwrap();

        let hits = index.search(&[2.0, 0.0], 4).unwrap();
        let same = hits.iter().find(|hit| hit.id == 0).unwrap();
        assert!(same.distance.abs() < 1e-6, "{same:?}");
        let opposite = hits.iter().find(|hit| hit.id == 3).unwrap();
        assert!((opposite.distance - 2.0).abs() < 1e-6, "{opposite:?}");
        let orthogonal = hits.iter().find(|hit| hit.id == 2).unwrap();
        assert!((orthogonal.distance - 1.0).abs() < 1e-6, "{orthogonal:?}");
    }

    #[test]
    fn cosine_search_ignores_magnitude() {
        // The first vector is ten thousand times shorter than the second, yet
        // it points exactly at the query while the long one is 45 degrees off.
        // A cosine index must rank the short one first.
        let data = [0.01f32, 0.0, 100.0, 100.0, 0.0, 0.001];
        let config = IvfPqConfig::default().with_nlist(2).flat();
        let index = IvfPq::train(&data, 3, 2, Metric::Cosine, config).unwrap();
        let hits = index.search(&[3.0, 0.0], 3).unwrap();
        assert_eq!(hits[0].id, 0, "the aligned vector must rank first");
        assert!(hits[0].distance.abs() < 1e-6, "{hits:?}");
        let angle = hits.iter().find(|hit| hit.id == 1).unwrap();
        assert!((angle.distance - 0.292_893_2).abs() < 1e-5, "{angle:?}");
        let orthogonal = hits.iter().find(|hit| hit.id == 2).unwrap();
        assert!((orthogonal.distance - 1.0).abs() < 1e-6, "{orthogonal:?}");
    }

    #[test]
    fn inner_product_flat_postings_rank_by_dot_product() {
        let data = [1.0f32, 0.0, 1.0, 1.0, -1.0, 0.0];
        let config = IvfPqConfig::default().with_nlist(2).flat();
        let index = IvfPq::train(&data, 3, 2, Metric::InnerProduct, config).unwrap();
        let hits = index.search(&[2.0, 2.0], 3).unwrap();
        assert_eq!(top_ids(&hits), vec![1, 0, 2]);
        assert!((hits[0].distance + 4.0).abs() < 1e-6, "{:?}", hits[0]);
    }

    #[test]
    fn training_is_deterministic() {
        let (count, dim) = (500usize, 16usize);
        let data = clustered_data(count, dim, 10, 0.5, 101);
        let config = IvfPqConfig::default()
            .with_nlist(10)
            .with_pq(4, 16)
            .with_seed(1234);
        let a = IvfPq::train(&data, count, dim, Metric::L2, config).unwrap();
        let b = IvfPq::train(&data, count, dim, Metric::L2, config).unwrap();
        assert_eq!(a.centroids(), b.centroids());
        let query = &data[0..dim];
        assert_eq!(
            top_ids(&a.search(query, 10).unwrap()),
            top_ids(&b.search(query, 10).unwrap())
        );
    }

    #[test]
    fn different_seeds_produce_different_codebooks() {
        let (count, dim) = (500usize, 16usize);
        let data = clustered_data(count, dim, 10, 0.5, 102);
        let a = IvfPq::train(
            &data,
            count,
            dim,
            Metric::L2,
            IvfPqConfig::default()
                .with_nlist(10)
                .with_pq(4, 16)
                .with_seed(1),
        )
        .unwrap();
        let b = IvfPq::train(
            &data,
            count,
            dim,
            Metric::L2,
            IvfPqConfig::default()
                .with_nlist(10)
                .with_pq(4, 16)
                .with_seed(2),
        )
        .unwrap();
        assert_ne!(
            a.quantizer().unwrap().codebooks(),
            b.quantizer().unwrap().codebooks()
        );
    }

    #[test]
    fn stats_describe_the_postings() {
        let (count, dim) = (600usize, 16usize);
        let data = clustered_data(count, dim, 12, 0.5, 111);
        let config = IvfPqConfig::default().with_nlist(12).with_pq(4, 32);
        let index = IvfPq::train(&data, count, dim, Metric::L2, config).unwrap();
        let stats = index.stats();
        assert_eq!(stats.live, count);
        assert_eq!(stats.deleted, 0);
        assert_eq!(stats.dim, dim);
        assert_eq!(stats.nlist, 12);
        assert_eq!(stats.bytes_per_vector, 4);
        assert_eq!(stats.posting_bytes, count * 4);
        assert!(stats.list_len_min > 0, "k-means should fill every list");
        assert!(stats.list_len_max >= stats.list_len_min);
        assert!((stats.list_len_mean - count as f64 / 12.0).abs() < 1.0);
        assert_eq!(stats.empty_lists, 0);
        index.verify_integrity().unwrap();
        assert!(index.memory_bytes() >= count * 4);
    }

    #[test]
    fn probe_order_clamps_and_prefers_near_centroids() {
        let (count, dim) = (400usize, 8usize);
        let data = clustered_data(count, dim, 8, 0.5, 121);
        let config = IvfPqConfig::default().with_nlist(8).flat();
        let index = IvfPq::train(&data, count, dim, Metric::L2, config).unwrap();
        // nprobe above nlist must not panic or repeat lists.
        let everything = index
            .search_with_probes(&data[0..dim], 5, usize::MAX)
            .unwrap();
        assert_eq!(everything.len(), 5);
        let one = index.search_with_probes(&data[0..dim], 1, 1).unwrap();
        assert_eq!(one[0].id, 0);
    }

    #[test]
    fn flat_get_returns_prepared_vectors_and_pq_get_returns_nothing() {
        let data = [3.0f32, 4.0, 0.0, 2.0];
        let flat = IvfPq::train(
            &data,
            2,
            2,
            Metric::Cosine,
            IvfPqConfig::default().with_nlist(1).flat(),
        )
        .unwrap();
        let stored = flat.get(0).unwrap();
        assert!((stored[0] - 0.6).abs() < 1e-6 && (stored[1] - 0.8).abs() < 1e-6);

        let pq = IvfPq::train(
            &data,
            2,
            2,
            Metric::L2,
            IvfPqConfig::default().with_nlist(1).with_pq(2, 2),
        )
        .unwrap();
        assert!(pq.get(0).is_none());
        assert_eq!(pq.slot_count(), 2);
    }
}
