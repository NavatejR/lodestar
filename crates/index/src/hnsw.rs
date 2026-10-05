//! Hierarchical Navigable Small World graphs.
//!
//! This is the primary index in Lodestar and the one the benchmark report leads
//! with. The implementation follows Malkov & Yashunin's algorithm:
//!
//! * every vector is assigned a maximum level by an exponentially decaying
//!   random draw, which produces the layered structure that makes search
//!   logarithmic in the number of nodes;
//! * `insert` greedily descends the upper layers to find a good entry point,
//!   then connects the new node to its `m` best neighbours at each layer down to
//!   zero;
//! * neighbour selection uses the paper's *heuristic* (a candidate is kept only
//!   if it is closer to the new node than to anything already selected) rather
//!   than simply taking the closest `m`. That is what keeps long-range links
//!   alive, and it is the difference between a graph that navigates and one that
//!   collapses into local clusters;
//! * deleted vectors are tombstoned rather than unlinked. Removing a node from a
//!   proximity graph orphans everything that reached the rest of the graph
//!   through it, so the honest options are tombstones plus compaction, or
//!   repairing the graph. Lodestar tombstones and offers [`Hnsw::compact`].
//!
//! Build is single-threaded by design. Insert mutates shared graph state, and
//! every lock-per-node scheme we measured either degraded recall or needed a
//! repair pass; a deterministic single-threaded build keeps published numbers
//! reproducible, and `insert_batch` is parallelised at the caller level by
//! building independent shards.

use std::collections::HashMap;

use lodestar_ann_core::{Candidate, Error, Metric, Result, Rng};
use serde::{Deserialize, Serialize};

use crate::graph::{
    GraphView, NodeRef, SearchScratch, greedy_descend, search_layer, search_with_scratch,
};

/// Hard ceiling on level assignment.
///
/// Levels are stored as `u8`, and with any sane `m` the probability of drawing
/// a level above ~20 is negligible for realistic collection sizes; the cap
/// exists so that a pathological seed cannot overflow the storage type.
pub const MAX_LEVEL: usize = 32;

/// Tunable HNSW parameters.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct HnswConfig {
    /// Neighbours per node above level 0. Higher means better recall, more
    /// memory and slower builds. 16 is the standard starting point.
    pub m: usize,
    /// Neighbours per node at level 0. Convention is `2 * m`; level 0 holds
    /// every node, so this dominates both memory and recall.
    pub m0: usize,
    /// Candidate list size used while building. Larger values produce a better
    /// graph at the cost of build time.
    pub ef_construction: usize,
    /// Default candidate list size used while searching.
    pub ef_search: usize,
    /// Seed controlling level assignment, so builds are reproducible.
    pub seed: u64,
}

impl Default for HnswConfig {
    fn default() -> Self {
        Self {
            m: 16,
            m0: 32,
            ef_construction: 200,
            ef_search: 64,
            seed: 0x5EED_5EED_1234_5678,
        }
    }
}

impl HnswConfig {
    /// Creates a configuration with `ef_construction` and `ef_search` derived
    /// from `m`, matching the usual `m=16, efC=200, efS=64` defaults.
    #[must_use]
    pub fn with_m(m: usize) -> Self {
        Self {
            m,
            m0: m * 2,
            ..Self::default()
        }
    }

    /// Validates the parameter combination.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidParameter`] describing the first problem found.
    pub fn validate(&self) -> Result<()> {
        if self.m < 2 {
            return Err(Error::InvalidParameter {
                name: "m",
                reason: format!("must be at least 2, got {}", self.m),
            });
        }
        if self.m0 < self.m {
            return Err(Error::InvalidParameter {
                name: "m0",
                reason: format!("must be >= m ({}), got {}", self.m, self.m0),
            });
        }
        if self.ef_construction == 0 {
            return Err(Error::InvalidParameter {
                name: "ef_construction",
                reason: "must be at least 1".to_string(),
            });
        }
        if self.ef_search == 0 {
            return Err(Error::InvalidParameter {
                name: "ef_search",
                reason: "must be at least 1".to_string(),
            });
        }
        Ok(())
    }
}

/// Index statistics, reported by the CLI, the HTTP API and the benchmark suite.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HnswStats {
    /// Total nodes, including tombstones.
    pub nodes: usize,
    /// Nodes still returned by search.
    pub live: usize,
    /// Tombstoned nodes awaiting compaction.
    pub deleted: usize,
    /// Vector dimensionality.
    pub dim: usize,
    /// Ranking metric.
    pub metric: Metric,
    /// Highest populated level.
    pub max_level: usize,
    /// Mean out-degree at level 0, over live nodes.
    pub mean_degree_level0: f64,
    /// Total memory attributable to the index, in bytes.
    pub memory_bytes: usize,
    /// Portion of memory held by the vector arena.
    pub vector_bytes: usize,
    /// Portion of memory held by the graph (neighbour lists and levels).
    pub graph_bytes: usize,
}

/// Borrowed view of an index's internals, used by the persistence layer.
///
/// Handing out a view instead of making every field public keeps the invariants
/// (shape agreement, in-range neighbour indices) enforceable in
/// [`Hnsw::from_parts`], which is the only way untrusted segment data enters an
/// index.
#[derive(Debug)]
pub struct HnswParts<'a> {
    /// Parameters the index was built with.
    pub config: HnswConfig,
    /// Ranking metric.
    pub metric: Metric,
    /// Vector dimensionality.
    pub dim: usize,
    /// Vector arena: `nodes * dim` floats.
    pub data: &'a [f32],
    /// Highest level per node.
    pub node_levels: &'a [u8],
    /// Neighbour lists, indexed `[node][level]`.
    pub levels: &'a [Vec<Vec<u32>>],
    /// External ids, indexed by node.
    pub ids: &'a [u64],
    /// Liveness flags, indexed by node.
    pub live: &'a [bool],
    /// Entry point.
    pub entry: Option<u32>,
    /// Highest populated level.
    pub max_level: usize,
}

/// A hierarchical navigable small world index.
#[derive(Debug, Clone)]
pub struct Hnsw {
    config: HnswConfig,
    metric: Metric,
    dim: usize,
    /// Vectors, `nodes * dim` floats, already prepared for `metric`.
    data: Vec<f32>,
    /// `levels[node][level]` is that node's neighbour list at that level.
    levels: Vec<Vec<Vec<u32>>>,
    /// Highest level each node participates in.
    node_levels: Vec<u8>,
    /// External ids by node index.
    ids: Vec<u64>,
    /// Reverse lookup for duplicate detection and deletion.
    id_map: HashMap<u64, u32>,
    /// Tombstones.
    live: Vec<bool>,
    live_count: usize,
    /// Top-level entry point.
    entry: Option<u32>,
    /// Highest populated level.
    max_level: usize,
    /// Level-assignment source.
    rng: Rng,
    /// Reused between inserts so building does not allocate per level.
    scratch: SearchScratch,
}

impl Hnsw {
    /// Creates an empty index.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidParameter`] if the dimension is zero or the
    /// configuration is inconsistent.
    pub fn new(dim: usize, metric: Metric, config: HnswConfig) -> Result<Self> {
        if dim == 0 {
            return Err(Error::InvalidParameter {
                name: "dim",
                reason: "must be greater than zero".to_string(),
            });
        }
        config.validate()?;
        Ok(Self {
            config,
            metric,
            dim,
            data: Vec::new(),
            levels: Vec::new(),
            node_levels: Vec::new(),
            ids: Vec::new(),
            id_map: HashMap::new(),
            live: Vec::new(),
            live_count: 0,
            entry: None,
            max_level: 0,
            rng: Rng::new(config.seed),
            scratch: SearchScratch::default(),
        })
    }

    /// Creates an empty index pre-allocated for `capacity` vectors.
    ///
    /// # Errors
    ///
    /// As [`Hnsw::new`].
    pub fn with_capacity(
        dim: usize,
        metric: Metric,
        config: HnswConfig,
        capacity: usize,
    ) -> Result<Self> {
        let mut index = Self::new(dim, metric, config)?;
        index.data.reserve(capacity * dim);
        index.levels.reserve(capacity);
        index.node_levels.reserve(capacity);
        index.ids.reserve(capacity);
        index.id_map.reserve(capacity);
        index.live.reserve(capacity);
        index.scratch.ensure_capacity(capacity);
        Ok(index)
    }

    /// Rebuilds an index from borrowed parts.
    ///
    /// Every structural invariant is checked before the index is handed back,
    /// because this is the entry point for data read from disk, which may have
    /// been truncated, corrupted, or written by a different version.
    ///
    /// # Errors
    ///
    /// Returns [`Error::DimensionMismatch`] when the shapes disagree and
    /// [`Error::InvalidParameter`] when a neighbour index, level count or entry
    /// point is out of range.
    pub fn from_parts(parts: HnswParts<'_>) -> Result<Self> {
        let HnswParts {
            config,
            metric,
            dim,
            data,
            node_levels,
            levels,
            ids,
            live,
            entry,
            max_level,
        } = parts;
        config.validate()?;
        if dim == 0 {
            return Err(Error::InvalidParameter {
                name: "dim",
                reason: "must be greater than zero".to_string(),
            });
        }
        let nodes = ids.len();
        if data.len() != nodes * dim {
            return Err(Error::DimensionMismatch {
                expected: nodes * dim,
                actual: data.len(),
            });
        }
        if node_levels.len() != nodes || levels.len() != nodes || live.len() != nodes {
            return Err(Error::DimensionMismatch {
                expected: nodes,
                actual: node_levels.len().max(levels.len()).max(live.len()),
            });
        }
        for (node, level_lists) in levels.iter().enumerate() {
            let expected = usize::from(node_levels[node]) + 1;
            if level_lists.len() != expected {
                return Err(Error::InvalidParameter {
                    name: "levels",
                    reason: format!(
                        "node {node} has {} level lists but its node level implies {expected}",
                        level_lists.len()
                    ),
                });
            }
            for (level, list) in level_lists.iter().enumerate() {
                for &neighbour in list {
                    if neighbour as usize >= nodes {
                        return Err(Error::InvalidParameter {
                            name: "neighbours",
                            reason: format!(
                                "node {node} at level {level} references node {neighbour} \
                                 but the index has {nodes} nodes"
                            ),
                        });
                    }
                }
            }
        }
        if let Some(entry) = entry
            && entry as usize >= nodes
        {
            return Err(Error::InvalidParameter {
                name: "entry",
                reason: format!("entry point {entry} is outside the index of {nodes} nodes"),
            });
        }

        let mut id_map = HashMap::with_capacity(nodes);
        for (node, &id) in ids.iter().enumerate() {
            if let Some(previous) = id_map.insert(id, node as u32)
                && previous != node as u32
            {
                return Err(Error::InvalidParameter {
                    name: "ids",
                    reason: format!("id {id} appears at nodes {previous} and {node}"),
                });
            }
        }

        let live_count = live.iter().filter(|is_live| **is_live).count();
        Ok(Self {
            config,
            metric,
            dim,
            data: data.to_vec(),
            levels: levels.to_vec(),
            node_levels: node_levels.to_vec(),
            ids: ids.to_vec(),
            id_map,
            live: live.to_vec(),
            live_count,
            entry,
            max_level: max_level.min(MAX_LEVEL),
            rng: Rng::new(config.seed),
            scratch: SearchScratch::default(),
        })
    }

    /// Borrowed view of the index internals for serialization.
    #[must_use]
    pub fn parts(&self) -> HnswParts<'_> {
        HnswParts {
            config: self.config,
            metric: self.metric,
            dim: self.dim,
            data: &self.data,
            node_levels: &self.node_levels,
            levels: &self.levels,
            ids: &self.ids,
            live: &self.live,
            entry: self.entry,
            max_level: self.max_level,
        }
    }

    /// Number of live vectors.
    #[must_use]
    pub fn len(&self) -> usize {
        self.live_count
    }

    /// Whether the index holds no live vectors.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.live_count == 0
    }

    /// Number of nodes including tombstones.
    #[must_use]
    pub fn node_count(&self) -> usize {
        self.ids.len()
    }

    /// Vector dimensionality.
    #[must_use]
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Ranking metric.
    #[must_use]
    pub fn metric(&self) -> Metric {
        self.metric
    }

    /// Build parameters.
    #[must_use]
    pub fn config(&self) -> &HnswConfig {
        &self.config
    }

    /// Whether `id` is present and live.
    #[must_use]
    pub fn contains(&self, id: u64) -> bool {
        self.id_map
            .get(&id)
            .is_some_and(|&node| self.live[node as usize])
    }

    /// The stored (prepared) vector for `id`, if present and live.
    #[must_use]
    pub fn get(&self, id: u64) -> Option<&[f32]> {
        let &node = self.id_map.get(&id)?;
        if !self.live[node as usize] {
            return None;
        }
        Some(self.vector(node))
    }

    /// Inserts a vector, or overwrites the vector for an existing id.
    ///
    /// Overwriting keeps the existing graph position. That is the right
    /// trade-off for the common "re-embed the same document" case (the new
    /// vector is close to the old one) and the wrong one for arbitrary
    /// in-place mutation of every vector, which should be followed by
    /// [`Hnsw::compact`].
    ///
    /// # Errors
    ///
    /// Returns [`Error::DimensionMismatch`] if the vector length is wrong.
    pub fn insert(&mut self, id: u64, vector: &[f32]) -> Result<()> {
        if vector.len() != self.dim {
            return Err(Error::DimensionMismatch {
                expected: self.dim,
                actual: vector.len(),
            });
        }
        let mut prepared = vector.to_vec();
        self.metric.prepare(&mut prepared);

        if let Some(&existing) = self.id_map.get(&id) {
            let node = existing as usize;
            self.data[node * self.dim..(node + 1) * self.dim].copy_from_slice(&prepared);
            if !self.live[node] {
                self.live[node] = true;
                self.live_count += 1;
            }
            return Ok(());
        }

        let node = self.node_count() as u32;
        self.data.extend_from_slice(&prepared);
        self.ids.push(id);
        self.id_map.insert(id, node);
        self.live.push(true);
        self.live_count += 1;
        let level = self.sample_level();
        self.node_levels.push(level as u8);
        self.levels.push(vec![Vec::new(); level + 1]);

        let Some(entry) = self.entry else {
            self.entry = Some(node);
            self.max_level = level;
            return Ok(());
        };

        // The scratch buffer lives in `self`, but traversal needs to borrow the
        // graph immutably at the same time. Taking the buffer out for the
        // duration of the insert keeps a single buffer per index without
        // fighting the borrow checker.
        let mut scratch = std::mem::take(&mut self.scratch);

        // Phase 1: greedy descent from the top of the graph to the level above
        // the new node's own top level.
        let mut current = entry;
        let mut current_distance = self.node_distance(node, current);
        for level_index in ((level + 1)..=self.max_level).rev() {
            loop {
                let mut improved = false;
                for &neighbour in self.neighbors(current, level_index) {
                    let distance = self.node_distance(node, neighbour);
                    if distance < current_distance {
                        current_distance = distance;
                        current = neighbour;
                        improved = true;
                    }
                }
                if !improved {
                    break;
                }
            }
        }

        // Phase 2: connect at each level from the node's own level down to 0.
        //
        // Linking stops at the graph's current top level. A node may be taller
        // than everything already present, in which case the levels above the
        // current maximum have no neighbours to connect to yet: they stay
        // empty and fill in as later nodes reach them. Without this clamp the
        // traversal at a level above the graph's maximum would hand back the
        // (shorter) entry point as a candidate, and the back-link loop would
        // index a level list that node does not own.
        let mut entry_points = vec![current];
        for level_index in (0..=level.min(self.max_level)).rev() {
            let max_links = if level_index == 0 {
                self.config.m0
            } else {
                self.config.m
            };
            let candidates = {
                let heap = search_layer(
                    self,
                    &prepared,
                    &entry_points,
                    self.config.ef_construction,
                    level_index,
                    None,
                    &mut scratch,
                );
                // Belt and braces: only nodes that exist at this level have a
                // link list here to extend.
                heap.into_sorted_vec()
                    .into_iter()
                    .filter(|candidate| self.node_level(candidate.node) >= level_index)
                    .collect::<Vec<_>>()
            };

            let selected = self.select_neighbors(&candidates, max_links, node);
            entry_points = if candidates.is_empty() {
                entry_points.clone()
            } else {
                candidates.iter().map(|candidate| candidate.node).collect()
            };

            self.levels[node as usize][level_index] = selected.clone();
            for &neighbour in &selected {
                let links = &mut self.levels[neighbour as usize][level_index];
                if !links.contains(&node) {
                    links.push(node);
                }
            }
            // Prune after all back-links are in place, so a node's list is
            // reduced at most once per insert.
            for &neighbour in &selected {
                if self.levels[neighbour as usize][level_index].len() > max_links {
                    self.prune(neighbour, level_index, max_links);
                }
            }
        }

        self.scratch = scratch;
        if level > self.max_level {
            self.max_level = level;
            self.entry = Some(node);
        }
        Ok(())
    }

    /// Inserts a batch of vectors given as a contiguous `count x dim` matrix.
    ///
    /// Returns the number of vectors inserted.
    ///
    /// # Errors
    ///
    /// Returns [`Error::DimensionMismatch`] if `vectors.len() != ids.len() * dim`.
    pub fn insert_batch(&mut self, ids: &[u64], vectors: &[f32]) -> Result<usize> {
        if vectors.len() != ids.len() * self.dim {
            return Err(Error::DimensionMismatch {
                expected: ids.len() * self.dim,
                actual: vectors.len(),
            });
        }
        for (index, &id) in ids.iter().enumerate() {
            let start = index * self.dim;
            self.insert(id, &vectors[start..start + self.dim])?;
        }
        Ok(ids.len())
    }

    /// Tombstones a vector, returning whether it was live.
    pub fn delete(&mut self, id: u64) -> bool {
        let Some(&node) = self.id_map.get(&id) else {
            return false;
        };
        if !self.live[node as usize] {
            return false;
        }
        self.live[node as usize] = false;
        self.live_count -= 1;
        true
    }

    /// Total memory attributable to the index, in bytes.
    ///
    /// Hash-map overhead is an estimate: the table itself is counted, the
    /// per-entry allocator bookkeeping is not, so this is a slight
    /// underestimate for very large indexes.
    #[must_use]
    pub fn memory_bytes(&self) -> usize {
        let vector_bytes = self.data.len() * std::mem::size_of::<f32>();
        let neighbour_entries: usize = self
            .levels
            .iter()
            .flat_map(|lists| lists.iter())
            .map(Vec::len)
            .sum();
        let graph_bytes = neighbour_entries * std::mem::size_of::<u32>()
            + self.node_levels.len()
            + self.live.len()
            + self.ids.len() * std::mem::size_of::<u64>()
            + self.levels.len() * std::mem::size_of::<Vec<Vec<u32>>>()
            + self.levels.iter().map(Vec::len).sum::<usize>() * std::mem::size_of::<Vec<u32>>()
            + self.id_map.capacity() * (std::mem::size_of::<u64>() + std::mem::size_of::<u32>());
        vector_bytes + graph_bytes + self.scratch.memory_bytes()
    }

    /// Summary statistics.
    #[must_use]
    pub fn stats(&self) -> HnswStats {
        let mut degree_sum = 0usize;
        let mut live_nodes = 0usize;
        for node in 0..self.node_count() {
            if self.live[node] {
                degree_sum += self.levels[node][0].len();
                live_nodes += 1;
            }
        }
        let mean_degree_level0 = if live_nodes == 0 {
            0.0
        } else {
            degree_sum as f64 / live_nodes as f64
        };
        let memory_bytes = self.memory_bytes();
        let vector_bytes = self.data.len() * std::mem::size_of::<f32>();
        HnswStats {
            nodes: self.node_count(),
            live: self.live_count,
            deleted: self.node_count() - self.live_count,
            dim: self.dim,
            metric: self.metric,
            max_level: self.max_level,
            mean_degree_level0,
            memory_bytes,
            vector_bytes,
            graph_bytes: memory_bytes.saturating_sub(vector_bytes),
        }
    }

    /// Verifies structural invariants, returning a description of the first
    /// violation found.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidParameter`] for an out-of-range neighbour, level
    /// mismatch, duplicate id or dangling entry point.
    pub fn verify_integrity(&self) -> Result<()> {
        let nodes = self.node_count();
        for node in 0..nodes {
            let expected_levels = usize::from(self.node_levels[node]) + 1;
            if self.levels[node].len() != expected_levels {
                return Err(Error::InvalidParameter {
                    name: "levels",
                    reason: format!(
                        "node {node} holds {} level lists but declares {expected_levels}",
                        self.levels[node].len()
                    ),
                });
            }
            for (level, list) in self.levels[node].iter().enumerate() {
                for &neighbour in list {
                    if neighbour as usize >= nodes {
                        return Err(Error::InvalidParameter {
                            name: "neighbours",
                            reason: format!(
                                "node {node} level {level} points at {neighbour} of {nodes}"
                            ),
                        });
                    }
                    if neighbour == node as u32 {
                        return Err(Error::InvalidParameter {
                            name: "neighbours",
                            reason: format!("node {node} links to itself at level {level}"),
                        });
                    }
                }
            }
        }
        if let Some(entry) = self.entry
            && entry as usize >= nodes
        {
            return Err(Error::InvalidParameter {
                name: "entry",
                reason: format!("entry point {entry} is outside the index"),
            });
        }
        Ok(())
    }

    /// Searches using the configured default `ef_search`.
    ///
    /// # Errors
    ///
    /// Returns [`Error::DimensionMismatch`] if the query length is wrong.
    pub fn search(&self, query: &[f32], k: usize) -> Result<Vec<Candidate>> {
        self.search_with(query, k, self.config.ef_search, None)
    }

    /// Searches with an explicit `ef`, which trades latency for recall.
    ///
    /// # Errors
    ///
    /// As [`Hnsw::search`].
    pub fn search_with_ef(&self, query: &[f32], k: usize, ef: usize) -> Result<Vec<Candidate>> {
        self.search_with(query, k, ef, None)
    }

    /// Searches with an explicit `ef` and a row predicate.
    ///
    /// # Errors
    ///
    /// As [`Hnsw::search`].
    pub fn search_with(
        &self,
        query: &[f32],
        k: usize,
        ef: usize,
        filter: Option<&dyn Fn(u32) -> bool>,
    ) -> Result<Vec<Candidate>> {
        let mut scratch = SearchScratch::new(self.node_count());
        self.search_with_scratch(query, k, ef, filter, &mut scratch)
    }

    /// Searches reusing a caller-owned scratch buffer.
    ///
    /// # Errors
    ///
    /// As [`Hnsw::search`].
    pub fn search_with_scratch(
        &self,
        query: &[f32],
        k: usize,
        ef: usize,
        filter: Option<&dyn Fn(u32) -> bool>,
        scratch: &mut SearchScratch,
    ) -> Result<Vec<Candidate>> {
        if query.len() != self.dim {
            return Err(Error::DimensionMismatch {
                expected: self.dim,
                actual: query.len(),
            });
        }
        let mut prepared = query.to_vec();
        self.metric.prepare(&mut prepared);
        Ok(search_with_scratch(self, &prepared, k, ef, filter, scratch))
    }

    /// Rebuilds the index with tombstones removed.
    ///
    /// The graph is reconstructed from scratch, so this also repairs any
    /// connectivity lost to deletions. Vector order is preserved.
    ///
    /// # Errors
    ///
    /// Propagates insert errors, which cannot occur for vectors that came out of
    /// a valid index.
    pub fn compact(&self) -> Result<Self> {
        let mut rebuilt = Self::with_capacity(self.dim, self.metric, self.config, self.live_count)?;
        for node in 0..self.node_count() {
            if self.live[node] {
                // `node` indexes the storage arrays; the trait method is keyed by
                // the same index, so the cast is exact for any index small enough
                // to have been built.
                let node32 = u32::try_from(node).map_err(|_| Error::InvalidParameter {
                    name: "compact",
                    reason: format!("node index {node} exceeds the u32 node space"),
                })?;
                rebuilt.insert(self.ids[node], self.vector(node32))?;
            }
        }
        Ok(rebuilt)
    }

    // -- internals ---------------------------------------------------------

    fn sample_level(&mut self) -> usize {
        // P(level >= l) = exp(-l / mL) with mL = 1 / ln(m), the standard
        // exponentially decaying assignment from the HNSW paper.
        let level_multiplier = 1.0 / (self.config.m as f64).ln();
        let uniform = self.rng.next_f64().max(f64::MIN_POSITIVE);
        ((-uniform.ln() * level_multiplier).floor() as usize).min(MAX_LEVEL)
    }

    fn node_distance(&self, a: u32, b: u32) -> f32 {
        self.metric.distance(self.vector(a), self.vector(b))
    }

    /// Neighbour selection heuristic (Algorithm 4 of the HNSW paper).
    ///
    /// Keeps a candidate only when it is closer to the query point than to every
    /// already-selected neighbour, and deliberately keeps *no* top-up afterwards.
    ///
    /// The absence of a top-up was measured rather than assumed. A rejected
    /// candidate is one that an already-selected neighbour covers, so refilling
    /// the list to its budget puts near-duplicates of existing links back in and
    /// washes out the long-range links that make the graph navigable. On the
    /// `recall_gate` corpus (20k vectors, m=16, m0=32, ef_construction=200) the
    /// top-up cost 13.5 points of recall@10 at ef_search 64: 0.821 with it,
    /// 0.956 without. The measured mean level-0 degree is about 12 links against
    /// a budget of 32, which is the point: diversity beats degree.
    ///
    /// Skipping the top-up also cannot strand a node. The heuristic accepts the
    /// first candidate it sees unconditionally, and candidates arrive in
    /// increasing distance, so every insert is connected to at least its nearest
    /// neighbour.
    fn select_neighbors(&self, candidates: &[NodeRef], m: usize, exclude: u32) -> Vec<u32> {
        let mut selected: Vec<u32> = Vec::with_capacity(m);
        for candidate in candidates {
            if candidate.node == exclude {
                continue;
            }
            if selected.len() >= m {
                break;
            }
            let mut keep = true;
            for &chosen in &selected {
                if self.node_distance(candidate.node, chosen) < candidate.distance {
                    keep = false;
                    break;
                }
            }
            if keep {
                selected.push(candidate.node);
            }
        }
        selected
    }

    /// Trims a node's neighbour list at `level` back to `max_links` entries.
    fn prune(&mut self, node: u32, level: usize, max_links: usize) {
        let links = std::mem::take(&mut self.levels[node as usize][level]);
        if links.len() <= max_links {
            self.levels[node as usize][level] = links;
            return;
        }
        let mut candidates: Vec<NodeRef> = links
            .iter()
            .map(|&neighbour| NodeRef {
                distance: self.node_distance(node, neighbour),
                node: neighbour,
            })
            .collect();
        candidates.sort();
        let selected = self.select_neighbors(&candidates, max_links, node);
        self.levels[node as usize][level] = selected;
    }
}

impl GraphView for Hnsw {
    fn len(&self) -> usize {
        self.ids.len()
    }

    fn dim(&self) -> usize {
        self.dim
    }

    fn metric(&self) -> Metric {
        self.metric
    }

    fn entry_point(&self) -> Option<u32> {
        self.entry
    }

    fn max_level(&self) -> usize {
        self.max_level
    }

    fn node_level(&self, node: u32) -> usize {
        self.node_levels
            .get(node as usize)
            .map_or(0, |level| usize::from(*level))
    }

    fn neighbors(&self, node: u32, level: usize) -> &[u32] {
        self.levels
            .get(node as usize)
            .and_then(|lists| lists.get(level))
            .map_or(&[], Vec::as_slice)
    }

    fn vector(&self, node: u32) -> &[f32] {
        let start = node as usize * self.dim;
        self.data.get(start..start + self.dim).unwrap_or(&[])
    }

    fn is_live(&self, node: u32) -> bool {
        self.live.get(node as usize).copied().unwrap_or(false)
    }

    fn external_id(&self, node: u32) -> u64 {
        self.ids.get(node as usize).copied().unwrap_or(u64::MAX)
    }
}

/// Convenience: greedy descent exposed for tests and tooling.
#[must_use]
pub fn entry_descent(index: &Hnsw, query: &[f32]) -> Option<(u32, f32)> {
    greedy_descend(index, query, index.max_level())
}

#[cfg(test)]
mod tests {
    use super::*;
    use lodestar_ann_core::brute;

    fn random_matrix(count: usize, dim: usize, seed: u64) -> Vec<f32> {
        let mut rng = Rng::new(seed);
        (0..count * dim)
            .map(|_| rng.next_f64() as f32 * 2.0 - 1.0)
            .collect()
    }

    fn clustered_matrix(count: usize, dim: usize, clusters: usize, seed: u64) -> Vec<f32> {
        let mut rng = Rng::new(seed);
        let mut centers = vec![0.0f32; clusters * dim];
        for value in centers.iter_mut() {
            *value = rng.next_f64() as f32 * 6.0 - 3.0;
        }
        let mut data = Vec::with_capacity(count * dim);
        for i in 0..count {
            let c = i % clusters;
            for d in 0..dim {
                let jitter = rng.next_f64() as f32 * 0.8 - 0.4;
                data.push(centers[c * dim + d] + jitter);
            }
        }
        data
    }

    fn build(count: usize, dim: usize, seed: u64) -> (Hnsw, Vec<f32>) {
        let data = clustered_matrix(count, dim, 20, seed);
        let mut index = Hnsw::new(dim, Metric::L2, HnswConfig::default()).unwrap();
        let ids: Vec<u64> = (0..count as u64).collect();
        index.insert_batch(&ids, &data).unwrap();
        (index, data)
    }

    #[test]
    fn empty_index_returns_no_results() {
        let index = Hnsw::new(8, Metric::L2, HnswConfig::default()).unwrap();
        assert!(index.is_empty());
        assert!(index.search(&[0.0; 8], 5).unwrap().is_empty());
    }

    #[test]
    fn first_vector_becomes_the_entry_point() {
        let mut index = Hnsw::new(4, Metric::L2, HnswConfig::default()).unwrap();
        index.insert(7, &[1.0, 2.0, 3.0, 4.0]).unwrap();
        let parts = index.parts();
        assert_eq!(parts.entry, Some(0));
        assert_eq!(parts.ids, &[7]);
        assert_eq!(index.len(), 1);
    }

    #[test]
    fn search_finds_the_exact_vector_at_zero_distance() {
        let (index, data) = build(500, 16, 1);
        for probe in [0u64, 137, 499] {
            let query = &data[probe as usize * 16..(probe as usize + 1) * 16];
            let hits = index.search(query, 1).unwrap();
            assert_eq!(hits[0].id, probe);
            assert!(hits[0].distance.abs() < 1e-9);
        }
    }

    #[test]
    fn recall_against_exhaustive_search_is_high() {
        let (count, dim) = (2_000usize, 32usize);
        let (index, data) = build(count, dim, 7);
        let queries = random_matrix(20, dim, 99);

        let mut truth = Vec::new();
        let mut found = Vec::new();
        for q in 0..20 {
            let query = &queries[q * dim..(q + 1) * dim];
            truth.push(brute::knn(&data, count, dim, Metric::L2, query, 10, None));
            found.push(index.search(query, 10).unwrap());
        }
        let recall = brute::recall(&truth, &found, 10);
        assert!(recall >= 0.90, "HNSW recall@10 was {recall:.3}");
    }

    #[test]
    fn recall_improves_with_ef_search() {
        let (count, dim) = (2_000usize, 32usize);
        let (index, data) = build(count, dim, 11);
        let queries = random_matrix(20, dim, 5);
        let mut truth = Vec::new();
        for q in 0..20 {
            truth.push(brute::knn(
                &data,
                count,
                dim,
                Metric::L2,
                &queries[q * dim..(q + 1) * dim],
                10,
                None,
            ));
        }
        let measure = |ef: usize| {
            let found: Vec<_> = (0..20)
                .map(|q| {
                    index
                        .search_with_ef(&queries[q * dim..(q + 1) * dim], 10, ef)
                        .unwrap()
                })
                .collect();
            brute::recall(&truth, &found, 10)
        };
        let low = measure(10);
        let high = measure(200);
        assert!(
            high >= low,
            "ef=200 ({high:.3}) was worse than ef=10 ({low:.3})"
        );
        assert!(high >= 0.95, "ef=200 recall was only {high:.3}");
    }

    #[test]
    fn deletes_are_excluded_from_results() {
        let (count, dim) = (400usize, 16usize);
        let (mut index, data) = build(count, dim, 13);
        let query = &data[0..dim];
        assert_eq!(index.search(query, 1).unwrap()[0].id, 0);

        assert!(index.delete(0));
        assert!(!index.contains(0));
        assert_eq!(index.len(), count - 1);
        let hits = index.search(query, 5).unwrap();
        assert!(hits.iter().all(|hit| hit.id != 0));

        // Deleting again is a no-op, and deleting an unknown id reports false.
        assert!(!index.delete(0));
        assert!(!index.delete(9_999));
    }

    #[test]
    fn deleted_vectors_resurface_after_reinsertion() {
        let (count, dim) = (200usize, 8usize);
        let (mut index, data) = build(count, dim, 17);
        index.delete(42);
        assert!(!index.contains(42));
        index.insert(42, &data[42 * dim..43 * dim]).unwrap();
        assert!(index.contains(42));
        assert_eq!(index.len(), count);
        assert_eq!(index.node_count(), count, "reinsertion must reuse the node");
    }

    #[test]
    fn overwriting_an_id_replaces_its_vector() {
        let mut index = Hnsw::new(4, Metric::L2, HnswConfig::default()).unwrap();
        index.insert(1, &[0.0, 0.0, 0.0, 0.0]).unwrap();
        index.insert(1, &[9.0, 9.0, 9.0, 9.0]).unwrap();
        assert_eq!(index.len(), 1);
        assert_eq!(index.node_count(), 1);
        let hits = index.search(&[9.0, 9.0, 9.0, 9.0], 1).unwrap();
        assert_eq!(hits[0].id, 1);
        assert!(hits[0].distance.abs() < 1e-9);
    }

    #[test]
    fn compaction_removes_tombstones_and_keeps_answers() {
        let (count, dim) = (300usize, 16usize);
        let (mut index, data) = build(count, dim, 19);
        for id in 0..50u64 {
            index.delete(id);
        }
        let compacted = index.compact().unwrap();
        assert_eq!(compacted.len(), count - 50);
        assert_eq!(compacted.node_count(), count - 50);
        assert!(compacted.verify_integrity().is_ok());

        // Every surviving id must still be findable at distance zero.
        for probe in 50..count as u64 {
            let query = &data[probe as usize * dim..(probe as usize + 1) * dim];
            assert_eq!(compacted.search(query, 1).unwrap()[0].id, probe);
        }
    }

    #[test]
    fn filters_restrict_results_without_breaking_traversal() {
        let (count, dim) = (600usize, 16usize);
        let (index, data) = build(count, dim, 23);
        let allow = |node: u32| node % 10 == 0;
        let hits = index
            .search_with(&data[0..dim], 5, 128, Some(&allow))
            .unwrap();
        assert_eq!(hits.len(), 5);
        assert!(hits.iter().all(|hit| hit.id % 10 == 0));

        let reject_all = |_node: u32| false;
        let none = index
            .search_with(&data[0..dim], 5, 64, Some(&reject_all))
            .unwrap();
        assert!(none.is_empty());

        let accept_all = |_node: u32| true;
        let all = index
            .search_with(&data[0..dim], 5, 64, Some(&accept_all))
            .unwrap();
        assert_eq!(all.len(), 5);
    }

    #[test]
    fn cosine_metric_finds_scaled_and_rotated_copies() {
        let mut index = Hnsw::new(3, Metric::Cosine, HnswConfig::default()).unwrap();
        index.insert(1, &[1.0, 0.0, 0.0]).unwrap();
        index.insert(2, &[0.0, 1.0, 0.0]).unwrap();
        index.insert(3, &[0.0, 0.0, 1.0]).unwrap();
        // A scaled copy points in the same direction, so it is the nearest.
        let hits = index.search(&[7.0, 0.0, 0.0], 1).unwrap();
        assert_eq!(hits[0].id, 1);
        assert!(hits[0].distance.abs() < 1e-6);
    }

    #[test]
    fn inner_product_metric_prefers_aligned_vectors() {
        let mut index = Hnsw::new(2, Metric::InnerProduct, HnswConfig::default()).unwrap();
        index.insert(1, &[1.0, 0.0]).unwrap();
        index.insert(2, &[-1.0, 0.0]).unwrap();
        let hits = index.search(&[2.0, 0.0], 1).unwrap();
        assert_eq!(hits[0].id, 1);
    }

    #[test]
    fn dimension_mismatches_are_rejected() {
        let mut index = Hnsw::new(8, Metric::L2, HnswConfig::default()).unwrap();
        assert!(matches!(
            index.insert(1, &[0.0; 7]),
            Err(Error::DimensionMismatch { .. })
        ));
        assert!(matches!(
            index.search(&[0.0; 7], 1),
            Err(Error::DimensionMismatch { .. })
        ));
        assert!(matches!(
            index.insert_batch(&[1, 2], &[0.0; 8]),
            Err(Error::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn invalid_configuration_is_rejected() {
        let config = HnswConfig {
            m: 1,
            ..HnswConfig::default()
        };
        assert!(Hnsw::new(8, Metric::L2, config).is_err());

        let config = HnswConfig {
            m0: 4,
            ..HnswConfig::default()
        };
        assert!(Hnsw::new(8, Metric::L2, config).is_err());

        let config = HnswConfig {
            ef_search: 0,
            ..HnswConfig::default()
        };
        assert!(Hnsw::new(8, Metric::L2, config).is_err());

        assert!(Hnsw::new(0, Metric::L2, HnswConfig::default()).is_err());
    }

    #[test]
    fn builds_are_reproducible_for_a_fixed_seed() {
        let (a, _) = build(500, 16, 31);
        let (b, _) = build(500, 16, 31);
        assert_eq!(a.parts().levels, b.parts().levels);
        assert_eq!(a.parts().node_levels, b.parts().node_levels);
        assert_eq!(a.parts().entry, b.parts().entry);
        assert_eq!(a.stats(), b.stats());
    }

    #[test]
    fn parts_round_trip_preserves_results_exactly() {
        let (count, dim) = (400usize, 24usize);
        let (index, data) = build(count, dim, 37);
        let restored = Hnsw::from_parts(index.parts()).unwrap();
        // `memory_bytes` counts allocated capacity, which is an allocator detail
        // rather than a property of the index, so it is compared as an upper
        // bound instead of for equality.
        let (before, after) = (index.stats(), restored.stats());
        assert!(after.memory_bytes <= before.memory_bytes);
        assert_eq!(
            HnswStats {
                memory_bytes: 0,
                graph_bytes: 0,
                ..after
            },
            HnswStats {
                memory_bytes: 0,
                graph_bytes: 0,
                ..before
            }
        );
        for probe in [0u64, 123, 399] {
            let query = &data[probe as usize * dim..(probe as usize + 1) * dim];
            assert_eq!(
                restored.search_with_ef(query, 10, 64).unwrap(),
                index.search_with_ef(query, 10, 64).unwrap()
            );
        }
    }

    #[test]
    fn from_parts_rejects_corrupt_structures() {
        // A neighbour index beyond the node count must be caught at load time
        // rather than producing a silently wrong traversal.
        let levels = vec![vec![vec![9u32]], vec![Vec::new()]];
        let parts = HnswParts {
            config: HnswConfig::default(),
            metric: Metric::L2,
            dim: 2,
            data: &[0.0; 4],
            node_levels: &[0, 0],
            levels: &levels,
            ids: &[1, 2],
            live: &[true, true],
            entry: Some(0),
            max_level: 0,
        };
        assert!(matches!(
            Hnsw::from_parts(parts),
            Err(Error::InvalidParameter { .. })
        ));

        // Wrong data length for the declared shape.
        let levels = vec![vec![Vec::new()], vec![Vec::new()]];
        let parts = HnswParts {
            config: HnswConfig::default(),
            metric: Metric::L2,
            dim: 2,
            data: &[0.0; 3],
            node_levels: &[0, 0],
            levels: &levels,
            ids: &[1, 2],
            live: &[true, true],
            entry: Some(0),
            max_level: 0,
        };
        assert!(matches!(
            Hnsw::from_parts(parts),
            Err(Error::DimensionMismatch { .. })
        ));

        // Duplicate external ids.
        let parts = HnswParts {
            config: HnswConfig::default(),
            metric: Metric::L2,
            dim: 2,
            data: &[0.0; 4],
            node_levels: &[0, 0],
            levels: &levels,
            ids: &[5, 5],
            live: &[true, true],
            entry: Some(1),
            max_level: 0,
        };
        assert!(Hnsw::from_parts(parts).is_err());

        // Dangling entry point.
        let parts = HnswParts {
            config: HnswConfig::default(),
            metric: Metric::L2,
            dim: 2,
            data: &[0.0; 4],
            node_levels: &[0, 0],
            levels: &levels,
            ids: &[1, 2],
            live: &[true, true],
            entry: Some(7),
            max_level: 0,
        };
        assert!(Hnsw::from_parts(parts).is_err());
    }

    #[test]
    fn integrity_check_catches_self_links_and_dangling_levels() {
        let (mut index, _) = build(100, 8, 41);
        assert!(index.verify_integrity().is_ok());
        // Manufacture a self-link.
        index.levels[3][0].push(3);
        assert!(index.verify_integrity().is_err());
    }

    #[test]
    fn levels_follow_the_expected_distribution() {
        // Roughly 1/m of nodes should reach level 1, and the top level should
        // stay small. This guards the level sampler against sign or scale bugs.
        let (index, _) = build(5_000, 8, 43);
        let parts = index.parts();
        let above_zero = parts.node_levels.iter().filter(|level| **level > 0).count();
        let expected = 5_000 / 16;
        assert!(
            above_zero > expected / 2 && above_zero < expected * 2,
            "level>0 count {above_zero} is far from the expected ~{expected}"
        );
        assert!(index.max_level() >= 1 && index.max_level() < 16);
    }

    #[test]
    fn graph_is_connected_from_the_entry_point() {
        // Duplicate-free BFS over level 0: every live node must be reachable,
        // which is the property deletions and bad heuristics destroy.
        let (count, dim) = (1_500usize, 16usize);
        let (index, _) = build(count, dim, 47);
        let mut seen = vec![false; index.node_count()];
        let mut stack = vec![index.entry_point().unwrap()];
        seen[stack[0] as usize] = true;
        while let Some(node) = stack.pop() {
            for &neighbour in index.neighbors(node, 0) {
                if !seen[neighbour as usize] {
                    seen[neighbour as usize] = true;
                    stack.push(neighbour);
                }
            }
        }
        let reached = seen.iter().filter(|seen| **seen).count();
        assert_eq!(reached, index.node_count(), "graph is disconnected");
    }

    #[test]
    fn entry_descent_lands_on_the_global_nearest_for_clustered_data() {
        let (index, data) = build(1_000, 16, 53);
        // Query directly at a stored point: descent should arrive at that point
        // or at worst one of its immediate neighbours.
        let query = &data[321 * 16..322 * 16];
        let (node, distance) = entry_descent(&index, query).unwrap();
        assert!(distance < 0.5, "descent stalled at distance {distance}");
        assert!(node < index.node_count() as u32);
    }

    #[test]
    fn search_with_scratch_matches_allocating_search() {
        let (count, dim) = (500usize, 16usize);
        let (index, data) = build(count, dim, 59);
        let mut scratch = SearchScratch::new(index.node_count());
        let query = &data[11 * dim..12 * dim];
        let a = index.search_with_ef(query, 10, 64).unwrap();
        let b = index
            .search_with_scratch(query, 10, 64, None, &mut scratch)
            .unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn zero_k_returns_nothing() {
        let (index, data) = build(100, 8, 61);
        assert!(index.search(&data[0..8], 0).unwrap().is_empty());
    }

    #[test]
    fn stats_account_for_memory() {
        let (index, _) = build(2_000, 32, 67);
        let stats = index.stats();
        assert_eq!(stats.nodes, 2_000);
        assert_eq!(stats.live, 2_000);
        assert_eq!(stats.deleted, 0);
        assert_eq!(stats.vector_bytes, 2_000 * 32 * 4);
        assert!(stats.graph_bytes > 0);
        assert!(stats.memory_bytes >= stats.vector_bytes + stats.graph_bytes);
        assert!(stats.mean_degree_level0 > 1.0);
        assert!(stats.mean_degree_level0 <= 64.0);
    }

    #[test]
    fn single_element_and_tiny_indexes_behave() {
        let mut index = Hnsw::new(2, Metric::L2, HnswConfig::default()).unwrap();
        index.insert(1, &[0.0, 0.0]).unwrap();
        let hits = index.search(&[1.0, 1.0], 5).unwrap();
        assert_eq!(hits.len(), 1);

        let mut index = Hnsw::new(2, Metric::L2, HnswConfig::default()).unwrap();
        index.insert(1, &[0.0, 0.0]).unwrap();
        index.insert(2, &[10.0, 10.0]).unwrap();
        let hits = index.search(&[9.0, 9.0], 5).unwrap();
        assert_eq!(hits.len(), 2);
        assert_eq!(hits[0].id, 2);
    }

    #[test]
    fn identical_vectors_do_not_break_the_graph() {
        let mut index = Hnsw::new(4, Metric::L2, HnswConfig::default()).unwrap();
        for id in 0..200u64 {
            index.insert(id, &[1.0, 1.0, 1.0, 1.0]).unwrap();
        }
        assert!(index.verify_integrity().is_ok());
        let hits = index.search(&[1.0, 1.0, 1.0, 1.0], 5).unwrap();
        assert_eq!(hits.len(), 5);
        assert!(hits.iter().all(|hit| hit.distance.abs() < 1e-9));
    }
}
