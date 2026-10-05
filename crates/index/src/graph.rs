//! Read-only graph view and the best-first traversal shared by all backends.
//!
//! Search lives here rather than inside [`crate::hnsw::Hnsw`] for one reason:
//! the same traversal has to run over an index that owns its vectors (the
//! build-time path) *and* over an index whose vectors are memory-mapped zero-copy
//! segments (the serving path). Expressing traversal against
//! [`GraphView`] means both paths execute identical code, so a recall figure
//! measured against the in-memory index is evidence about the served index too.
//! Because the trait is generic and statically dispatched, the compiler
//! monomorphises it and the abstraction costs nothing at runtime.

use std::cmp::Reverse;
use std::collections::BinaryHeap;

use lodestar_ann_core::{Candidate, Metric};

/// Everything traversal needs to know about an index.
pub trait GraphView {
    /// Number of nodes, including deleted-but-not-yet-compacted ones.
    fn len(&self) -> usize;
    /// Vector dimensionality.
    fn dim(&self) -> usize;
    /// Metric used for ranking.
    fn metric(&self) -> Metric;
    /// Top-level entry point, if the graph has any nodes.
    fn entry_point(&self) -> Option<u32>;
    /// Highest populated level.
    fn max_level(&self) -> usize;
    /// Highest level at which `node` participates.
    fn node_level(&self, node: u32) -> usize;
    /// Neighbours of `node` at `level`.
    ///
    /// Returns an empty slice when the node does not reach `level`; callers are
    /// expected to ask only for levels up to [`GraphView::node_level`].
    fn neighbors(&self, node: u32, level: usize) -> &[u32];
    /// The prepared vector for `node`.
    fn vector(&self, node: u32) -> &[f32];
    /// Whether `node` is still present (false for tombstones).
    fn is_live(&self, node: u32) -> bool;
    /// External id for `node`.
    fn external_id(&self, node: u32) -> u64;

    /// Whether the graph has no nodes at all.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Reusable visited set for traversal.
///
/// Allocating a `len`-sized visited bitmap per query would dominate the cost of
/// a search on large indexes. Instead the buffer is kept between calls and
/// invalidated by bumping an epoch counter: a node is "seen" when its slot holds
/// the current epoch. Nothing has to be cleared, so resetting is O(1).
#[derive(Debug, Clone, Default)]
pub struct SearchScratch {
    visited: Vec<u32>,
    epoch: u32,
}

impl SearchScratch {
    /// Creates a scratch buffer able to cover `nodes` nodes without reallocating.
    #[must_use]
    pub fn new(nodes: usize) -> Self {
        Self {
            visited: vec![0; nodes],
            epoch: 1,
        }
    }

    /// Grows the buffer if needed. Never shrinks, so steady-state searches on a
    /// stable index do not allocate at all.
    pub fn ensure_capacity(&mut self, nodes: usize) {
        if self.visited.len() < nodes {
            self.visited.resize(nodes, 0);
        }
    }

    /// Number of nodes the buffer can cover.
    #[must_use]
    pub fn capacity(&self) -> usize {
        self.visited.len()
    }

    /// Bytes held by the buffer, for memory accounting.
    #[must_use]
    pub fn memory_bytes(&self) -> usize {
        self.visited.len() * std::mem::size_of::<u32>()
    }

    fn reset(&mut self) {
        let (next, overflow) = self.epoch.overflowing_add(1);
        if overflow {
            // Epoch wrapped: clear so stale marks cannot be mistaken for fresh.
            self.visited.iter_mut().for_each(|slot| *slot = 0);
            self.epoch = 1;
        } else {
            self.epoch = next;
        }
    }

    /// Marks `node` visited, returning `true` the first time it is seen.
    #[inline]
    fn visit(&mut self, node: u32) -> bool {
        let slot = &mut self.visited[node as usize];
        if *slot == self.epoch {
            false
        } else {
            *slot = self.epoch;
            true
        }
    }
}

/// Internal candidate: a node index and its distance from the query.
///
/// Distinct from [`Candidate`], which carries an *external* id. Mixing the two
/// up would silently produce wrong results, so they are different types.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct NodeRef {
    pub distance: f32,
    pub node: u32,
}

impl Eq for NodeRef {}

impl Ord for NodeRef {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.distance
            .partial_cmp(&other.distance)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| self.node.cmp(&other.node))
    }
}

impl PartialOrd for NodeRef {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

/// Greedy descent from the entry point down to (but not including) `level`.
///
/// At every level the walk moves to any neighbour that is closer than the
/// current node and repeats until no improvement exists, which is the classic
/// `SEARCH-LAYER(q, ep, ef=1, lc)` loop from the HNSW paper.
pub(crate) fn greedy_descend<G: GraphView + ?Sized>(
    graph: &G,
    query: &[f32],
    from_level: usize,
) -> Option<(u32, f32)> {
    let mut current = graph.entry_point()?;
    let metric = graph.metric();
    let mut current_distance = metric.distance(query, graph.vector(current));

    for level in (1..=from_level.min(graph.max_level())).rev() {
        loop {
            let mut improved = false;
            for &neighbour in graph.neighbors(current, level) {
                let distance = metric.distance(query, graph.vector(neighbour));
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
    Some((current, current_distance))
}

/// Best-first search restricted to `level`.
///
/// Returns at most `ef` accepted nodes, ordered by increasing distance. Nodes
/// rejected by `filter` are never returned, but they *are* still traversed:
/// aborting traversal at a filtered-out node would disconnect the reachable
/// region and destroy recall for selective filters.
pub(crate) fn search_layer<G: GraphView + ?Sized>(
    graph: &G,
    query: &[f32],
    entry_points: &[u32],
    ef: usize,
    level: usize,
    filter: Option<&dyn Fn(u32) -> bool>,
    scratch: &mut SearchScratch,
) -> BinaryHeap<NodeRef> {
    let metric = graph.metric();
    let ef = ef.max(1);
    scratch.ensure_capacity(graph.len());
    scratch.reset();

    // `candidates` is a max-heap: its root is the *worst* accepted node, which
    // is the one to evict when the frontier improves. `frontier` is a min-heap
    // of nodes still to expand, best first.
    let mut candidates: BinaryHeap<NodeRef> = BinaryHeap::with_capacity(ef + 1);
    let mut frontier: BinaryHeap<Reverse<NodeRef>> = BinaryHeap::with_capacity(ef + 1);

    for &entry in entry_points {
        if entry as usize >= graph.len() || !scratch.visit(entry) {
            continue;
        }
        let distance = metric.distance(query, graph.vector(entry));
        let accepted = graph.is_live(entry) && filter.is_none_or(|accept| accept(entry));
        if accepted {
            candidates.push(NodeRef {
                distance,
                node: entry,
            });
            if candidates.len() > ef {
                candidates.pop();
            }
        }
        frontier.push(Reverse(NodeRef {
            distance,
            node: entry,
        }));
    }

    while let Some(Reverse(current)) = frontier.pop() {
        // Stop once the frontier is worse than everything we have kept.
        if candidates.len() >= ef
            && let Some(worst) = candidates.peek()
            && current.distance > worst.distance
        {
            break;
        }

        for &neighbour in graph.neighbors(current.node, level) {
            if neighbour as usize >= graph.len() || !scratch.visit(neighbour) {
                continue;
            }
            let distance = metric.distance(query, graph.vector(neighbour));
            let accepted =
                graph.is_live(neighbour) && filter.is_none_or(|accept| accept(neighbour));
            if accepted
                && (candidates.len() < ef
                    || candidates.peek().is_some_and(|w| distance < w.distance))
            {
                candidates.push(NodeRef {
                    distance,
                    node: neighbour,
                });
                if candidates.len() > ef {
                    candidates.pop();
                }
            }
            // A filtered-out node still has to be expanded: the path to the
            // accepted nodes may run straight through it.
            let should_expand = match candidates.peek() {
                Some(worst) => candidates.len() < ef || distance < worst.distance,
                None => true,
            };
            if should_expand {
                frontier.push(Reverse(NodeRef {
                    distance,
                    node: neighbour,
                }));
            }
        }
    }

    candidates
}

/// Full top-k search: greedy descent to level 0, then a best-first sweep.
///
/// A fresh scratch buffer is allocated on every call, which is fine for
/// one-shot use but wasteful in a server; use [`search_with_scratch`] there.
#[must_use]
pub fn search<G: GraphView + ?Sized>(
    graph: &G,
    query: &[f32],
    k: usize,
    ef: usize,
    filter: Option<&dyn Fn(u32) -> bool>,
) -> Vec<Candidate> {
    let mut scratch = SearchScratch::new(graph.len());
    search_with_scratch(graph, query, k, ef, filter, &mut scratch)
}

/// Full top-k search reusing a caller-owned scratch buffer.
#[must_use]
pub fn search_with_scratch<G: GraphView + ?Sized>(
    graph: &G,
    query: &[f32],
    k: usize,
    ef: usize,
    filter: Option<&dyn Fn(u32) -> bool>,
    scratch: &mut SearchScratch,
) -> Vec<Candidate> {
    if k == 0 || graph.is_empty() || query.len() != graph.dim() {
        return Vec::new();
    }
    // `ef` below `k` cannot return k results; silently raise it rather than
    // returning short lists that look like recall failures.
    let ef = ef.max(k);
    let Some((entry, _)) = greedy_descend(graph, query, graph.max_level()) else {
        return Vec::new();
    };

    let found = search_layer(graph, query, &[entry], ef, 0, filter, scratch);
    let mut results: Vec<NodeRef> = found.into_sorted_vec();
    results.truncate(k);
    results
        .into_iter()
        .map(|node| Candidate {
            distance: node.distance,
            id: graph.external_id(node.node),
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scratch_reports_first_visit_only() {
        let mut scratch = SearchScratch::new(8);
        assert!(scratch.visit(3));
        assert!(!scratch.visit(3));
        assert!(scratch.visit(4));
        scratch.reset();
        // After a reset every node counts as unseen again.
        assert!(scratch.visit(3));
    }

    #[test]
    fn scratch_grows_but_never_shrinks() {
        let mut scratch = SearchScratch::new(4);
        assert_eq!(scratch.capacity(), 4);
        scratch.ensure_capacity(100);
        assert!(scratch.capacity() >= 100);
        scratch.ensure_capacity(10);
        assert!(scratch.capacity() >= 100);
        assert!(scratch.memory_bytes() >= 100 * 4);
    }

    #[test]
    fn scratch_handles_epoch_wrap_without_false_misses() {
        let mut scratch = SearchScratch::new(4);
        // Simulate an epoch at the very top of its range.
        scratch.epoch = u32::MAX;
        assert!(scratch.visit(1));
        scratch.reset();
        // The wrap must clear the buffer, so node 1 is seen again.
        assert!(scratch.visit(1));
    }

    #[test]
    fn node_ref_orders_by_distance_then_node() {
        let mut values = [
            NodeRef {
                distance: 1.0,
                node: 9,
            },
            NodeRef {
                distance: 0.5,
                node: 4,
            },
            NodeRef {
                distance: 0.5,
                node: 2,
            },
        ];
        values.sort();
        assert_eq!(values[0].node, 2);
        assert_eq!(values[1].node, 4);
        assert_eq!(values[2].node, 9);
    }
}
