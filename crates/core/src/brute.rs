//! Exact k-nearest-neighbour search.
//!
//! This module is the reference implementation the whole project is measured
//! against. It is deliberately simple: scan every vector, keep a bounded
//! max-heap of the best `k` candidates. Because it is exhaustive it defines
//! ground truth, so the recall numbers published in `BENCHMARKS.md` are
//! computed from it rather than from another approximate library.
//!
//! It is also a legitimate search strategy in its own right for collections
//! below roughly ten thousand vectors, where a graph index costs more to build
//! and traverse than a straight scan.

use std::collections::BinaryHeap;

use crate::{Candidate, Metric};

/// Optional predicate deciding whether a row index is eligible for results.
pub type RowFilter<'a> = &'a dyn Fn(usize) -> bool;

/// Exact `k`-nearest neighbours for a single query.
///
/// `data` is a contiguous `count x dim` matrix and `query` must have length
/// `dim`. `data` must already be prepared for `metric` (see
/// [`Metric::prepare`]).
///
/// Returns at most `k` candidates ordered by increasing distance.
///
/// # Panics
///
/// Panics if `data.len() != count * dim` or `query.len() != dim`. Those are
/// programming errors, and silently truncating would corrupt recall numbers.
#[must_use]
pub fn knn(
    data: &[f32],
    count: usize,
    dim: usize,
    metric: Metric,
    query: &[f32],
    k: usize,
    filter: Option<RowFilter<'_>>,
) -> Vec<Candidate> {
    assert_eq!(data.len(), count * dim, "data must be count x dim");
    assert_eq!(query.len(), dim, "query must have the index dimension");
    if k == 0 || count == 0 {
        return Vec::new();
    }

    // Max-heap keyed by Candidate ordering: the worst element is on top, so the
    // heap can be capped at k entries.
    let mut heap: BinaryHeap<Candidate> = BinaryHeap::with_capacity(k + 1);
    for row in 0..count {
        if let Some(allow) = filter {
            if !allow(row) {
                continue;
            }
        }
        let vector = &data[row * dim..(row + 1) * dim];
        let distance = metric.distance(query, vector);
        if heap.len() < k {
            heap.push(Candidate {
                distance,
                id: row as u64,
            });
        } else if let Some(worst) = heap.peek() {
            if distance < worst.distance {
                heap.pop();
                heap.push(Candidate {
                    distance,
                    id: row as u64,
                });
            }
        }
    }

    let mut out: Vec<Candidate> = heap.into_vec();
    out.sort_unstable();
    out
}

/// Exact `k`-nearest neighbours for many queries, in parallel.
///
/// Returns one result vector per query, in the order the queries were given.
///
/// # Panics
///
/// Panics if the input shapes are inconsistent.
#[must_use]
pub fn knn_batch(
    data: &[f32],
    count: usize,
    dim: usize,
    metric: Metric,
    queries: &[f32],
    nq: usize,
    k: usize,
) -> Vec<Vec<Candidate>> {
    use rayon::prelude::*;

    assert_eq!(data.len(), count * dim, "data must be count x dim");
    assert_eq!(queries.len(), nq * dim, "queries must be nq x dim");
    (0..nq)
        .into_par_iter()
        .map(|q| {
            let query = &queries[q * dim..(q + 1) * dim];
            knn(data, count, dim, metric, query, k, None)
        })
        .collect()
}

/// Recall of `results` against `truth`, both ordered by increasing distance.
///
/// Recall is `|results ∩ truth[:k]| / k`, averaged over all queries. Both inputs
/// must be keyed by the *same* external ids.
#[must_use]
pub fn recall(truth: &[Vec<Candidate>], results: &[Vec<Candidate>], k: usize) -> f64 {
    assert_eq!(truth.len(), results.len(), "query counts must match");
    if k == 0 || truth.is_empty() {
        return 0.0;
    }
    let mut hits = 0usize;
    let mut total = 0usize;
    for (golden, found) in truth.iter().zip(results) {
        let golden: std::collections::HashSet<u64> = golden
            .iter()
            .take(k)
            .map(|candidate| candidate.id)
            .collect();
        for candidate in found.iter().take(k) {
            if golden.contains(&candidate.id) {
                hits += 1;
            }
        }
        total += k;
    }
    hits as f64 / total as f64
}

/// Mean reciprocal rank of the first correct hit, per query.
///
/// Complements [`recall`]: recall@k says nothing about ordering within the top
/// `k`, while MRR rewards putting the right answer first.
#[must_use]
pub fn mrr(truth: &[Vec<Candidate>], results: &[Vec<Candidate>]) -> f64 {
    assert_eq!(truth.len(), results.len(), "query counts must match");
    if truth.is_empty() {
        return 0.0;
    }
    let mut total = 0.0f64;
    for (golden, found) in truth.iter().zip(results) {
        let Some(best) = golden.first() else {
            continue;
        };
        if let Some(rank) = found.iter().position(|c| c.id == best.id) {
            total += 1.0 / (rank as f64 + 1.0);
        }
    }
    total / truth.len() as f64
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Rng;

    fn random_matrix(count: usize, dim: usize, seed: u64) -> Vec<f32> {
        let mut rng = Rng::new(seed);
        (0..count * dim)
            .map(|_| rng.next_f64() as f32 * 4.0 - 2.0)
            .collect()
    }

    #[test]
    fn finds_the_exact_nearest_neighbour() {
        let data = [0.0f32, 0.0, 1.0, 1.0, 5.0, 5.0];
        let hits = knn(&data, 3, 2, Metric::L2, &[1.1, 1.1], 1, None);
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].id, 1);
    }

    #[test]
    fn results_are_sorted_by_distance() {
        let data = random_matrix(500, 16, 1);
        let mut query = vec![0.0f32; 16];
        query[0] = 0.25;
        let hits = knn(&data, 500, 16, Metric::L2, &query, 20, None);
        assert_eq!(hits.len(), 20);
        for pair in hits.windows(2) {
            assert!(pair[0].distance <= pair[1].distance);
        }
    }

    #[test]
    fn self_query_returns_the_vector_itself_at_zero_distance() {
        let dim = 24;
        let data = random_matrix(64, dim, 5);
        for row in 0..64 {
            let query = &data[row * dim..(row + 1) * dim];
            let hits = knn(&data, 64, dim, Metric::L2, query, 1, None);
            assert_eq!(hits[0].id, row as u64);
            assert!(hits[0].distance.abs() < 1e-9);
        }
    }

    #[test]
    fn k_larger_than_corpus_returns_everything() {
        let data = random_matrix(3, 4, 9);
        let hits = knn(&data, 3, 4, Metric::L2, &[0.0f32; 4], 10, None);
        assert_eq!(hits.len(), 3);
    }

    #[test]
    fn zero_k_and_empty_index_return_nothing() {
        let data = random_matrix(5, 4, 11);
        assert!(knn(&data, 5, 4, Metric::L2, &[0.0f32; 4], 0, None).is_empty());
        assert!(knn(&[], 0, 4, Metric::L2, &[0.0f32; 4], 3, None).is_empty());
    }

    #[test]
    fn filter_excludes_rows() {
        let data = random_matrix(50, 8, 13);
        let query = &data[0..8];
        let allow = |row: usize| row % 2 == 0;
        let hits = knn(&data, 50, 8, Metric::L2, query, 10, Some(&allow));
        assert!(hits.iter().all(|hit| hit.id % 2 == 0));
        assert_eq!(hits.len(), 10);
    }

    #[test]
    fn filter_that_rejects_everything_returns_nothing() {
        let data = random_matrix(10, 4, 17);
        let reject = |_row: usize| false;
        let hits = knn(&data, 10, 4, Metric::L2, &[0.0f32; 4], 5, Some(&reject));
        assert!(hits.is_empty());
    }

    #[test]
    fn batch_matches_single_query_results() {
        let (count, dim, nq) = (200usize, 12usize, 5usize);
        let data = random_matrix(count, dim, 19);
        let queries = random_matrix(nq, dim, 23);
        let batch = knn_batch(&data, count, dim, Metric::L2, &queries, nq, 7);
        assert_eq!(batch.len(), nq);
        for q in 0..nq {
            let single = knn(
                &data,
                count,
                dim,
                Metric::L2,
                &queries[q * dim..(q + 1) * dim],
                7,
                None,
            );
            assert_eq!(batch[q], single);
        }
    }

    #[test]
    fn recall_is_one_for_identical_result_sets_and_zero_for_disjoint() {
        let dim = 8;
        let data = random_matrix(100, dim, 29);
        let queries = random_matrix(4, dim, 31);
        let truth = knn_batch(&data, 100, dim, Metric::L2, &queries, 4, 10);
        assert!((recall(&truth, &truth, 10) - 1.0).abs() < 1e-12);

        let disjoint: Vec<Vec<Candidate>> = truth
            .iter()
            .map(|hits| {
                hits.iter()
                    .map(|c| Candidate {
                        id: c.id + 1_000_000,
                        distance: c.distance,
                    })
                    .collect()
            })
            .collect();
        assert_eq!(recall(&truth, &disjoint, 10), 0.0);
    }

    #[test]
    fn recall_counts_partial_overlap_correctly() {
        let truth = vec![vec![
            Candidate {
                distance: 0.0,
                id: 1,
            },
            Candidate {
                distance: 0.1,
                id: 2,
            },
            Candidate {
                distance: 0.2,
                id: 3,
            },
            Candidate {
                distance: 0.3,
                id: 4,
            },
        ]];
        // Two of the four ground-truth ids recovered.
        let results = vec![vec![
            Candidate {
                distance: 0.0,
                id: 1,
            },
            Candidate {
                distance: 0.05,
                id: 99,
            },
            Candidate {
                distance: 0.15,
                id: 3,
            },
            Candidate {
                distance: 0.35,
                id: 98,
            },
        ]];
        assert!((recall(&truth, &results, 4) - 0.5).abs() < 1e-12);
    }

    #[test]
    fn mrr_uses_the_rank_of_the_first_hit() {
        let truth = vec![vec![Candidate {
            distance: 0.0,
            id: 42,
        }]];
        let first = vec![vec![Candidate {
            distance: 0.0,
            id: 42,
        }]];
        assert!((mrr(&truth, &first) - 1.0).abs() < 1e-12);

        let third = vec![vec![
            Candidate {
                distance: 0.1,
                id: 1,
            },
            Candidate {
                distance: 0.2,
                id: 2,
            },
            Candidate {
                distance: 0.3,
                id: 42,
            },
        ]];
        assert!((mrr(&truth, &third) - 1.0 / 3.0).abs() < 1e-12);

        let missing = vec![vec![Candidate {
            distance: 0.1,
            id: 7,
        }]];
        assert_eq!(mrr(&truth, &missing), 0.0);
    }

    #[test]
    fn cosine_metric_behaves_as_expected() {
        // Two unit vectors 90 degrees apart, one identical to the query.
        let mut data = vec![1.0f32, 0.0, 0.0, 1.0];
        let mut query = vec![1.0f32, 0.0];
        Metric::Cosine.prepare(&mut data[0..2]);
        Metric::Cosine.prepare(&mut data[2..4]);
        Metric::Cosine.prepare(&mut query);
        let hits = knn(&data, 2, 2, Metric::Cosine, &query, 2, None);
        assert_eq!(hits[0].id, 0);
        assert!(hits[1].distance > hits[0].distance);
    }

    #[test]
    #[should_panic(expected = "count x dim")]
    fn shape_mismatch_panics_loudly() {
        let data = vec![0.0f32; 10];
        let _ = knn(&data, 3, 4, Metric::L2, &[0.0f32; 4], 1, None);
    }
}
