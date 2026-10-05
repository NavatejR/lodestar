//! Recall regression gate.
//!
//! An approximate index has exactly one guarantee worth regression-testing: how
//! many of the true nearest neighbours come back. Latency is a property of the
//! machine, recall is a property of the algorithm, so every figure here is
//! deterministic (fixed seeds, fixed configuration) and therefore testable in CI.
//!
//! The tests are `#[ignore]`d so that a routine `cargo test` stays fast, and are
//! run in the release profile — `make gate` — because a debug build is around
//! twenty times slower:
//!
//! ```text
//! cargo test --release -p lodestar-ann-index --test recall_gate -- --ignored
//! ```
//!
//! ## Why the corpus looks the way it does
//!
//! Recall@10 is only a meaningful measurement when the top 10 *is* a meaningful
//! question. If the 10th and 100th nearest neighbours differ by about 1% in
//! distance, recall@10 mostly measures which side of a tie the search landed on
//! and no implementation scores well: on such a corpus with `M=16`,
//! `ef_construction=200` and `ef_search=64`, hnswlib measures 0.64 and this
//! implementation 0.67. Both exceed 0.94 on the corpus used below.
//! [`concentrated_corpus_is_still_navigable`] keeps the degenerate case in the
//! gate as a *structural* check (reachability at large `ef`) rather than as a
//! ranking threshold.
//!
//! The main corpus therefore plants structure the way real embeddings have it:
//! 40 clusters in **8 intrinsic dimensions**, mapped into 64 ambient dimensions
//! by a fixed random projection. Distances stay well-separated, so the ranking
//! is determined by the data instead of by rounding.
//!
//! Every floor below is a floor, not a target: the measured value on the
//! reference machine (Apple A18 Pro, 8 GB) sits next to it, and [`report`]
//! prints the live figures on every run so that a change is visible in review.

use lodestar_ann_core::{Candidate, Metric, Rng, brute};
use lodestar_ann_index::hnsw::{Hnsw, HnswConfig};
use lodestar_ann_index::ivfpq::{IvfPq, IvfPqConfig};

/// Ambient dimensionality, as an embedding model might produce.
const DIM: usize = 64;
/// Intrinsic dimensionality of the planted structure.
const INTRINSIC: usize = 8;
/// Number of indexed vectors.
const COUNT: usize = 20_000;
/// Number of planted clusters.
const CLUSTERS: usize = 40;
/// Neighbours returned per query.
const K: usize = 10;
/// Number of query vectors.
const QUERIES: usize = 200;
/// Seed for the indexed corpus.
const DATA_SEED: u64 = 0x10DA_7A5E_2026_0001;
/// Seed for the queries, kept separate so queries are not copies of data rows.
const QUERY_SEED: u64 = 0x10DA_7A5E_2026_0002;
/// IVF list count used by every IVF gate below.
const NLIST: usize = 64;
/// Default IVF probe count for [`NLIST`] lists.
const NPROBE: usize = 16;

/// Draws `count` vectors from the planted structure.
///
/// `spread` scales the within-cluster jitter, which is what controls how
/// separated the true neighbours are: see the module docs.
fn corpus(count: usize, seed: u64, spread: f32) -> Vec<f32> {
    let mut rng = Rng::new(seed);
    // A fixed random projection, i.i.d. normal divided by sqrt(DIM) so that
    // distances keep their scale. Random and therefore reproducible, not
    // orthonormal: a linear map with small distortion is all the structure
    // needs, and it keeps the test free of a matrix decomposition.
    let mut projection = vec![0.0f32; DIM * INTRINSIC];
    for value in projection.iter_mut() {
        let mut sum = 0.0f64;
        for _ in 0..12 {
            sum += rng.next_f64();
        }
        *value = ((sum - 6.0) / (DIM as f64).sqrt()) as f32;
    }
    let mut centres = vec![0.0f32; CLUSTERS * INTRINSIC];
    for value in centres.iter_mut() {
        *value = (rng.next_f64() as f32 - 0.5) * 10.0;
    }
    let mut data = Vec::with_capacity(count * DIM);
    for row in 0..count {
        let cluster = row % CLUSTERS;
        let mut point = [0.0f32; INTRINSIC];
        for (d, value) in point.iter_mut().enumerate() {
            *value = centres[cluster * INTRINSIC + d] + (rng.next_f64() as f32 - 0.5) * spread;
        }
        for out in 0..DIM {
            let mut sum = 0.0f32;
            for (d, &value) in point.iter().enumerate() {
                sum += projection[out * INTRINSIC + d] * value;
            }
            data.push(sum);
        }
    }
    data
}

/// Draws a corpus with no low-dimensional structure at all: every cluster
/// centre gets independent per-dimension jitter in all 64 dimensions, which is
/// what makes the 10th and 100th neighbours nearly equidistant (measured ratio
/// 1.02 on this data, against 1.33 for [`corpus`]).
fn flat_corpus(count: usize, seed: u64) -> Vec<f32> {
    let mut rng = Rng::new(seed);
    let mut centres = vec![0.0f32; CLUSTERS * DIM];
    for value in centres.iter_mut() {
        *value = (rng.next_f64() as f32 - 0.5) * 10.0;
    }
    let mut data = Vec::with_capacity(count * DIM);
    for row in 0..count {
        let cluster = row % CLUSTERS;
        for d in 0..DIM {
            data.push(centres[cluster * DIM + d] + (rng.next_f64() as f32 - 0.5) * 0.6);
        }
    }
    data
}

/// Exact ground truth for every query.
fn truth(data: &[f32], queries: &[f32], k: usize) -> Vec<Vec<Candidate>> {
    brute::knn_batch(data, COUNT, DIM, Metric::L2, queries, QUERIES, k)
}

fn query_at(queries: &[f32], index: usize) -> &[f32] {
    &queries[index * DIM..(index + 1) * DIM]
}

/// Prints a measured figure next to its floor, so a threshold change is
/// reviewable from the CI log alone.
fn report(label: &str, measured: f64, floor: f64) {
    println!("{label:<36} {measured:.4}  (floor {floor})");
}

/// Re-ranks a shortlist with exact distances, the configuration that is
/// actually served: quantization then only has to keep the right rows inside
/// the shortlist instead of ordering them.
fn rerank(data: &[f32], query: &[f32], shortlist: Vec<Candidate>, k: usize) -> Vec<Candidate> {
    let mut exact: Vec<Candidate> = shortlist
        .iter()
        .map(|hit| Candidate {
            distance: Metric::L2.distance(
                query,
                &data[hit.id as usize * DIM..(hit.id as usize + 1) * DIM],
            ),
            id: hit.id,
        })
        .collect();
    exact.sort_unstable();
    exact.truncate(k);
    exact
}

#[test]
#[ignore = "calibrated for the release profile; run with `make gate`"]
fn hnsw_recall_gate() {
    let data = corpus(COUNT, DATA_SEED, 0.6);
    let queries = corpus(QUERIES, QUERY_SEED, 0.6);
    let expected = truth(&data, &queries, K);

    // Measured: 0.9560 at the default configuration.
    let mut index = Hnsw::new(DIM, Metric::L2, HnswConfig::default()).unwrap();
    index
        .insert_batch(&(0..COUNT as u64).collect::<Vec<_>>(), &data)
        .unwrap();

    let hits: Vec<Vec<Candidate>> = (0..QUERIES)
        .map(|q| index.search(query_at(&queries, q), K).unwrap())
        .collect();
    let recall = brute::recall(&expected, &hits, K);
    report("hnsw m=16 ef_construction=200", recall, 0.90);
    assert!(
        recall >= 0.90,
        "HNSW recall@{K} fell to {recall} (floor 0.90)"
    );

    // ef_search is the knob users reach for, so it has to work in both
    // directions: widen it and recall must not fall.
    let wide: Vec<Vec<Candidate>> = (0..QUERIES)
        .map(|q| index.search_with_ef(query_at(&queries, q), K, 256).unwrap())
        .collect();
    let wide_recall = brute::recall(&expected, &wide, K);
    report("hnsw ef_search=256", wide_recall, 0.95);
    assert!(
        wide_recall >= recall,
        "raising ef_search lowered recall: {wide_recall} < {recall}"
    );
    assert!(
        wide_recall >= 0.95,
        "HNSW recall@{K} at ef_search=256 fell to {wide_recall} (floor 0.95)"
    );
}

#[test]
#[ignore = "calibrated for the release profile; run with `make gate`"]
fn ivf_recall_gate() {
    let data = corpus(COUNT, DATA_SEED, 0.6);
    let queries = corpus(QUERIES, QUERY_SEED, 0.6);
    let expected = truth(&data, &queries, K);

    // --- IVF-Flat: recall must rise with nprobe, and full probe is exact ------
    let flat = IvfPq::train(
        &data,
        COUNT,
        DIM,
        Metric::L2,
        IvfPqConfig::default()
            .with_nlist(NLIST)
            .with_nprobe(NPROBE)
            .flat(),
    )
    .unwrap();

    let mut previous = -1.0f64;
    for nprobe in [1usize, 4, NPROBE, NLIST] {
        let hits: Vec<Vec<Candidate>> = (0..QUERIES)
            .map(|q| {
                flat.search_with_probes(query_at(&queries, q), K, nprobe)
                    .unwrap()
            })
            .collect();
        let recall = brute::recall(&expected, &hits, K);
        report(
            &format!("ivf-flat nprobe={nprobe}/{NLIST}"),
            recall,
            previous,
        );
        assert!(
            recall >= previous,
            "recall fell from {previous} to {recall} at nprobe={nprobe}"
        );
        previous = recall;
    }
    // Measured: 1.0000 at full probe, which is an invariant rather than a
    // statistical claim: probing every list with exact postings is a scan.
    assert!(
        (previous - 1.0).abs() < f64::EPSILON,
        "full-probe IVF-Flat recall was {previous}, expected exactly 1.0"
    );

    let exhaustive: Vec<Vec<Candidate>> = (0..QUERIES)
        .map(|q| {
            flat.search_with_probes(query_at(&queries, q), K, usize::MAX)
                .unwrap()
        })
        .collect();
    assert_eq!(
        exhaustive, expected,
        "an exhaustive flat probe must reproduce brute force, ordering included"
    );

    // --- IVF-PQ: 8 subspaces x 256 centroids with residual encoding ----------
    let mut pq_config = IvfPqConfig::default()
        .with_nlist(NLIST)
        .with_nprobe(NPROBE)
        .with_pq(8, 256);
    pq_config.train_limit = 8_192;
    let pq = IvfPq::train(&data, COUNT, DIM, Metric::L2, pq_config).unwrap();

    let adc_hits: Vec<Vec<Candidate>> = (0..QUERIES)
        .map(|q| pq.search(query_at(&queries, q), K).unwrap())
        .collect();
    let adc_recall = brute::recall(&expected, &adc_hits, K);
    // Measured: 0.5945. This is a sanity floor rather than a quality target:
    // 8 bytes of code per vector is a 64x compression, so raw ADC ordering is
    // expected to be mediocre and the reranked figure below is the one that
    // matters.
    report("ivf-pq nprobe=16 adc", adc_recall, 0.50);
    assert!(
        adc_recall >= 0.50,
        "IVF-PQ raw ADC recall@{K} fell to {adc_recall} (floor 0.50)"
    );

    // Measured: 0.9910 after reranking a 64-candidate shortlist.
    let reranked: Vec<Vec<Candidate>> = (0..QUERIES)
        .map(|q| {
            let query = query_at(&queries, q);
            rerank(&data, query, pq.search(query, 64).unwrap(), K)
        })
        .collect();
    let reranked_recall = brute::recall(&expected, &reranked, K);
    report("ivf-pq nprobe=16 reranked", reranked_recall, 0.90);
    assert!(
        reranked_recall >= adc_recall,
        "reranking lowered recall: {reranked_recall} < {adc_recall}"
    );
    assert!(
        reranked_recall >= 0.90,
        "IVF-PQ reranked recall@{K} fell to {reranked_recall} (floor 0.90)"
    );

    // Full probe with a shortlist of the whole collection is exact by
    // construction: every row is scored, so reranking must reproduce truth.
    // Asking for `K` rows instead would only be a statistical claim.
    let exhaustive_pq: Vec<Vec<Candidate>> = (0..QUERIES)
        .map(|q| {
            let query = query_at(&queries, q);
            let shortlist = pq.search_with_probes(query, COUNT, usize::MAX).unwrap();
            assert_eq!(shortlist.len(), COUNT);
            rerank(&data, query, shortlist, K)
        })
        .collect();
    assert_eq!(
        exhaustive_pq, expected,
        "full-probe PQ reranking must reproduce brute force"
    );
}

/// The degenerate case from the module docs: a corpus whose 10th and 100th
/// neighbours are nearly equidistant, so ranking thresholds are meaningless.
/// Reachability is not meaningless, though: a graph that cannot reach a whole
/// cluster shows up as a recall collapse at large `ef`, and that is exactly the
/// failure this test exists to catch. It caught one during development, when a
/// neighbour-selection step refilled pruned slots with near-duplicates of
/// existing links and left a region of the graph unreachable.
#[test]
#[ignore = "calibrated for the release profile; run with `make gate`"]
fn concentrated_corpus_is_still_navigable() {
    let data = flat_corpus(COUNT, DATA_SEED);
    let queries = flat_corpus(QUERIES, QUERY_SEED);
    let expected = truth(&data, &queries, K);

    let mut index = Hnsw::new(DIM, Metric::L2, HnswConfig::default()).unwrap();
    index
        .insert_batch(&(0..COUNT as u64).collect::<Vec<_>>(), &data)
        .unwrap();

    // Reported, not asserted: at the default width this corpus is a
    // tie-breaking lottery, which is the whole reason the ranking floors above
    // use the well-separated corpus. Both implementations land in the 0.6-0.9
    // band here and neither is "wrong" — the data does not determine a top 10.
    let narrow: Vec<Vec<Candidate>> = (0..QUERIES)
        .map(|q| index.search(query_at(&queries, q), K).unwrap())
        .collect();
    report(
        "hnsw concentrated ef_search=64",
        brute::recall(&expected, &narrow, K),
        0.0,
    );

    let hits: Vec<Vec<Candidate>> = (0..QUERIES)
        .map(|q| {
            index
                .search_with_ef(query_at(&queries, q), K, 1024)
                .unwrap()
        })
        .collect();
    let recall = brute::recall(&expected, &hits, K);
    report("hnsw concentrated ef_search=1024", recall, 0.90);

    // Where an unreachable region shows up: whole queries score zero, because
    // none of their true neighbours can be reached at all. Partial recall from
    // ranking noise looks completely different, so the worse tail is the
    // diagnostic that matters here.
    let mut empty = 0usize;
    let mut worst = 1.0f64;
    for (expected_row, hit_row) in expected.iter().zip(&hits) {
        let per_query = brute::recall(
            std::slice::from_ref(expected_row),
            std::slice::from_ref(hit_row),
            K,
        );
        worst = worst.min(per_query);
        if per_query == 0.0 {
            empty += 1;
        }
    }
    report("hnsw concentrated worst query", worst, 0.0);
    report("hnsw concentrated zero-recall count", empty as f64, 0.0);

    // A full-width traverse from the entry point reaches everything that is
    // reachable, so a zero-recall query left at that width is unreachable from
    // the entry point rather than merely hard to navigate to.
    let full: Vec<Vec<Candidate>> = (0..QUERIES)
        .map(|q| {
            index
                .search_with_ef(query_at(&queries, q), K, COUNT)
                .unwrap()
        })
        .collect();
    let mut unreachable = 0usize;
    for (expected_row, hit_row) in expected.iter().zip(&full) {
        if brute::recall(
            std::slice::from_ref(expected_row),
            std::slice::from_ref(hit_row),
            K,
        ) == 0.0
        {
            unreachable += 1;
        }
    }
    assert!(
        unreachable == 0,
        "{unreachable} of {QUERIES} queries cannot reach any true neighbour even \
         at ef_search={COUNT}, which means part of the graph is unreachable"
    );
    assert!(
        recall >= 0.90,
        "a wide search on a concentrated corpus fell to {recall} (floor 0.90), \
         which points at unreachable regions rather than at ranking noise"
    );
}
