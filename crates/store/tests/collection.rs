//! Collection behaviour: what a caller sees across upserts, deletes, flushes,
//! compactions and reopens.
//!
//! The segment format has its own suite (`segment.rs`); this one holds the
//! rules that span files — that the log, the manifest and the segments agree
//! — including the regressions that taught us why: a flush with nothing to
//! seal once dropped the log's delete records without persisting the
//! tombstones, and deleted ids came back after a reopen.

use lodestar_ann_core::{Candidate, Metric};
use lodestar_ann_index::hnsw::HnswConfig;
use lodestar_ann_store::Collection;

const DIM: usize = 8;

/// The vector stored for `id`: its first component is the id, so an
/// exact-vector query is only answered by that id.
fn vector_for(id: u64) -> Vec<f32> {
    let mut vector = vec![0.0f32; DIM];
    vector[0] = id as f32;
    vector[1] = (id % 7) as f32;
    vector[2] = (id % 13) as f32;
    vector[3] = 1.0;
    vector
}

/// Creates a collection with `count` vectors already sealed into a segment.
fn sealed_collection(directory: &std::path::Path, name: &str, count: u64) -> Collection {
    let mut collection =
        Collection::create(directory, name, DIM, Metric::L2, HnswConfig::default()).unwrap();
    let ids: Vec<u64> = (0..count).collect();
    let mut vectors = Vec::with_capacity(ids.len() * DIM);
    for id in &ids {
        vectors.extend(vector_for(*id));
    }
    collection.upsert_batch(&ids, &vectors).unwrap();
    collection.flush().unwrap();
    collection
}

/// The nearest neighbour for an exact stored vector: its own id.
fn nearest(collection: &Collection, id: u64) -> Option<Candidate> {
    let results = collection.search(&vector_for(id), 1, 64).unwrap();
    results.into_iter().next()
}

#[test]
fn upserts_are_searchable_before_and_after_a_flush_and_reopen() {
    let directory = tempfile::tempdir().unwrap();
    let root = directory.path();
    let collection = sealed_collection(root, "round", 200);

    // Before the flush was even a thing, the tail answers too.
    let found = nearest(&collection, 42).expect("id 42 is live");
    assert_eq!(found.id, 42);
    assert!(found.distance < 1e-6);
    drop(collection);

    // Reopen: the sealed segment is mapped, the log replayed, and the answers
    // are the same.
    let collection = Collection::open(root, "round").unwrap();
    let found = nearest(&collection, 42).expect("id 42 is live after reopen");
    assert_eq!(found.id, 42);
    assert!(found.distance < 1e-6);
    assert_eq!(collection.live_len(), 200);
    collection.verify().unwrap();
}

#[test]
fn a_flush_with_nothing_to_seal_still_persists_tombstones() {
    // The regression: delete ids that live in sealed segments, then flush.
    // The tail is empty, so there is no segment to write, and the log — the
    // only durable record of the deletes — used to be reset anyway. After a
    // reopen the deleted ids came back.
    let directory = tempfile::tempdir().unwrap();
    let root = directory.path();
    let mut collection = sealed_collection(root, "tombstones", 100);

    collection.delete_batch(&[3, 7, 91]).unwrap();
    assert_eq!(collection.live_len(), 97);
    collection.flush().unwrap(); // empty tail: the fixed path
    drop(collection);

    let collection = Collection::open(root, "tombstones").unwrap();
    assert_eq!(collection.live_len(), 97, "deleted ids came back");
    for id in [3u64, 7, 91] {
        assert!(
            nearest(&collection, id).is_none_or(|hit| hit.id != id),
            "deleted id {id} is still searchable"
        );
    }
    collection.verify().unwrap();
}

#[test]
fn tombstones_survive_a_reopen_without_a_flush() {
    // Deletes that are still only in the log must be replayed on open.
    let directory = tempfile::tempdir().unwrap();
    let root = directory.path();
    let mut collection = sealed_collection(root, "wal", 100);
    collection.delete_batch(&[5, 55]).unwrap();
    drop(collection);

    let collection = Collection::open(root, "wal").unwrap();
    assert_eq!(collection.live_len(), 98);
    for id in [5u64, 55] {
        assert!(nearest(&collection, id).is_none_or(|hit| hit.id != id));
    }
}

#[test]
fn an_upsert_replaces_an_older_copy_and_the_newest_wins() {
    let directory = tempfile::tempdir().unwrap();
    let root = directory.path();
    let mut collection = sealed_collection(root, "replace", 50);

    // Put id 9 at a different place and flush, so the old copy lives in a
    // sealed segment and the new one in the tail.
    let moved = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    collection.upsert(9, &moved).unwrap();
    collection.flush().unwrap();

    let mut results = collection.search(&moved, 1, 64).unwrap();
    assert_eq!(results.len(), 1);
    let hit = results.pop().unwrap();
    assert_eq!(hit.id, 9);
    assert!(hit.distance < 1e-6, "the new copy must win: {hit:?}");
    assert_eq!(collection.live_len(), 50, "a replacement is not a new node");
}

#[test]
fn compaction_merges_segments_and_drops_tombstones() {
    let directory = tempfile::tempdir().unwrap();
    let root = directory.path();
    let mut collection = sealed_collection(root, "compact", 120);

    // A second segment, then deletions across both.
    let ids: Vec<u64> = (120..180).collect();
    let mut vectors = Vec::with_capacity(ids.len() * DIM);
    for id in &ids {
        vectors.extend(vector_for(*id));
    }
    collection.upsert_batch(&ids, &vectors).unwrap();
    collection.flush().unwrap();
    collection.delete_batch(&[0, 60, 120]).unwrap();

    collection.compact().unwrap();
    let stats = collection.stats();
    assert_eq!(stats.segments, 1, "compaction leaves one segment");
    assert_eq!(stats.tombstoned_ids, 0, "compaction drops tombstones");
    assert_eq!(collection.live_len(), 177);
    for id in [0u64, 60, 120] {
        assert!(nearest(&collection, id).is_none_or(|hit| hit.id != id));
    }
    drop(collection);

    // And the compacted collection is exactly what a fresh open reads.
    let collection = Collection::open(root, "compact").unwrap();
    assert_eq!(collection.live_len(), 177);
    collection.verify().unwrap();
}

#[test]
fn a_torn_log_tail_is_repaired_not_fatal() {
    let directory = tempfile::tempdir().unwrap();
    let root = directory.path();
    drop(sealed_collection(root, "torn", 40));

    // Simulate a crash mid-record: half a record appended to the log.
    let wal = root.join("torn").join("wal.log");
    let mut bytes = std::fs::read(&wal).unwrap();
    let before = bytes.len();
    bytes.extend_from_slice(&[0x4c, 0x4f, 0x44, 0x57, 0x00, 0x01]);
    std::fs::write(&wal, &bytes).unwrap();

    let collection = Collection::open(root, "torn").unwrap();
    assert_eq!(collection.live_len(), 40, "the good records still replay");
    let after = std::fs::metadata(&wal).unwrap().len();
    assert!(
        after < before as u64 + 6,
        "the torn tail is truncated on repair"
    );
}

#[test]
fn a_vector_of_the_wrong_dimension_is_rejected() {
    let directory = tempfile::tempdir().unwrap();
    let mut collection = Collection::create(
        directory.path(),
        "dims",
        DIM,
        Metric::L2,
        HnswConfig::default(),
    )
    .unwrap();
    assert!(collection.upsert(1, &[1.0, 2.0]).is_err());
    assert!(collection.upsert_batch(&[1], &[1.0; DIM + 1]).is_err());
    assert!(collection.search(&[1.0; DIM - 1], 3, 16).is_err());
}
