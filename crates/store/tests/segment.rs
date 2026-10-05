//! Segment format tests: what a written file promises, and what happens when
//! the bytes are not what they claim to be.
//!
//! Every corruption test writes a valid file, then damages exactly one thing and
//! asserts that opening fails with a named error. A parser that panics on
//! damaged input is a denial-of-service bug in any service that reads files it
//! did not write itself, so "must not panic" is tested too, with random bytes.

use lodestar_ann_core::{Candidate, Metric, Rng};
use lodestar_ann_index::graph::GraphView;
use lodestar_ann_index::hnsw::{Hnsw, HnswConfig};
use lodestar_ann_store::mapped::Segment;
use lodestar_ann_store::segment::{self, FOOTER_LEN, HEADER_LEN};
use proptest::prelude::*;

/// Builds a small HNSW and writes it as a segment, returning both.
fn build_segment(
    directory: &std::path::Path,
    count: usize,
    dim: usize,
    seed: u64,
) -> (Hnsw, std::path::PathBuf) {
    let mut rng = Rng::new(seed);
    let data: Vec<f32> = (0..count * dim)
        .map(|_| rng.next_f64() as f32 * 2.0 - 1.0)
        .collect();
    let mut index = Hnsw::new(dim, Metric::L2, HnswConfig::default()).unwrap();
    index
        .insert_batch(&(0..count as u64).collect::<Vec<_>>(), &data)
        .unwrap();
    let path = directory.join("segment-000001.seg");
    segment::write_segment(&path, &index.parts()).unwrap();
    (index, path)
}

/// Reads a little-endian u64 from a byte slice.
fn read_u64(bytes: &[u8], at: usize) -> u64 {
    u64::from_le_bytes(bytes[at..at + 8].try_into().unwrap())
}

#[test]
fn round_trip_preserves_the_graph_and_its_answers() {
    let directory = tempfile::tempdir().unwrap();
    let (index, path) = build_segment(directory.path(), 500, 24, 7);
    let segment = Segment::open(&path).unwrap();

    assert_eq!(segment.len(), index.node_count());
    assert_eq!(segment.live_count(), index.len());
    assert_eq!(segment.dim(), 24);
    assert_eq!(segment.metric(), Metric::L2);
    assert_eq!(segment.max_level(), index.max_level());
    assert_eq!(segment.entry_point(), index.entry_point());
    for node in 0..segment.len() as u32 {
        assert_eq!(segment.external_id(node), u64::from(node));
        assert!(segment.is_live(node));
        assert_eq!(segment.node_level(node), index.node_level(node));
        assert_eq!(
            segment.neighbors(node, 0),
            index.neighbors(node, 0),
            "level-0 links for node {node}"
        );
    }
    segment.verify().unwrap();

    // The strongest statement available: the mapped index answers exactly like
    // the one it was written from, for every query, including ordering.
    let mut rng = Rng::new(11);
    for _ in 0..25 {
        let query: Vec<f32> = (0..24).map(|_| rng.next_f64() as f32 * 2.0 - 1.0).collect();
        let expected: Vec<Candidate> = index.search_with_ef(&query, 10, 64).unwrap();
        let found = segment.search(&query, 10, 64).unwrap();
        assert_eq!(found, expected);
    }
}

#[test]
fn tombstones_and_empty_segments_survive_the_round_trip() {
    let directory = tempfile::tempdir().unwrap();
    let mut index = Hnsw::new(4, Metric::L2, HnswConfig::default()).unwrap();
    index
        .insert_batch(
            &[0, 1, 2, 3],
            &[
                0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0,
            ],
        )
        .unwrap();
    assert!(index.delete(2));
    let path = directory.path().join("tombstones.seg");
    segment::write_segment(&path, &index.parts()).unwrap();
    let segment = Segment::open(&path).unwrap();
    assert_eq!(segment.len(), 4);
    assert_eq!(segment.live_count(), 3);
    assert!(!segment.is_live(2));
    segment.verify().unwrap();

    let empty = Hnsw::new(4, Metric::L2, HnswConfig::default()).unwrap();
    let empty_path = directory.path().join("empty.seg");
    segment::write_segment(&empty_path, &empty.parts()).unwrap();
    let segment = Segment::open(&empty_path).unwrap();
    assert!(segment.is_empty());
    assert_eq!(segment.entry_point(), None);
    assert!(
        segment
            .search(&[0.0, 0.0, 0.0, 0.0], 5, 16)
            .unwrap()
            .is_empty()
    );
    segment.verify().unwrap();
}

#[test]
fn header_and_footer_are_checked() {
    let directory = tempfile::tempdir().unwrap();
    let (_index, path) = build_segment(directory.path(), 50, 8, 3);
    let good = std::fs::read(&path).unwrap();

    // A flipped bit in the header is caught by its checksum.
    let mut damaged = good.clone();
    damaged[12] ^= 0xFF;
    std::fs::write(&path, &damaged).unwrap();
    let error = Segment::open(&path).unwrap_err();
    assert!(error.to_string().contains("checksum"), "{error}");
    std::fs::write(&path, &good).unwrap();

    // A flipped bit in the footer is caught by its checksum.
    let mut damaged = good.clone();
    let last = damaged.len() - FOOTER_LEN;
    damaged[last + 10] ^= 0x01;
    std::fs::write(&path, &damaged).unwrap();
    let error = Segment::open(&path).unwrap_err();
    assert!(error.to_string().contains("checksum"), "{error}");
    std::fs::write(&path, &good).unwrap();

    // A truncated file is rejected before anything is read out of it.
    std::fs::write(&path, &good[..HEADER_LEN + 8]).unwrap();
    let error = Segment::open(&path).unwrap_err();
    assert!(error.to_string().contains("shorter"), "{error}");
    std::fs::write(&path, &good).unwrap();

    // Cutting the file short and moving the footer to the new end: the footer
    // itself decodes, but the length it records no longer matches the file,
    // and that disagreement is caught before anything else is read.
    let mut truncated = good[..good.len() - 32].to_vec();
    let footer_at = good.len() - FOOTER_LEN;
    let tail = truncated.len() - FOOTER_LEN;
    truncated[tail..].copy_from_slice(&good[footer_at..]);
    std::fs::write(&path, &truncated).unwrap();
    let error = Segment::open(&path).unwrap_err();
    let message = error.to_string();
    assert!(message.contains("records"), "{message}");
}

#[test]
fn a_flipped_neighbour_is_caught_by_verify_not_by_open() {
    // This is the case the checksums exist for: the graph still *looks* correct
    // - every offset still points inside the link block - but a neighbour index
    // has been replaced, which would silently change search results.
    let directory = tempfile::tempdir().unwrap();
    let (_index, path) = build_segment(directory.path(), 200, 16, 5);
    let good = std::fs::read(&path).unwrap();
    let header = &good[..HEADER_LEN];
    let directory_at = read_u64(header, 104);
    // The first level's directory entry follows the header's directory block.
    let first = directory_at as usize;
    let links_at = read_u64(&good[first..], 8) as usize;

    let mut damaged = good.clone();
    let original = u32::from_le_bytes(damaged[links_at..links_at + 4].try_into().unwrap());
    let replacement: u32 = if original == 0 { 1 } else { 0 };
    damaged[links_at..links_at + 4].copy_from_slice(&replacement.to_le_bytes());
    std::fs::write(&path, &damaged).unwrap();

    let segment = Segment::open(&path).unwrap();
    let error = segment.verify().unwrap_err();
    assert!(error.to_string().contains("checksum"), "{error}");
    // And the structural checks are documented as not catching this one, so the
    // test says so rather than pretending otherwise.
    assert!(segment.len() == 200);
}

#[test]
fn structural_damage_is_rejected_at_open() {
    let directory = tempfile::tempdir().unwrap();
    let (_index, path) = build_segment(directory.path(), 60, 8, 6);
    let good = std::fs::read(&path).unwrap();
    let header = &good[..HEADER_LEN];
    let levels_at = read_u64(header, 72) as usize;

    // A node level above the header's maximum is rejected.
    let mut damaged = good.clone();
    damaged[levels_at] = 200;
    // Keep the header checksum valid: rewrite it after the change, otherwise the
    // test would only prove that the checksum works, which another test covers.
    write_header_checksum(&mut damaged);
    std::fs::write(&path, &damaged).unwrap();
    let error = Segment::open(&path).unwrap_err();
    let message = error.to_string();
    assert!(
        message.contains("level") || message.contains("links"),
        "{message}"
    );
    std::fs::write(&path, &good).unwrap();

    // A dimensionality that does not match the vector block is rejected.
    let mut damaged = good.clone();
    damaged[12..16].copy_from_slice(&999u32.to_le_bytes());
    write_header_checksum(&mut damaged);
    std::fs::write(&path, &damaged).unwrap();
    let error = Segment::open(&path).unwrap_err();
    assert!(error.to_string().contains("vector block"), "{error}");
}

/// Re-stamps the checksums after a test has edited header fields: the
/// header's own CRC, the footer's copy of it, and the footer's self-CRC.
/// Without the footer stamps, opening would stop at "footer does not describe
/// this header" before the test's structural damage is ever examined.
fn write_header_checksum(bytes: &mut [u8]) {
    let crc = segment::crc32(&bytes[..HEADER_LEN - 4]);
    bytes[HEADER_LEN - 4..HEADER_LEN].copy_from_slice(&crc.to_le_bytes());
    let footer_at = bytes.len() - FOOTER_LEN;
    bytes[footer_at + 40..footer_at + 44].copy_from_slice(&crc.to_le_bytes());
    let footer_crc = segment::crc32(&bytes[footer_at..footer_at + FOOTER_LEN - 4]);
    bytes[footer_at + FOOTER_LEN - 4..footer_at + FOOTER_LEN]
        .copy_from_slice(&footer_crc.to_le_bytes());
}

#[test]
fn a_newer_format_version_is_reported_not_guessed() {
    let directory = tempfile::tempdir().unwrap();
    let (_index, path) = build_segment(directory.path(), 10, 4, 1);
    let mut damaged = std::fs::read(&path).unwrap();
    damaged[8..12].copy_from_slice(&99u32.to_le_bytes());
    write_header_checksum(&mut damaged);
    std::fs::write(&path, &damaged).unwrap();
    let error = Segment::open(&path).unwrap_err();
    let message = error.to_string();
    assert!(
        message.contains("unsupported format version 99"),
        "{message}"
    );
}

proptest! {
    /// Random bytes must never panic the parser, and a parse that succeeds must
    /// survive the checksum pass.
    #[test]
    fn arbitrary_bytes_never_panic(bytes in proptest::collection::vec(any::<u8>(), 0..4096)) {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("random.seg");
        std::fs::write(&path, &bytes).unwrap();
        if let Ok(segment) = Segment::open(&path) {
            let _ = segment.verify();
            let _ = segment.search(&vec![0.0; segment.dim()], 3, 16);
            let _ = segment.info();
        }
    }

    /// A valid segment with an arbitrary single-byte flip must never panic
    /// either: it either opens or reports an error, and never reads out of
    /// bounds.
    #[test]
    fn single_byte_corruption_never_panics(
        offset in any::<prop::sample::Index>(),
        value in any::<u8>(),
    ) {
        let directory = tempfile::tempdir().unwrap();
        let (_index, path) = build_segment(directory.path(), 40, 8, 9);
        let mut bytes = std::fs::read(&path).unwrap();
        let at = offset.index(bytes.len());
        bytes[at] ^= value;
        std::fs::write(&path, &bytes).unwrap();
        if let Ok(segment) = Segment::open(&path) {
            let _ = segment.verify();
            let _ = segment.search(&vec![0.0; segment.dim()], 3, 16);
        }
    }
}
