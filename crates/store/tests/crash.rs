//! Crash consistency under `SIGKILL`.
//!
//! The durability rules in the crate documentation are promises about what a
//! process that dies mid-write leaves behind. They can only be checked by
//! killing a writer at arbitrary points and reopening what it left, which is
//! what this test does: it spawns a child process that writes and flushes in a
//! loop, kills it with a hard signal at a randomised moment, and then requires
//! that every write the child acknowledged is still searchable and that the
//! whole collection still passes the checksum pass.
//!
//! The child is this same test binary running an `#[ignore]`d test, which is
//! the cheapest way to get a cooperating writer without a second build target.
//! Kill points are drawn from a seeded RNG, so a failure reproduces with the
//! same delays on rerun.

use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

use lodestar_ann_core::{Metric, Rng};
use lodestar_ann_index::hnsw::HnswConfig;
use lodestar_ann_store::Collection;

/// Environment variable that tells the child which directory to write in.
const CRASH_DIR_ENV: &str = "LODESTAR_CRASH_DIR";
/// Dimensionality used by both sides.
const DIM: usize = 8;
/// Batch size the child writes before acknowledging.
const BATCH: usize = 32;

/// The vector stored for `id`, computed identically on both sides.
///
/// The first component is the id itself, so two different ids never share a
/// vector and an exact-vector query can only be answered by that id.
fn vector_for(id: u64) -> Vec<f32> {
    vec![
        id as f32,
        (id % 7) as f32,
        (id % 13) as f32,
        (id % 101) as f32,
        1.0,
        2.0,
        3.0,
        4.0,
    ]
}

#[test]
fn sigkill_during_writes_never_loses_an_acknowledged_write() {
    for round in 0..3u64 {
        let workspace = tempfile::tempdir().unwrap();
        let root = workspace.path().to_path_buf();

        let mut child = spawn_writer(&root);
        wait_until_ready(&root, &mut child);

        // Kill after a randomised delay: early runs hit the write-ahead log,
        // later ones hit flushes and compactions mid-flight.
        let mut rng = Rng::new(0x5EED_0000 + round);
        let delay: f64 = 60.0 + rng.next_f64() * 440.0;
        std::thread::sleep(Duration::from_millis(delay as u64));
        child.kill().expect("child is still running");
        child.wait().unwrap();

        let acknowledged = acknowledged_ids(&root);
        assert!(
            acknowledged.len() >= BATCH,
            "round {round}: the child acknowledged {} writes, so the kill \
             landed before any real work and the round proves nothing",
            acknowledged.len()
        );

        // The reopen is the test: it must succeed, it must pass the checksum
        // pass, and every acknowledged write must be searchable with its exact
        // vector back.
        let mut collection = Collection::open(&root, "crash")
            .unwrap_or_else(|error| panic!("round {round}: reopen failed: {error}"));
        collection
            .verify()
            .unwrap_or_else(|error| panic!("round {round}: verify failed: {error}"));

        for id in &acknowledged {
            let found = collection
                .search(&vector_for(*id), 1, 64)
                .unwrap_or_else(|error| panic!("round {round}: search failed: {error}"));
            assert!(
                !found.is_empty() && found[0].id == *id && found[0].distance < 1e-6,
                "round {round}: acknowledged write {id} is not searchable after \
                 the kill; found {found:?}"
            );
        }

        // And the reopened collection keeps working: new writes land and are
        // immediately durable by the same rules.
        let next_id = *acknowledged.iter().max().unwrap() + 1;
        collection
            .upsert(next_id, &vector_for(next_id))
            .unwrap_or_else(|error| panic!("round {round}: post-crash write failed: {error}"));
        let found = collection.search(&vector_for(next_id), 1, 64).unwrap();
        assert_eq!(found[0].id, next_id);
        drop(collection);
    }
}

/// Spawns this test binary as a writer child.
fn spawn_writer(root: &Path) -> Child {
    let exe = std::env::current_exe().expect("test binary path");
    Command::new(exe)
        .args(["--exact", "crash_writer_child", "--ignored", "--nocapture"])
        .env(CRASH_DIR_ENV, root)
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .expect("spawn the writer child")
}

/// Waits until the child has flushed its first batch, bounded so a hung child
/// fails the test instead of hanging it.
fn wait_until_ready(root: &Path, child: &mut Child) {
    let marker = ready_marker(root);
    let deadline = Instant::now() + Duration::from_secs(20);
    while !marker.exists() {
        if Instant::now() > deadline {
            let _ = child.kill();
            panic!("writer child never became ready");
        }
        std::thread::sleep(Duration::from_millis(5));
    }
}

/// Marker file path: inside the workspace but outside the collection, so the
/// child can create it without touching anything the reopen reads.
fn ready_marker(root: &Path) -> PathBuf {
    root.join("writer-ready")
}

/// The ids the child acknowledged, read from its log.
///
/// The log is a plain append-only file written after each acknowledged call,
/// so a kill can tear its final line but no complete line can precede the
/// write it records. Only complete lines are trusted.
fn acknowledged_ids(root: &Path) -> Vec<u64> {
    let bytes = std::fs::read(root.join("acknowledged.log")).unwrap_or_default();
    let text = String::from_utf8_lossy(&bytes);
    let mut ids = Vec::new();
    for line in text.lines() {
        if let Ok(id) = line.trim().parse::<u64>() {
            ids.push(id);
        }
    }
    ids
}

/// The writer child: writes forever until it is killed.
///
/// Runs as `cargo test -- --exact crash_writer_child --ignored` with
/// [`CRASH_DIR_ENV`] set; without the variable it does nothing, so running the
/// ignored tests by hand stays harmless.
#[test]
#[ignore = "child writer, spawned by the crash test in this file"]
fn crash_writer_child() {
    let Some(root) = std::env::var_os(CRASH_DIR_ENV) else {
        return;
    };
    let root = PathBuf::from(root);
    let mut collection =
        Collection::open_or_create(&root, "crash", DIM, Metric::L2, HnswConfig::default())
            .expect("open the collection");
    let mut log = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(root.join("acknowledged.log"))
        .expect("open the acknowledgement log");

    let mut next_id = 0u64;
    let mut batches = 0usize;
    loop {
        let ids: Vec<u64> = (next_id..next_id + BATCH as u64).collect();
        let mut vectors = Vec::with_capacity(ids.len() * DIM);
        for id in &ids {
            vectors.extend_from_slice(&vector_for(*id));
        }
        collection
            .upsert_batch(&ids, &vectors)
            .expect("the child's writes succeed until it is killed");
        // The acknowledgement: after the collection call returned, the write
        // is durable. Record it for the parent to check.
        for id in &ids {
            writeln!(log, "{id}").expect("write the acknowledgement");
        }
        log.sync_data().expect("sync the acknowledgement");
        next_id += BATCH as u64;
        batches += 1;

        if batches % 4 == 0 {
            collection
                .flush()
                .expect("the child's flushes succeed until it is killed");
        }
        if batches == 4 {
            // One signal that the first flush is done; the parent may kill at
            // any point after this and before anything else.
            std::fs::write(root.join("writer-ready"), b"ready\n").expect("write the ready marker");
        }
        if batches % 24 == 0 {
            collection
                .compact()
                .expect("the child's compactions succeed until it is killed");
        }
    }
}
