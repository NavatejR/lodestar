//! Fuzz the write-ahead log reader.
//!
//! `read_all` must reject complete records with bad checksums and stop at a
//! torn tail; `replay_and_repair` must additionally truncate that tail
//! without damaging the valid prefix. Both are fed arbitrary bytes at an
//! arbitrary dimensionality, which is the situation after a crash mid-write
//! — or after a bit flip on disk.

#![no_main]

use std::sync::atomic::{AtomicU64, Ordering};

use libfuzzer_sys::fuzz_target;
use lodestar_ann_store::wal;

static NEXT_FILE: AtomicU64 = AtomicU64::new(0);

fuzz_target!(|data: &[u8]| {
    // The dimensionality comes from the input so one corpus entry exercises
    // many header/body layouts: every dim in 1..=8 has to fail cleanly when
    // the byte count cannot belong to it.
    let dim = 1 + (data.first().copied().unwrap_or(0) as usize % 8);
    let path = std::env::temp_dir().join(format!(
        "lodestar-fuzz-wal-{}-{}.log",
        std::process::id(),
        NEXT_FILE.fetch_add(1, Ordering::Relaxed)
    ));
    if std::fs::write(&path, data).is_err() {
        return;
    }
    let _ = wal::read_all(&path, dim);
    let _ = wal::replay_and_repair(&path, dim);
    let _ = std::fs::remove_file(&path);
});
