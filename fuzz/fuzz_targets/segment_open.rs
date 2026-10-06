//! Fuzz the segment reader: everything that happens between "bytes on disk"
//! and "a mapped, validated `Segment`".
//!
//! The interesting property is that `Segment::open` promises to refuse any
//! file whose structure it cannot prove, so the mapping handed to callers
//! never yields an out-of-bounds read. A fuzzer that throws arbitrary bytes
//! at `open` + `verify` is exactly the test that promise needs.

#![no_main]

use std::sync::atomic::{AtomicU64, Ordering};

use libfuzzer_sys::fuzz_target;
use lodestar_ann_store::Segment;

static NEXT_FILE: AtomicU64 = AtomicU64::new(0);

fuzz_target!(|data: &[u8]| {
    // A unique scratch file per iteration: libFuzzer drives one input at a
    // time today, but the counter keeps the targets safe under -jobs > 1.
    let path = std::env::temp_dir().join(format!(
        "lodestar-fuzz-segment-{}-{}.seg",
        std::process::id(),
        NEXT_FILE.fetch_add(1, Ordering::Relaxed)
    ));
    if std::fs::write(&path, data).is_err() {
        return;
    }
    if let Ok(segment) = Segment::open(&path) {
        // Structural validation passed; exercise every read-only accessor so
        // a header that validates but misreports its own bounds shows up here
        // rather than in production.
        let _ = segment.verify();
        let _ = segment.dim();
        let _ = segment.metric();
        let _ = segment.len();
        let _ = segment.live_count();
        let _ = segment.config();
        let _ = segment.mapped_bytes();
    }
    let _ = std::fs::remove_file(&path);
});
