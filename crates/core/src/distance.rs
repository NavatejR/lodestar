//! Distance kernels with runtime CPU feature dispatch.
//!
//! Three implementations exist for each of the two base operations (squared L2
//! and dot product):
//!
//! * `*_scalar` — portable, four-way unrolled, auto-vectorised by LLVM.
//! * `*_neon` — hand-written NEON, used on `aarch64` where NEON is guaranteed.
//! * `*_avx2` — hand-written AVX2 + FMA, selected at runtime on `x86_64` only
//!   when the CPU reports both features.
//!
//! Dispatch happens once per process through a [`std::sync::OnceLock`], so the
//! hot path pays a single atomic load and an indirect call rather than a
//! `cpuid` query. [`active_kernel`] reports which implementation is live; the
//! HTTP API exposes it on `/v1/stats` and the benchmark report records it, so a
//! published number can always be traced to the code path that produced it.
//!
//! ## Floating-point accuracy
//!
//! SIMD implementations accumulate four or eight lanes in parallel, so their
//! summation order differs from the scalar version. Results agree to within a
//! few ULPs (tests assert `1e-4` relative error) which is far below the margin
//! that could change a nearest-neighbour ranking, but it does mean that
//! distances are not bit-identical between code paths. Recall numbers published
//! by this project are always measured with a single, disclosed kernel.

use std::sync::OnceLock;

/// Signature of a distance kernel.
///
/// Implementations may read fewer elements than the shorter slice contains, but
/// must never read past the end of either operand.
pub type Kernel = fn(&[f32], &[f32]) -> f32;

/// Which SIMD implementation the process selected.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KernelKind {
    /// Portable scalar code, used when no SIMD path is available.
    Scalar,
    /// Hand-written NEON (aarch64).
    Neon,
    /// Hand-written AVX2 + FMA (x86_64).
    Avx2,
}

impl KernelKind {
    /// Lower-case name used in logs, JSON output and the benchmark report.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Scalar => "scalar",
            Self::Neon => "neon",
            Self::Avx2 => "avx2",
        }
    }
}

impl std::fmt::Display for KernelKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

// ---------------------------------------------------------------------------
// Portable scalar kernels
// ---------------------------------------------------------------------------

/// Squared euclidean distance, portable implementation.
///
/// Operands are read up to `min(a.len(), b.len())` elements.
#[must_use]
pub fn l2_squared_scalar(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len());
    let mut acc = [0.0f32; 4];
    let chunks = n / 4;
    for i in 0..chunks {
        let base = i * 4;
        for k in 0..4 {
            let d = a[base + k] - b[base + k];
            acc[k] += d * d;
        }
    }
    let mut sum = (acc[0] + acc[1]) + (acc[2] + acc[3]);
    for i in (chunks * 4)..n {
        let d = a[i] - b[i];
        sum += d * d;
    }
    sum
}

/// Inner (dot) product, portable implementation.
#[must_use]
pub fn inner_product_scalar(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len());
    let mut acc = [0.0f32; 4];
    let chunks = n / 4;
    for i in 0..chunks {
        let base = i * 4;
        for k in 0..4 {
            acc[k] += a[base + k] * b[base + k];
        }
    }
    let mut sum = (acc[0] + acc[1]) + (acc[2] + acc[3]);
    for i in (chunks * 4)..n {
        sum += a[i] * b[i];
    }
    sum
}

// ---------------------------------------------------------------------------
// aarch64 / NEON
// ---------------------------------------------------------------------------

/// Squared euclidean distance using NEON.
///
/// NEON is part of the aarch64 baseline ABI, so no `target_feature` gate or
/// runtime probe is required on this architecture.
#[cfg(target_arch = "aarch64")]
#[must_use]
pub fn l2_squared_neon(a: &[f32], b: &[f32]) -> f32 {
    // SAFETY: NEON is guaranteed on aarch64. Every pointer offset stays below
    // `chunks * 4 <= n`, and `n` is bounded by the length of both slices.
    unsafe {
        use core::arch::aarch64::{vaddvq_f32, vdupq_n_f32, vfmaq_f32, vld1q_f32, vsubq_f32};
        let n = a.len().min(b.len());
        let chunks = n / 4;
        let ap = a.as_ptr();
        let bp = b.as_ptr();
        let mut acc = vdupq_n_f32(0.0);
        for i in 0..chunks {
            let va = vld1q_f32(ap.add(i * 4));
            let vb = vld1q_f32(bp.add(i * 4));
            let d = vsubq_f32(va, vb);
            acc = vfmaq_f32(acc, d, d);
        }
        let mut sum = vaddvq_f32(acc);
        for i in (chunks * 4)..n {
            let d = a[i] - b[i];
            sum += d * d;
        }
        sum
    }
}

/// Inner product using NEON.
#[cfg(target_arch = "aarch64")]
#[must_use]
pub fn inner_product_neon(a: &[f32], b: &[f32]) -> f32 {
    // SAFETY: as above — NEON is baseline on aarch64 and offsets are in range.
    unsafe {
        use core::arch::aarch64::{vaddvq_f32, vdupq_n_f32, vfmaq_f32, vld1q_f32};
        let n = a.len().min(b.len());
        let chunks = n / 4;
        let ap = a.as_ptr();
        let bp = b.as_ptr();
        let mut acc = vdupq_n_f32(0.0);
        for i in 0..chunks {
            let va = vld1q_f32(ap.add(i * 4));
            let vb = vld1q_f32(bp.add(i * 4));
            acc = vfmaq_f32(acc, va, vb);
        }
        let mut sum = vaddvq_f32(acc);
        for i in (chunks * 4)..n {
            sum += a[i] * b[i];
        }
        sum
    }
}

// ---------------------------------------------------------------------------
// x86_64 / AVX2 + FMA
// ---------------------------------------------------------------------------

/// Squared euclidean distance using AVX2 and FMA.
///
/// # Safety
///
/// The caller must ensure the CPU supports both `avx2` and `fma`. In practice
/// this is enforced by [`detect`] before the function pointer is published.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn l2_squared_avx2_impl(a: &[f32], b: &[f32]) -> f32 {
    use core::arch::x86_64::{
        _mm256_fmadd_ps, _mm256_loadu_ps, _mm256_setzero_ps, _mm256_storeu_ps, _mm256_sub_ps,
    };
    // SAFETY: edition 2024 requires explicit unsafe blocks even inside an
    // `unsafe fn`. Everything here is safe under the function's documented
    // contract: the caller guaranteed AVX2 + FMA support, and every pointer
    // offset stays below `chunks * 8 <= n`, which is bounded by both slice
    // lengths.
    unsafe {
        let n = a.len().min(b.len());
        let chunks = n / 8;
        let ap = a.as_ptr();
        let bp = b.as_ptr();
        let mut acc = _mm256_setzero_ps();
        for i in 0..chunks {
            let va = _mm256_loadu_ps(ap.add(i * 8));
            let vb = _mm256_loadu_ps(bp.add(i * 8));
            let d = _mm256_sub_ps(va, vb);
            acc = _mm256_fmadd_ps(d, d, acc);
        }
        // Horizontal sum via memory: simpler than a shuffle chain and the epilogue
        // runs once per query, not per dimension.
        let mut lanes = [0.0f32; 8];
        _mm256_storeu_ps(lanes.as_mut_ptr(), acc);
        let mut sum: f32 = lanes.iter().sum();
        for i in (chunks * 8)..n {
            let d = a[i] - b[i];
            sum += d * d;
        }
        sum
    }
}

/// Inner product using AVX2 and FMA.
///
/// # Safety
///
/// The caller must ensure the CPU supports both `avx2` and `fma`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn inner_product_avx2_impl(a: &[f32], b: &[f32]) -> f32 {
    use core::arch::x86_64::{
        _mm256_fmadd_ps, _mm256_loadu_ps, _mm256_setzero_ps, _mm256_storeu_ps,
    };
    // SAFETY: edition 2024 requires explicit unsafe blocks even inside an
    // `unsafe fn`. Same contract as the L2 kernel above: AVX2 + FMA
    // guaranteed by the caller, every offset in bounds.
    unsafe {
        let n = a.len().min(b.len());
        let chunks = n / 8;
        let ap = a.as_ptr();
        let bp = b.as_ptr();
        let mut acc = _mm256_setzero_ps();
        for i in 0..chunks {
            let va = _mm256_loadu_ps(ap.add(i * 8));
            let vb = _mm256_loadu_ps(bp.add(i * 8));
            acc = _mm256_fmadd_ps(va, vb, acc);
        }
        let mut lanes = [0.0f32; 8];
        _mm256_storeu_ps(lanes.as_mut_ptr(), acc);
        let mut sum: f32 = lanes.iter().sum();
        for i in (chunks * 8)..n {
            sum += a[i] * b[i];
        }
        sum
    }
}

/// Safe wrapper around the AVX2 L2 kernel, used as a `fn` pointer.
#[cfg(target_arch = "x86_64")]
fn l2_squared_avx2(a: &[f32], b: &[f32]) -> f32 {
    // SAFETY: only reachable when `detect` reported AVX2 + FMA support.
    unsafe { l2_squared_avx2_impl(a, b) }
}

/// Safe wrapper around the AVX2 dot-product kernel.
#[cfg(target_arch = "x86_64")]
fn inner_product_avx2(a: &[f32], b: &[f32]) -> f32 {
    // SAFETY: only reachable when `detect` reported AVX2 + FMA support.
    unsafe { inner_product_avx2_impl(a, b) }
}

// ---------------------------------------------------------------------------
// Dispatch
// ---------------------------------------------------------------------------

static L2_KERNEL: OnceLock<Kernel> = OnceLock::new();
static DOT_KERNEL: OnceLock<Kernel> = OnceLock::new();
static KIND: OnceLock<KernelKind> = OnceLock::new();

#[cfg(target_arch = "aarch64")]
fn detect() -> KernelKind {
    KernelKind::Neon
}

#[cfg(target_arch = "x86_64")]
fn detect() -> KernelKind {
    if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
        KernelKind::Avx2
    } else {
        KernelKind::Scalar
    }
}

#[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
fn detect() -> KernelKind {
    KernelKind::Scalar
}

/// Reports which SIMD implementation this process is using.
#[must_use]
pub fn active_kernel() -> KernelKind {
    *KIND.get_or_init(detect)
}

fn select_l2(kind: KernelKind) -> Kernel {
    match kind {
        #[cfg(target_arch = "aarch64")]
        KernelKind::Neon => l2_squared_neon,
        #[cfg(target_arch = "x86_64")]
        KernelKind::Avx2 => l2_squared_avx2,
        // `Scalar`, plus any variant that cannot be produced on this
        // architecture: `detect` returns only `Neon` on aarch64 and only
        // `Avx2`/`Scalar` on x86_64.
        _ => l2_squared_scalar,
    }
}

fn select_dot(kind: KernelKind) -> Kernel {
    match kind {
        #[cfg(target_arch = "aarch64")]
        KernelKind::Neon => inner_product_neon,
        #[cfg(target_arch = "x86_64")]
        KernelKind::Avx2 => inner_product_avx2,
        // See `select_l2` for why the fallback is a wildcard.
        _ => inner_product_scalar,
    }
}

/// Squared euclidean distance using the best kernel available on this CPU.
#[inline]
#[must_use]
pub fn l2_squared(a: &[f32], b: &[f32]) -> f32 {
    let kernel = *L2_KERNEL.get_or_init(|| select_l2(active_kernel()));
    kernel(a, b)
}

/// Inner product using the best kernel available on this CPU.
#[inline]
#[must_use]
pub fn inner_product(a: &[f32], b: &[f32]) -> f32 {
    let kernel = *DOT_KERNEL.get_or_init(|| select_dot(active_kernel()));
    kernel(a, b)
}

/// Distance under `metric` between two *prepared* vectors.
///
/// See [`crate::Metric::prepare`] for what "prepared" means.
#[inline]
#[must_use]
pub fn distance(metric: crate::Metric, a: &[f32], b: &[f32]) -> f32 {
    metric.distance(a, b)
}

/// Computes squared-L2 distance from one query to a contiguous block of
/// `out.len()` vectors of width `dim`, writing the results into `out`.
///
/// # Panics
///
/// Panics if `data.len() != out.len() * dim`.
pub fn l2_squared_batch(query: &[f32], data: &[f32], dim: usize, out: &mut [f32]) {
    assert_eq!(
        data.len(),
        out.len() * dim,
        "batch data length must equal rows * dim"
    );
    assert_eq!(
        query.len(),
        dim,
        "query length must equal the declared dimension"
    );
    for (row, slot) in data.chunks_exact(dim).zip(out.iter_mut()) {
        *slot = l2_squared(query, row);
    }
}

/// Computes inner product from one query to a contiguous block of vectors.
///
/// # Panics
///
/// Panics if `data.len() != out.len() * dim`.
pub fn inner_product_batch(query: &[f32], data: &[f32], dim: usize, out: &mut [f32]) {
    assert_eq!(
        data.len(),
        out.len() * dim,
        "batch data length must equal rows * dim"
    );
    for (row, slot) in data.chunks_exact(dim).zip(out.iter_mut()) {
        *slot = inner_product(query, row);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Rng;

    fn random_vectors(dim: usize, count: usize, seed: u64) -> (Vec<f32>, Vec<f32>) {
        let mut rng = Rng::new(seed);
        let a: Vec<f32> = (0..count * dim)
            .map(|_| rng.next_f64() as f32 * 2.0 - 1.0)
            .collect();
        let b: Vec<f32> = (0..count * dim)
            .map(|_| rng.next_f64() as f32 * 2.0 - 1.0)
            .collect();
        (a, b)
    }

    #[test]
    fn scalar_kernel_matches_naive_reference() {
        // One vector of dimension 13: 13 is deliberately not a multiple of the
        // 4-lane unroll factor, so the scalar tail loop is exercised too.
        let (a, b) = random_vectors(13, 1, 1);
        assert_eq!(a.len(), 13);
        let expected: f32 = a.iter().zip(&b).map(|(x, y)| (x - y) * (x - y)).sum();
        let got = l2_squared_scalar(&a, &b);
        assert!((got - expected).abs() < 1e-4 * expected.max(1.0));
    }

    #[test]
    fn dispatched_kernel_matches_scalar_across_dimensions() {
        // Dimensions deliberately include values that are not multiples of the
        // 4-lane (NEON) and 8-lane (AVX2) unroll factors.
        for dim in [1usize, 2, 3, 4, 7, 8, 15, 16, 17, 31, 64, 100, 128, 384] {
            let (a, b) = random_vectors(dim, 3, dim as u64 + 5);
            for i in 0..3 {
                let av = &a[i * dim..(i + 1) * dim];
                let bv = &b[i * dim..(i + 1) * dim];
                let scalar_l2 = l2_squared_scalar(av, bv);
                let fast_l2 = l2_squared(av, bv);
                assert!(
                    (scalar_l2 - fast_l2).abs() <= 1e-4 * scalar_l2.max(1.0),
                    "dim={dim} scalar={scalar_l2} fast={fast_l2}"
                );

                let scalar_dot = inner_product_scalar(av, bv);
                let fast_dot = inner_product(av, bv);
                assert!(
                    (scalar_dot - fast_dot).abs() <= 1e-4 * scalar_dot.abs().max(1.0),
                    "dim={dim} scalar={scalar_dot} fast={fast_dot}"
                );
            }
        }
    }

    #[test]
    fn identical_vectors_have_zero_l2_distance() {
        let v: Vec<f32> = (0..128).map(|i| i as f32 * 0.25).collect();
        assert_eq!(l2_squared(&v, &v), 0.0);
        assert_eq!(l2_squared_scalar(&v, &v), 0.0);
    }

    #[test]
    fn mismatched_lengths_do_not_panic() {
        let a = [1.0f32, 2.0, 3.0];
        let b = [1.0f32];
        // Documented behaviour: only the overlapping prefix is read.
        assert_eq!(l2_squared(&a, &b), 0.0);
        assert_eq!(inner_product(&a, &b), 1.0);
    }

    #[test]
    fn empty_input_is_zero() {
        assert_eq!(l2_squared(&[], &[]), 0.0);
        assert_eq!(inner_product(&[], &[]), 0.0);
    }

    #[test]
    fn batch_matches_single_vector_calls() {
        let dim = 96;
        let (query, data) = random_vectors(dim, 25, 77);
        let query = &query[..dim];
        let mut out = vec![0.0f32; 25];
        l2_squared_batch(query, &data, dim, &mut out);
        for (i, value) in out.iter().enumerate() {
            let expected = l2_squared(query, &data[i * dim..(i + 1) * dim]);
            assert!((value - expected).abs() < 1e-4 * expected.max(1.0));
        }

        let mut ip = vec![0.0f32; 25];
        inner_product_batch(query, &data, dim, &mut ip);
        for (i, value) in ip.iter().enumerate() {
            let expected = inner_product(query, &data[i * dim..(i + 1) * dim]);
            assert!((value - expected).abs() < 1e-4 * expected.abs().max(1.0));
        }
    }

    #[test]
    #[should_panic(expected = "rows * dim")]
    fn batch_rejects_inconsistent_shapes() {
        let query = vec![0.0f32; 4];
        let data = vec![0.0f32; 12];
        let mut out = vec![0.0f32; 2];
        l2_squared_batch(&query, &data, 4, &mut out);
    }

    #[test]
    fn kernel_reporting_is_consistent_with_architecture() {
        let kind = active_kernel();
        #[cfg(target_arch = "aarch64")]
        assert_eq!(kind, KernelKind::Neon);
        #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
        assert_eq!(kind, KernelKind::Scalar);
        assert!(!kind.as_str().is_empty());
    }

    #[test]
    fn distance_respects_metric_semantics() {
        let mut a = [1.0f32, 0.0];
        let mut b = [0.0f32, 1.0];
        assert!((distance(crate::Metric::L2, &a, &b) - 2.0).abs() < 1e-6);
        crate::Metric::Cosine.prepare(&mut a);
        crate::Metric::Cosine.prepare(&mut b);
        assert!((distance(crate::Metric::Cosine, &a, &b) - 1.0).abs() < 1e-6);
        assert!((distance(crate::Metric::InnerProduct, &a, &b)).abs() < 1e-6);
    }
}
