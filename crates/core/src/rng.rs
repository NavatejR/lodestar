//! A tiny deterministic PRNG.
//!
//! Lodestar must be reproducible: a benchmark run in CI has to produce the same
//! index, and therefore the same recall curve, as a run on a laptop. That rules
//! out `thread_rng` and any dependency whose algorithm is allowed to change
//! between versions. `SplitMix64` is small, fast, and fully specified, so it is
//! implemented here and used everywhere a seed is needed.

/// Deterministic `SplitMix64` pseudo-random number generator.
///
/// ```
/// use lodestar_ann_core::Rng;
///
/// let mut a = Rng::new(42);
/// let mut b = Rng::new(42);
/// assert_eq!(a.next_u64(), b.next_u64());
/// ```
#[derive(Debug, Clone)]
pub struct Rng {
    state: u64,
}

const GOLDEN_GAMMA: u64 = 0x9E37_79B9_7F4A_7C15;

impl Rng {
    /// Creates a generator from a 64-bit seed.
    ///
    /// Any seed is valid, including `0`.
    #[must_use]
    pub const fn new(seed: u64) -> Self {
        Self {
            state: seed.wrapping_add(GOLDEN_GAMMA),
        }
    }

    /// Returns the next 64-bit value in the sequence.
    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(GOLDEN_GAMMA);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Returns a uniformly distributed `f64` in `[0, 1)`.
    pub fn next_f64(&mut self) -> f64 {
        // 53 bits of mantissa; the result is in [0, 1) because 2^53 is excluded.
        (self.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }

    /// Returns a uniformly distributed `usize` in `[0, n)`.
    ///
    /// # Panics
    ///
    /// Panics if `n` is zero.
    pub fn below(&mut self, n: usize) -> usize {
        assert!(n > 0, "Rng::below requires a non-zero bound");
        (self.next_u64() % n as u64) as usize
    }

    /// Shuffles a slice in place (Fisher–Yates).
    pub fn shuffle<T>(&mut self, slice: &mut [T]) {
        for i in (1..slice.len()).rev() {
            let j = self.below(i + 1);
            slice.swap(i, j);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stream_is_deterministic_for_a_fixed_seed() {
        let mut a = Rng::new(7);
        let mut b = Rng::new(7);
        let xs: Vec<u64> = (0..8).map(|_| a.next_u64()).collect();
        let ys: Vec<u64> = (0..8).map(|_| b.next_u64()).collect();
        assert_eq!(xs, ys);
    }

    #[test]
    fn different_seeds_diverge() {
        let mut a = Rng::new(1);
        let mut b = Rng::new(2);
        assert_ne!(a.next_u64(), b.next_u64());
    }

    #[test]
    fn floats_stay_in_unit_interval() {
        let mut rng = Rng::new(99);
        for _ in 0..10_000 {
            let x = rng.next_f64();
            assert!((0.0..1.0).contains(&x), "out of range: {x}");
        }
    }

    #[test]
    fn bounds_are_respected() {
        let mut rng = Rng::new(3);
        for _ in 0..1_000 {
            assert!(rng.below(17) < 17);
        }
    }

    #[test]
    fn shuffle_preserves_elements() {
        let mut rng = Rng::new(11);
        let mut values: Vec<u32> = (0..64).collect();
        rng.shuffle(&mut values);
        values.sort_unstable();
        assert_eq!(values, (0..64).collect::<Vec<_>>());
    }

    #[test]
    #[should_panic(expected = "non-zero bound")]
    fn zero_bound_panics() {
        let mut rng = Rng::new(0);
        rng.below(0);
    }
}
