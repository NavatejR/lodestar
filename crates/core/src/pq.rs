//! Product quantization and the k-means trainer it depends on.
//!
//! Product quantization splits a `dim`-dimensional vector into `m` contiguous
//! subspaces of width `dsub = dim / m`, learns a codebook of `ksub` centroids
//! per subspace, and stores each vector as `m` one-byte centroid indices. With
//! `m = 16` that is 16 bytes per vector instead of 64 bytes for `dim = 16`, and
//! at `m = 32`, `dim = 128` it is 32 bytes instead of 512 — a 16x reduction
//! that is what makes billion-scale indexes feasible on a single machine.
//!
//! Search uses *asymmetric* distance computation (ADC): the query stays in full
//! precision, and a lookup table of partial squared distances is precomputed
//! once per subspace, so scoring a candidate costs `m` table lookups.
//!
//! The trainer is a k-means++ seeded Lloyd's algorithm. Empty clusters are
//! re-seeded to the point currently suffering the largest quantization error,
//! which is what stops codebooks from silently collapsing when the data has
//! fewer natural clusters than `ksub`.

use rayon::prelude::*;

use crate::error::{Error, Result};
use crate::rng::Rng;

/// Maximum number of centroids that a one-byte code can address.
pub const MAX_CENTROIDS: usize = 256;

/// Trains `k` centroids on a contiguous `count x dim` matrix.
///
/// # Errors
///
/// * [`Error::InvalidParameter`] if `dim` is zero, `k` is zero, or `k > 256`.
/// * [`Error::InsufficientTrainingData`] if `count < k`.
/// * [`Error::DimensionMismatch`] if `data.len() != count * dim`.
pub fn kmeans(
    data: &[f32],
    count: usize,
    dim: usize,
    k: usize,
    iters: usize,
    seed: u64,
) -> Result<Vec<f32>> {
    if dim == 0 || k == 0 || k > MAX_CENTROIDS {
        return Err(Error::InvalidParameter {
            name: "kmeans",
            reason: format!("dim={dim} and k={k} must be non-zero with k <= {MAX_CENTROIDS}"),
        });
    }
    if count < k {
        return Err(Error::InsufficientTrainingData {
            needed: k,
            got: count,
        });
    }
    if data.len() != count * dim {
        return Err(Error::DimensionMismatch {
            expected: count * dim,
            actual: data.len(),
        });
    }

    let mut centroids = kmeans_plus_plus_init(data, count, dim, k, seed);
    let mut assignments = vec![u32::MAX; count];
    let mut point_distances = vec![f32::INFINITY; count];
    let mut previous_inertia = f64::INFINITY;

    for iteration in 0..iters {
        // --- assignment step (parallel over points) ---
        assignments
            .par_iter_mut()
            .zip(point_distances.par_iter_mut())
            .enumerate()
            .for_each(|(i, (assign, dist))| {
                let row = &data[i * dim..(i + 1) * dim];
                let (best, best_distance) = nearest_centroid(row, &centroids, dim, k);
                *assign = best as u32;
                *dist = best_distance;
            });

        // --- update step ---
        let mut sums = vec![0.0f64; k * dim];
        let mut counts = vec![0usize; k];
        for (i, &assign) in assignments.iter().enumerate() {
            let c = assign as usize;
            counts[c] += 1;
            let row = &data[i * dim..(i + 1) * dim];
            let base = c * dim;
            for (d, &value) in row.iter().enumerate() {
                sums[base + d] += f64::from(value);
            }
        }

        let inertia: f64 = point_distances.iter().map(|&d| f64::from(d)).sum();

        for (c, count) in counts.iter_mut().enumerate() {
            if *count == 0 {
                // Re-seed the empty cluster onto the worst-served point. The
                // point's distance is then cleared so that the next empty
                // cluster in this iteration selects a different one.
                if let Some(point) = worst_served_point(&mut point_distances) {
                    assignments[point] = c as u32;
                    *count = 1;
                    let base = c * dim;
                    for d in 0..dim {
                        sums[base + d] = f64::from(data[point * dim + d]);
                    }
                } else {
                    // Nothing left to re-seed with: keep the previous centroid
                    // rather than collapsing it onto the origin.
                    continue;
                }
            }
            let base = c * dim;
            let inv = 1.0 / *count as f64;
            for d in 0..dim {
                centroids[base + d] = (sums[base + d] * inv) as f32;
            }
        }

        // Convergence: stop once the inertia stops improving meaningfully. The
        // check is skipped on the first iteration, where `previous_inertia` is
        // still infinite and every comparison would report convergence.
        if iteration > 0 {
            let improvement = previous_inertia - inertia;
            if improvement.abs() <= 1e-6 * previous_inertia.abs().max(1.0) {
                break;
            }
        }
        previous_inertia = inertia;
    }

    Ok(centroids)
}

/// Chooses the point with the largest distance to its assigned centroid, and
/// marks it as consumed so that a later empty cluster in the same iteration
/// picks the next-worst point instead of duplicating this one.
fn worst_served_point(distances: &mut [f32]) -> Option<usize> {
    let mut best: Option<(usize, f32)> = None;
    for (i, &d) in distances.iter().enumerate() {
        if d.is_finite() && best.is_none_or(|(_, best_distance)| d > best_distance) {
            best = Some((i, d));
        }
    }
    if let Some((index, _)) = best {
        // -inf is never a real squared distance, so the point can never be
        // chosen again by a later empty cluster in this iteration.
        distances[index] = f32::NEG_INFINITY;
    }
    best.map(|(index, _)| index)
}

/// k-means++ seeding: first centroid uniformly at random, the rest sampled with
/// probability proportional to their squared distance from the closest
/// already-chosen centroid.
fn kmeans_plus_plus_init(data: &[f32], count: usize, dim: usize, k: usize, seed: u64) -> Vec<f32> {
    let mut rng = Rng::new(seed);
    let mut centroids = Vec::with_capacity(k * dim);
    let first = rng.below(count);
    centroids.extend_from_slice(&data[first * dim..(first + 1) * dim]);

    let mut closest = vec![f32::INFINITY; count];
    for _ in 1..k {
        let base = centroids.len() - dim;
        let mut total = 0.0f64;
        for i in 0..count {
            let row = &data[i * dim..(i + 1) * dim];
            let d = crate::distance::l2_squared(row, &centroids[base..base + dim]);
            if d < closest[i] {
                closest[i] = d;
            }
            total += f64::from(closest[i]);
        }
        let chosen = if total > 0.0 {
            // Weighted sample by squared distance.
            let target = rng.next_f64() * total;
            let mut cumulative = 0.0f64;
            let mut pick = count - 1;
            for (i, &d) in closest.iter().enumerate() {
                cumulative += f64::from(d);
                if cumulative >= target {
                    pick = i;
                    break;
                }
            }
            pick
        } else {
            // All points coincide with chosen centroids; fall back to uniform.
            rng.below(count)
        };
        centroids.extend_from_slice(&data[chosen * dim..(chosen + 1) * dim]);
    }
    centroids
}

/// Returns the index of the nearest centroid and its squared distance.
fn nearest_centroid(row: &[f32], centroids: &[f32], dim: usize, k: usize) -> (usize, f32) {
    let mut best = 0usize;
    let mut best_distance = f32::INFINITY;
    for c in 0..k {
        let d = crate::distance::l2_squared(row, &centroids[c * dim..(c + 1) * dim]);
        if d < best_distance {
            best_distance = d;
            best = c;
        }
    }
    (best, best_distance)
}

/// A product quantizer: `m` independent codebooks over `dim / m`-dimensional
/// subspaces.
#[derive(Debug, Clone, PartialEq)]
pub struct ProductQuantizer {
    dim: usize,
    m: usize,
    ksub: usize,
    dsub: usize,
    codebooks: Vec<f32>,
}

impl ProductQuantizer {
    /// Trains a product quantizer.
    ///
    /// At most `train_limit` vectors are used for training (sampled
    /// deterministically without replacement); training k-means on the full
    /// corpus is unnecessary and would dominate build time at scale.
    ///
    /// # Errors
    ///
    /// * [`Error::InvalidParameter`] if `dim` is not divisible by `m`, if `m` is
    ///   zero, or if `ksub` is zero or greater than 256.
    /// * [`Error::InsufficientTrainingData`] if fewer than `ksub` vectors are
    ///   available for training.
    // Training knobs (m, ksub, iters, seed, train_limit) are all genuinely
    // independent parameters here; grouping them into a struct would only move
    // the same argument list to the struct definition.
    #[allow(clippy::too_many_arguments)]
    pub fn train(
        vectors: &[f32],
        count: usize,
        dim: usize,
        m: usize,
        ksub: usize,
        iters: usize,
        seed: u64,
        train_limit: usize,
    ) -> Result<Self> {
        if m == 0 || dim == 0 || dim % m != 0 {
            return Err(Error::InvalidParameter {
                name: "pq.m",
                reason: format!("dim={dim} must be a non-zero multiple of m={m}"),
            });
        }
        if ksub == 0 || ksub > MAX_CENTROIDS {
            return Err(Error::InvalidParameter {
                name: "pq.ksub",
                reason: format!("ksub={ksub} must be in 1..={MAX_CENTROIDS}"),
            });
        }
        if vectors.len() != count * dim {
            return Err(Error::DimensionMismatch {
                expected: count * dim,
                actual: vectors.len(),
            });
        }

        let dsub = dim / m;
        // Decide which rows to train on.
        let limit = if train_limit == 0 {
            count
        } else {
            train_limit.min(count)
        };
        if limit < ksub {
            return Err(Error::InsufficientTrainingData {
                needed: ksub,
                got: limit,
            });
        }
        let rows: Vec<usize> = if limit == count {
            (0..count).collect()
        } else {
            let mut all: Vec<usize> = (0..count).collect();
            let mut rng = Rng::new(seed ^ 0x5EED_1234_ABCD_0001);
            rng.shuffle(&mut all);
            all.truncate(limit);
            all.sort_unstable();
            all
        };

        // Materialise each subspace once, then train the codebooks in parallel.
        let codebooks: Result<Vec<Vec<f32>>> = (0..m)
            .into_par_iter()
            .map(|s| {
                let mut subspace = Vec::with_capacity(limit * dsub);
                for &row in &rows {
                    subspace.extend_from_slice(
                        &vectors[row * dim + s * dsub..row * dim + (s + 1) * dsub],
                    );
                }
                kmeans(
                    &subspace,
                    limit,
                    dsub,
                    ksub,
                    iters,
                    seed.wrapping_add(s as u64 * 0x9E37_79B9),
                )
            })
            .collect();

        let codebooks = codebooks?;
        let flat: Vec<f32> = codebooks.into_iter().flatten().collect();
        Self::from_parts(dim, m, ksub, flat)
    }

    /// Rebuilds a quantizer from raw parts, as stored in a segment.
    ///
    /// # Errors
    ///
    /// [`Error::DimensionMismatch`] if `codebooks.len() != m * ksub * (dim / m)`.
    pub fn from_parts(dim: usize, m: usize, ksub: usize, codebooks: Vec<f32>) -> Result<Self> {
        if m == 0 || dim == 0 || dim % m != 0 {
            return Err(Error::InvalidParameter {
                name: "pq.m",
                reason: format!("dim={dim} must be a non-zero multiple of m={m}"),
            });
        }
        let dsub = dim / m;
        let expected = m * ksub * dsub;
        if codebooks.len() != expected {
            return Err(Error::DimensionMismatch {
                expected,
                actual: codebooks.len(),
            });
        }
        Ok(Self {
            dim,
            m,
            ksub,
            dsub,
            codebooks,
        })
    }

    /// Dimensionality of the vectors this quantizer encodes.
    #[must_use]
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Number of subspaces.
    #[must_use]
    pub fn m(&self) -> usize {
        self.m
    }

    /// Number of centroids per subspace.
    #[must_use]
    pub fn ksub(&self) -> usize {
        self.ksub
    }

    /// Bytes per encoded vector.
    #[must_use]
    pub fn code_len(&self) -> usize {
        self.m
    }

    /// Bytes held by the codebooks.
    #[must_use]
    pub fn codebook_bytes(&self) -> usize {
        self.codebooks.len() * std::mem::size_of::<f32>()
    }

    /// The flattened codebooks, laid out `[subspace][centroid][dsub]`.
    #[must_use]
    pub fn codebooks(&self) -> &[f32] {
        &self.codebooks
    }

    /// Encodes one vector into `out`.
    ///
    /// # Errors
    ///
    /// [`Error::DimensionMismatch`] if the operand shapes are wrong.
    pub fn encode_into(&self, vector: &[f32], out: &mut [u8]) -> Result<()> {
        if vector.len() != self.dim {
            return Err(Error::DimensionMismatch {
                expected: self.dim,
                actual: vector.len(),
            });
        }
        if out.len() < self.m {
            return Err(Error::DimensionMismatch {
                expected: self.m,
                actual: out.len(),
            });
        }
        for (s, slot) in out[..self.m].iter_mut().enumerate() {
            let sub = &vector[s * self.dsub..(s + 1) * self.dsub];
            let table = &self.codebooks[s * self.ksub * self.dsub..(s + 1) * self.ksub * self.dsub];
            let (idx, _) = nearest_centroid(sub, table, self.dsub, self.ksub);
            *slot = idx as u8;
        }
        Ok(())
    }

    /// Convenience wrapper allocating the code.
    ///
    /// # Errors
    ///
    /// As [`ProductQuantizer::encode_into`].
    pub fn encode(&self, vector: &[f32]) -> Result<Vec<u8>> {
        let mut out = vec![0u8; self.m];
        self.encode_into(vector, &mut out)?;
        Ok(out)
    }

    /// Encodes a contiguous `count x dim` matrix into `count x m` codes.
    ///
    /// # Errors
    ///
    /// [`Error::DimensionMismatch`] if the shapes do not line up.
    pub fn encode_matrix(&self, vectors: &[f32], count: usize, out: &mut [u8]) -> Result<()> {
        if vectors.len() != count * self.dim {
            return Err(Error::DimensionMismatch {
                expected: count * self.dim,
                actual: vectors.len(),
            });
        }
        if out.len() != count * self.m {
            return Err(Error::DimensionMismatch {
                expected: count * self.m,
                actual: out.len(),
            });
        }
        out.par_chunks_mut(self.m)
            .enumerate()
            .for_each(|(i, code)| {
                let row = &vectors[i * self.dim..(i + 1) * self.dim];
                // Lengths were validated above, so this cannot fail.
                let _ = self.encode_into(row, code);
            });
        Ok(())
    }

    /// Builds the ADC lookup table for a query: `m * ksub` partial distances.
    ///
    /// # Errors
    ///
    /// * [`Error::DimensionMismatch`] if the query length is wrong.
    /// * [`Error::DimensionMismatch`] if `lut.len() < m * ksub`.
    pub fn adc_lut(&self, query: &[f32], lut: &mut [f32]) -> Result<()> {
        if query.len() != self.dim {
            return Err(Error::DimensionMismatch {
                expected: self.dim,
                actual: query.len(),
            });
        }
        if lut.len() < self.m * self.ksub {
            return Err(Error::DimensionMismatch {
                expected: self.m * self.ksub,
                actual: lut.len(),
            });
        }
        let m = self.m;
        let ksub = self.ksub;
        let dsub = self.dsub;
        lut[..m * ksub]
            .par_chunks_mut(ksub)
            .enumerate()
            .for_each(|(s, table)| {
                let sub = &query[s * dsub..(s + 1) * dsub];
                for (c, slot) in table.iter_mut().enumerate() {
                    let centroid =
                        &self.codebooks[(s * ksub + c) * dsub..(s * ksub + c + 1) * dsub];
                    *slot = crate::distance::l2_squared(sub, centroid);
                }
            });
        Ok(())
    }

    /// Scores a code against a precomputed ADC lookup table.
    ///
    /// # Panics
    ///
    /// Panics if `lut.len() < m * ksub` or `code.len() < m`; callers inside this
    /// crate always satisfy that.
    #[must_use]
    pub fn distance_from_lut(&self, lut: &[f32], code: &[u8]) -> f32 {
        let mut sum = 0.0f32;
        for s in 0..self.m {
            sum += lut[s * self.ksub + code[s] as usize];
        }
        sum
    }

    /// Convenience: builds a lookup table and scores one code.
    ///
    /// # Errors
    ///
    /// As [`ProductQuantizer::adc_lut`].
    pub fn distance(&self, query: &[f32], code: &[u8]) -> Result<f32> {
        let mut lut = vec![0.0f32; self.m * self.ksub];
        self.adc_lut(query, &mut lut)?;
        Ok(self.distance_from_lut(&lut, code))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Builds `count` points around `clusters` centres on a `[-4, 4]` cube.
    ///
    /// `noise` controls the spread of the cloud around each centre, and it
    /// matters more than it looks. With a tiny spread (0.05) every point in a
    /// cluster is effectively equidistant from the query, so "the" top-10 is an
    /// arbitrary choice among ~100 tied points and recall measures quantization
    /// noise rather than ranking ability. With a wide spread the top-10 is
    /// genuinely determined by the data, which is what a ranking test must
    /// exercise.
    fn clustered_data(
        count: usize,
        dim: usize,
        clusters: usize,
        noise: f32,
        seed: u64,
    ) -> Vec<f32> {
        let mut rng = Rng::new(seed);
        let mut centers = Vec::with_capacity(clusters * dim);
        for _ in 0..clusters * dim {
            centers.push(rng.next_f64() as f32 * 8.0 - 4.0);
        }
        let mut data = Vec::with_capacity(count * dim);
        for i in 0..count {
            let c = i % clusters;
            for d in 0..dim {
                let jitter = rng.next_f64() as f32 * noise - noise / 2.0;
                data.push(centers[c * dim + d] + jitter);
            }
        }
        data
    }

    #[test]
    fn kmeans_recovers_separated_clusters() {
        let (count, dim, clusters) = (600usize, 8usize, 6usize);
        let data = clustered_data(count, dim, clusters, 0.05, 1);
        let centroids = kmeans(&data, count, dim, clusters, 25, 7).unwrap();
        assert_eq!(centroids.len(), clusters * dim);

        // Each true cluster centre must have a centroid within the noise radius.
        let mut rng = Rng::new(1);
        let mut centers = Vec::new();
        for _ in 0..clusters * dim {
            centers.push(rng.next_f64() as f32 * 8.0 - 4.0);
        }
        for c in 0..clusters {
            let target = &centers[c * dim..(c + 1) * dim];
            let (_, distance) = nearest_centroid(target, &centroids, dim, clusters);
            assert!(distance < 1e-3, "cluster {c} not recovered: {distance}");
        }
    }

    #[test]
    fn kmeans_is_deterministic() {
        let data = clustered_data(300, 6, 4, 0.05, 3);
        let a = kmeans(&data, 300, 6, 4, 15, 42).unwrap();
        let b = kmeans(&data, 300, 6, 4, 15, 42).unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn kmeans_validates_its_inputs() {
        let data = vec![0.0f32; 32];
        assert!(matches!(
            kmeans(&data, 4, 8, 0, 5, 1),
            Err(Error::InvalidParameter { .. })
        ));
        assert!(matches!(
            kmeans(&data, 4, 8, 300, 5, 1),
            Err(Error::InvalidParameter { .. })
        ));
        assert!(matches!(
            kmeans(&data, 2, 8, 5, 5, 1),
            Err(Error::InsufficientTrainingData { .. })
        ));
        assert!(matches!(
            kmeans(&data, 5, 8, 2, 5, 1),
            Err(Error::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn duplicate_points_do_not_break_training() {
        // Every point is identical, so there is exactly one natural cluster but
        // the trainer is asked for eight.
        let data = vec![1.5f32; 100 * 4];
        let centroids = kmeans(&data, 100, 4, 8, 10, 5).unwrap();
        assert_eq!(centroids.len(), 8 * 4);
        assert!(centroids.iter().all(|x| x.is_finite()));
    }

    #[test]
    fn pq_round_trip_error_is_small_on_clustered_data() {
        let (count, dim, m, ksub) = (2048usize, 32usize, 8usize, 64usize);
        let data = clustered_data(count, dim, 16, 0.05, 11);
        let pq = ProductQuantizer::train(&data, count, dim, m, ksub, 20, 9, 4096).unwrap();
        assert_eq!(pq.code_len(), m);
        assert_eq!(pq.codebook_bytes(), m * ksub * (dim / m) * 4);

        let mut total_error = 0.0f64;
        for i in 0..128 {
            let row = &data[i * dim..(i + 1) * dim];
            let code = pq.encode(row).unwrap();
            let reconstructed = pq.distance(row, &code).unwrap();
            total_error += f64::from(reconstructed);
        }
        let mean = total_error / 128.0;
        assert!(mean < 1.0, "mean reconstruction error too high: {mean}");
    }

    /// Sorts `(index, distance)` pairs ascending, deterministically on ties.
    fn ranked(mut pairs: Vec<(usize, f32)>) -> Vec<(usize, f32)> {
        pairs.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap().then(a.0.cmp(&b.0)));
        pairs
    }

    #[test]
    fn adc_estimator_is_far_better_than_chance() {
        // Pure ADC scoring is a *coarse* estimator by design: a 32-dimensional
        // vector is represented by 8 bytes, so distances carry a quantization
        // error comparable to the gaps between near neighbours. This test pins
        // down that the encoder works at all — an off-by-one in the subspace
        // offset or LUT indexing collapses recall to ~0.5% (chance), while the
        // measured value on this data set is several times higher. The strong
        // correctness signal lives in the reranking test below, which is how PQ
        // is used in practice.
        let (count, dim, m, ksub) = (4000usize, 32usize, 8usize, 256usize);
        let data = clustered_data(count, dim, 50, 0.5, 13);
        let pq = ProductQuantizer::train(&data, count, dim, m, ksub, 20, 17, 4096).unwrap();

        let mut codes = vec![0u8; count * m];
        pq.encode_matrix(&data, count, &mut codes).unwrap();

        let query = &data[0..dim];
        let mut lut = vec![0.0f32; m * ksub];
        pq.adc_lut(query, &mut lut).unwrap();
        let approx = ranked(
            (0..count)
                .map(|i| (i, pq.distance_from_lut(&lut, &codes[i * m..(i + 1) * m])))
                .collect(),
        );
        let exact = ranked(
            (0..count)
                .map(|i| {
                    (
                        i,
                        crate::distance::l2_squared(query, &data[i * dim..(i + 1) * dim]),
                    )
                })
                .collect(),
        );

        let exact_top: std::collections::HashSet<usize> =
            exact.iter().take(10).map(|(i, _)| *i).collect();
        let adc_top: std::collections::HashSet<usize> =
            approx.iter().take(10).map(|(i, _)| *i).collect();
        let overlap = exact_top.intersection(&adc_top).count();
        assert!(
            overlap >= 3,
            "ADC-only recall@10 was {overlap}/10 (chance is ~0)"
        );
    }

    #[test]
    fn pq_candidates_plus_exact_reranking_recover_the_true_neighbours() {
        // The production pattern this project uses for IVFPQ: use the compressed
        // codes to cheaply shortlist candidates, then score those candidates
        // with full-precision vectors. Recall must be near-perfect, which is a
        // strong end-to-end check on encoding, LUT construction and code
        // layout together.
        let (count, dim, m, ksub) = (4000usize, 32usize, 8usize, 256usize);
        let data = clustered_data(count, dim, 50, 0.5, 13);
        let pq = ProductQuantizer::train(&data, count, dim, m, ksub, 20, 17, 4096).unwrap();

        let mut codes = vec![0u8; count * m];
        pq.encode_matrix(&data, count, &mut codes).unwrap();

        let query = &data[0..dim];
        let exact = ranked(
            (0..count)
                .map(|i| {
                    (
                        i,
                        crate::distance::l2_squared(query, &data[i * dim..(i + 1) * dim]),
                    )
                })
                .collect(),
        );
        let exact_top: std::collections::HashSet<usize> =
            exact.iter().take(10).map(|(i, _)| *i).collect();

        let mut lut = vec![0.0f32; m * ksub];
        pq.adc_lut(query, &mut lut).unwrap();
        let shortlist = ranked(
            (0..count)
                .map(|i| (i, pq.distance_from_lut(&lut, &codes[i * m..(i + 1) * m])))
                .collect(),
        );
        let shortlist: Vec<usize> = shortlist.iter().take(100).map(|(i, _)| *i).collect();

        let reranked = ranked(
            shortlist
                .iter()
                .map(|&i| {
                    (
                        i,
                        crate::distance::l2_squared(query, &data[i * dim..(i + 1) * dim]),
                    )
                })
                .collect(),
        );
        let reranked_top: std::collections::HashSet<usize> =
            reranked.iter().take(10).map(|(i, _)| *i).collect();
        let overlap = exact_top.intersection(&reranked_top).count();
        assert!(
            overlap >= 9,
            "recall@10 after shortlist of 100 + exact reranking was {overlap}/10"
        );
    }

    #[test]
    fn pq_rejects_non_divisible_dimensions() {
        let data = vec![0.0f32; 100];
        assert!(matches!(
            ProductQuantizer::train(&data, 10, 10, 3, 16, 5, 1, 0),
            Err(Error::InvalidParameter { .. })
        ));
        assert!(matches!(
            ProductQuantizer::train(&data, 10, 10, 5, 0, 5, 1, 0),
            Err(Error::InvalidParameter { .. })
        ));
    }

    #[test]
    fn pq_requires_enough_training_rows() {
        let data = vec![0.0f32; 10 * 8];
        assert!(matches!(
            ProductQuantizer::train(&data, 10, 8, 2, 64, 5, 1, 0),
            Err(Error::InsufficientTrainingData { .. })
        ));
    }

    #[test]
    fn from_parts_checks_codebook_size() {
        assert!(ProductQuantizer::from_parts(8, 2, 4, vec![0.0; 2 * 4 * 4]).is_ok());
        assert!(ProductQuantizer::from_parts(8, 2, 4, vec![0.0; 10]).is_err());
    }

    #[test]
    fn encode_reports_shape_errors() {
        let data = clustered_data(64, 8, 4, 0.05, 21);
        let pq = ProductQuantizer::train(&data, 64, 8, 2, 8, 10, 1, 0).unwrap();
        assert!(matches!(
            pq.encode(&[0.0f32; 7]),
            Err(Error::DimensionMismatch { .. })
        ));
        let mut lut = vec![0.0f32; 2 * 8 - 1];
        assert!(matches!(
            pq.adc_lut(&[0.0f32; 8], &mut lut),
            Err(Error::DimensionMismatch { .. })
        ));
    }
}
