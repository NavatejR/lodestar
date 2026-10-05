//! 8-bit scalar quantization (SQ8).
//!
//! Each dimension gets its own affine mapping `f32 -> u8`, derived from the
//! minimum and maximum observed during training. That is a 4x memory reduction
//! against raw `f32` storage, at the cost of a bounded per-dimension error of
//! half a quantization step.
//!
//! SQ8 is deliberately *simple*: it is the option you reach for when you want
//! a predictable memory ceiling and a predictable recall cost, without the
//! training complexity of product quantization. Both are benchmarked; see
//! `BENCHMARKS.md`.

use crate::error::{Error, Result};

/// Affine per-dimension quantizer.
#[derive(Debug, Clone, PartialEq)]
pub struct Sq8 {
    dim: usize,
    min: Vec<f32>,
    scale: Vec<f32>,
}

/// Number of distinct codes representable in one byte.
const LEVELS: f32 = 255.0;

impl Sq8 {
    /// Trains a quantizer on a contiguous `count x dim` matrix.
    ///
    /// # Errors
    ///
    /// * [`Error::DimensionMismatch`] if `vectors.len() != count * dim`.
    /// * [`Error::InsufficientTrainingData`] if `count` is zero.
    /// * [`Error::InvalidParameter`] if `dim` is zero.
    pub fn train(vectors: &[f32], count: usize, dim: usize) -> Result<Self> {
        if dim == 0 {
            return Err(Error::InvalidParameter {
                name: "dim",
                reason: "must be greater than zero".to_string(),
            });
        }
        if count == 0 {
            return Err(Error::InsufficientTrainingData { needed: 1, got: 0 });
        }
        if vectors.len() != count * dim {
            return Err(Error::DimensionMismatch {
                expected: count * dim,
                actual: vectors.len(),
            });
        }

        let mut min = vec![f32::INFINITY; dim];
        let mut max = vec![f32::NEG_INFINITY; dim];
        for row in vectors.chunks_exact(dim) {
            for (d, &value) in row.iter().enumerate() {
                if value < min[d] {
                    min[d] = value;
                }
                if value > max[d] {
                    max[d] = value;
                }
            }
        }

        let mut scale = Vec::with_capacity(dim);
        for d in 0..dim {
            let range = max[d] - min[d];
            // A constant (or non-finite) dimension cannot be quantized: pin the
            // scale to 1.0 so that every value encodes to zero and decodes back
            // to `min`. `is_nan` is checked explicitly because a NaN range must
            // be treated as degenerate here.
            if range.is_nan() || range <= 0.0 || !range.is_finite() {
                scale.push(1.0);
            } else {
                scale.push(range / LEVELS);
            }
        }
        Ok(Self { dim, min, scale })
    }

    /// Rebuilds a quantizer from raw parts, as stored in a segment header.
    ///
    /// # Errors
    ///
    /// Returns [`Error::DimensionMismatch`] if `min` and `scale` disagree in
    /// length or are empty, and [`Error::InvalidParameter`] if any value is not
    /// finite or any scale is not strictly positive.
    pub fn from_parts(min: Vec<f32>, scale: Vec<f32>) -> Result<Self> {
        if min.len() != scale.len() || min.is_empty() {
            return Err(Error::DimensionMismatch {
                expected: min.len(),
                actual: scale.len(),
            });
        }
        for (d, (&m, &s)) in min.iter().zip(&scale).enumerate() {
            if !m.is_finite() || !s.is_finite() || s <= 0.0 {
                return Err(Error::InvalidParameter {
                    name: "sq8_params",
                    reason: format!("dimension {d} has non-finite or non-positive parameters"),
                });
            }
        }
        Ok(Self {
            dim: min.len(),
            min,
            scale,
        })
    }

    /// Dimensionality this quantizer expects.
    #[must_use]
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Bytes required to store one encoded vector.
    #[must_use]
    pub fn code_len(&self) -> usize {
        self.dim
    }

    /// Bytes held by the quantization parameters.
    #[must_use]
    pub fn param_bytes(&self) -> usize {
        (self.min.len() + self.scale.len()) * std::mem::size_of::<f32>()
    }

    /// Per-dimension minima.
    #[must_use]
    pub fn min(&self) -> &[f32] {
        &self.min
    }

    /// Per-dimension step sizes.
    #[must_use]
    pub fn scale(&self) -> &[f32] {
        &self.scale
    }

    /// Largest absolute error a single dimension can suffer.
    #[must_use]
    pub fn max_abs_error(&self) -> f32 {
        self.scale.iter().copied().fold(0.0, f32::max) / 2.0
    }

    /// Encodes one vector into `out`.
    ///
    /// # Errors
    ///
    /// * [`Error::DimensionMismatch`] if `vector.len() != self.dim()`.
    /// * [`Error::DimensionMismatch`] if `out.len() < self.code_len()`.
    pub fn encode_into(&self, vector: &[f32], out: &mut [u8]) -> Result<()> {
        if vector.len() != self.dim {
            return Err(Error::DimensionMismatch {
                expected: self.dim,
                actual: vector.len(),
            });
        }
        if out.len() < self.dim {
            return Err(Error::DimensionMismatch {
                expected: self.dim,
                actual: out.len(),
            });
        }
        for d in 0..self.dim {
            let normalized = (vector[d] - self.min[d]) / self.scale[d];
            let clamped = normalized.clamp(0.0, LEVELS);
            // `+ 0.5` performs round-half-up; the clamp above guarantees the
            // result fits in a u8 without wrapping.
            out[d] = (clamped + 0.5) as u8;
        }
        Ok(())
    }

    /// Convenience wrapper allocating a fresh code buffer.
    ///
    /// # Errors
    ///
    /// As [`Sq8::encode_into`].
    pub fn encode(&self, vector: &[f32]) -> Result<Vec<u8>> {
        let mut out = vec![0u8; self.dim];
        self.encode_into(vector, &mut out)?;
        Ok(out)
    }

    /// Decodes one code back into `out` as `f32`.
    ///
    /// # Errors
    ///
    /// * [`Error::DimensionMismatch`] if `code.len() < self.code_len()`.
    /// * [`Error::DimensionMismatch`] if `out.len() < self.dim()`.
    pub fn decode_into(&self, code: &[u8], out: &mut [f32]) -> Result<()> {
        if code.len() < self.dim {
            return Err(Error::DimensionMismatch {
                expected: self.dim,
                actual: code.len(),
            });
        }
        if out.len() < self.dim {
            return Err(Error::DimensionMismatch {
                expected: self.dim,
                actual: out.len(),
            });
        }
        for d in 0..self.dim {
            out[d] = self.min[d] + f32::from(code[d]) * self.scale[d];
        }
        Ok(())
    }

    /// Decodes one code into a freshly allocated vector.
    ///
    /// # Errors
    ///
    /// As [`Sq8::decode_into`].
    pub fn decode(&self, code: &[u8]) -> Result<Vec<f32>> {
        let mut out = vec![0.0f32; self.dim];
        self.decode_into(code, &mut out)?;
        Ok(out)
    }

    /// Encodes a contiguous `count x dim` matrix into `count x code_len` bytes.
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
        if out.len() != count * self.dim {
            return Err(Error::DimensionMismatch {
                expected: count * self.dim,
                actual: out.len(),
            });
        }
        for (row, code) in vectors
            .chunks_exact(self.dim)
            .zip(out.chunks_exact_mut(self.dim))
        {
            self.encode_into(row, code)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Rng;

    fn sample(count: usize, dim: usize, seed: u64) -> Vec<f32> {
        let mut rng = Rng::new(seed);
        (0..count * dim)
            .map(|_| (rng.next_f64() as f32) * 10.0 - 5.0)
            .collect()
    }

    #[test]
    fn round_trip_error_is_bounded_by_half_a_step() {
        let dim = 33;
        let vectors = sample(200, dim, 1);
        let quantizer = Sq8::train(&vectors, 200, dim).unwrap();
        let bound = quantizer.max_abs_error() + 1e-6;

        for row in vectors.chunks_exact(dim) {
            let code = quantizer.encode(row).unwrap();
            let decoded = quantizer.decode(&code).unwrap();
            for (original, restored) in row.iter().zip(&decoded) {
                assert!(
                    (original - restored).abs() <= bound,
                    "error {} exceeds bound {bound}",
                    (original - restored).abs()
                );
            }
        }
    }

    #[test]
    fn memory_footprint_is_four_bytes_to_one() {
        let dim = 128;
        let vectors = sample(64, dim, 2);
        let quantizer = Sq8::train(&vectors, 64, dim).unwrap();
        assert_eq!(quantizer.code_len(), dim);
        // 64 vectors: 32 KiB as f32 versus 8 KiB encoded, plus 1 KiB of params.
        assert_eq!(vectors.len() * 4, 64 * dim * 4);
        assert_eq!(quantizer.code_len() * 64, 64 * dim);
        assert!(quantizer.param_bytes() < 64 * dim);
    }

    #[test]
    fn constant_dimensions_do_not_divide_by_zero() {
        // Every vector is identical: no dimension has any range to encode.
        let dim = 4;
        let vectors = vec![2.5f32; 40];
        let quantizer = Sq8::train(&vectors, 10, dim).unwrap();
        let code = quantizer.encode(&[2.5, 2.5, 2.5, 2.5]).unwrap();
        assert_eq!(code, vec![0, 0, 0, 0]);
        let decoded = quantizer.decode(&code).unwrap();
        for value in decoded {
            assert!((value - 2.5).abs() < 1e-6);
        }
    }

    #[test]
    fn values_outside_the_training_range_are_clamped() {
        let dim = 2;
        let vectors = vec![0.0f32, 0.0, 1.0, 1.0, 0.5, 0.25];
        let quantizer = Sq8::train(&vectors, 3, dim).unwrap();
        let code = quantizer.encode(&[100.0, -100.0]).unwrap();
        assert_eq!(code, vec![255, 0]);
    }

    #[test]
    fn train_rejects_bad_shapes() {
        let vectors = vec![0.0f32; 10];
        assert!(matches!(
            Sq8::train(&vectors, 3, 3),
            Err(Error::DimensionMismatch { .. })
        ));
        assert!(matches!(
            Sq8::train(&vectors, 0, 3),
            Err(Error::InsufficientTrainingData { .. })
        ));
        assert!(matches!(
            Sq8::train(&vectors, 1, 0),
            Err(Error::InvalidParameter { .. })
        ));
    }

    #[test]
    fn encode_decode_report_shape_errors() {
        let vectors = sample(10, 8, 3);
        let quantizer = Sq8::train(&vectors, 10, 8).unwrap();
        assert!(matches!(
            quantizer.encode(&[0.0f32; 7]),
            Err(Error::DimensionMismatch { .. })
        ));
        let mut too_small = [0u8; 4];
        assert!(matches!(
            quantizer.encode_into(&[0.0f32; 8], &mut too_small),
            Err(Error::DimensionMismatch { .. })
        ));
        let mut out = [0.0f32; 3];
        let code = vec![0u8; 8];
        assert!(matches!(
            quantizer.decode_into(&code, &mut out),
            Err(Error::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn from_parts_validates_parameters() {
        assert!(Sq8::from_parts(vec![0.0, 1.0], vec![1.0, 1.0]).is_ok());
        assert!(Sq8::from_parts(vec![0.0], vec![1.0, 1.0]).is_err());
        assert!(Sq8::from_parts(vec![0.0], vec![0.0]).is_err());
        assert!(Sq8::from_parts(vec![f32::NAN], vec![1.0]).is_err());
        assert!(Sq8::from_parts(vec![], vec![]).is_err());
    }

    #[test]
    fn matrix_encoding_matches_row_encoding() {
        let dim = 16;
        let count = 25;
        let vectors = sample(count, dim, 4);
        let quantizer = Sq8::train(&vectors, count, dim).unwrap();
        let mut matrix = vec![0u8; count * dim];
        quantizer
            .encode_matrix(&vectors, count, &mut matrix)
            .unwrap();
        for (i, row) in vectors.chunks_exact(dim).enumerate() {
            let single = quantizer.encode(row).unwrap();
            assert_eq!(&matrix[i * dim..(i + 1) * dim], &single[..]);
        }
    }

    #[test]
    fn quantized_distances_approximate_exact_distances() {
        // The practical property that matters: SQ8 must preserve the ranking of
        // a query against its true nearest neighbours on well-spread data.
        let dim = 64;
        let data = sample(500, dim, 5);
        let quantizer = Sq8::train(&data, 500, dim).unwrap();
        let query = &data[0..dim];

        let mut exact: Vec<(usize, f32)> = (0..500)
            .map(|i| {
                (
                    i,
                    crate::distance::l2_squared(query, &data[i * dim..(i + 1) * dim]),
                )
            })
            .collect();
        exact.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        let q_code = quantizer.encode(query).unwrap();
        let q_decoded = quantizer.decode(&q_code).unwrap();
        let mut approx: Vec<(usize, f32)> = (0..500)
            .map(|i| {
                let code = quantizer.encode(&data[i * dim..(i + 1) * dim]).unwrap();
                let restored = quantizer.decode(&code).unwrap();
                (i, crate::distance::l2_squared(&q_decoded, &restored))
            })
            .collect();
        approx.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        let exact_top: std::collections::HashSet<usize> =
            exact.iter().take(10).map(|(i, _)| *i).collect();
        let approx_top: std::collections::HashSet<usize> =
            approx.iter().take(10).map(|(i, _)| *i).collect();
        let overlap = exact_top.intersection(&approx_top).count();
        assert!(overlap >= 8, "SQ8 recall@10 was {overlap}/10");
    }
}
