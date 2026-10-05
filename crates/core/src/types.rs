//! Value types shared by every Lodestar crate.

use serde::{Deserialize, Serialize};

use crate::error::{Error, Result};

/// Similarity metric used for ranking.
///
/// Distances are always expressed as values where **smaller is closer**, which
/// keeps the graph traversal code metric-agnostic:
///
/// | metric | stored representation | distance |
/// |---|---|---|
/// | [`Metric::L2`] | raw vector | squared euclidean |
/// | [`Metric::Cosine`] | L2-normalised vector | `1 - dot(a, b)` |
/// | [`Metric::InnerProduct`] | raw vector | `-dot(a, b)` |
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Metric {
    /// Squared euclidean distance.
    L2,
    /// Cosine distance (`1 - cosine similarity`).
    Cosine,
    /// Negative dot product, so that ranking is still ascending.
    #[serde(rename = "inner_product", alias = "ip")]
    InnerProduct,
}

impl Metric {
    /// Canonical lower-case name, used in manifests, the HTTP API and the CLI.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::L2 => "l2",
            Self::Cosine => "cosine",
            Self::InnerProduct => "inner_product",
        }
    }

    /// Parses a metric name. Accepts `ip` as an alias for `inner_product`.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidParameter`] if the name is not recognised.
    pub fn parse(name: &str) -> Result<Self> {
        match name.trim().to_ascii_lowercase().as_str() {
            "l2" | "euclidean" => Ok(Self::L2),
            "cosine" | "cos" => Ok(Self::Cosine),
            "inner_product" | "ip" | "dot" => Ok(Self::InnerProduct),
            other => Err(Error::InvalidParameter {
                name: "metric",
                reason: format!("unknown metric `{other}`, expected l2 | cosine | inner_product"),
            }),
        }
    }

    /// Whether vectors must be L2-normalised when they enter the index.
    #[must_use]
    pub const fn normalizes_on_insert(self) -> bool {
        matches!(self, Self::Cosine)
    }

    /// Applies the metric's canonical pre-processing to a vector in place.
    ///
    /// For [`Metric::Cosine`] this L2-normalises the vector so that the inner
    /// product equals the cosine similarity. A zero vector is left untouched
    /// rather than turned into `NaN`.
    pub fn prepare(self, vector: &mut [f32]) {
        if self.normalizes_on_insert() {
            l2_normalize(vector);
        }
    }

    /// Distance between two *prepared* vectors.
    ///
    /// Both operands must have already been through [`Metric::prepare`].
    #[must_use]
    pub fn distance(self, a: &[f32], b: &[f32]) -> f32 {
        match self {
            Self::L2 => crate::distance::l2_squared(a, b),
            Self::Cosine => 1.0 - crate::distance::inner_product(a, b),
            Self::InnerProduct => -crate::distance::inner_product(a, b),
        }
    }
}

impl std::fmt::Display for Metric {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// L2-normalises a vector in place. Zero vectors are left unchanged.
///
/// The exact zero check is deliberate: normalising an all-zero vector would
/// produce `NaN`, which would then poison every comparison in the graph search.
pub fn l2_normalize(vector: &mut [f32]) {
    let mut sum = 0.0f64;
    for &x in vector.iter() {
        sum += f64::from(x) * f64::from(x);
    }
    if sum <= f64::MIN_POSITIVE {
        return;
    }
    let inv = (1.0 / sum.sqrt()) as f32;
    for x in vector.iter_mut() {
        *x *= inv;
    }
}

/// A ranked search hit.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Hit {
    /// External identifier supplied by the caller.
    pub id: u64,
    /// Distance under the index metric; smaller means closer.
    pub distance: f32,
}

/// Identifies a vector by external id and keeps a stable ordering.
///
/// Ordering is total: distances are compared first, then ids. Total ordering is
/// what allows the search heaps to be plain [`std::collections::BinaryHeap`]s
/// without wrapping distances in a fallible comparison, and it also makes
/// results deterministic when two vectors are equidistant.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Candidate {
    /// Distance from the query under the index metric.
    pub distance: f32,
    /// External id of the vector.
    pub id: u64,
}

impl Eq for Candidate {}

impl Ord for Candidate {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.distance
            .partial_cmp(&other.distance)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| self.id.cmp(&other.id))
    }
}

impl PartialOrd for Candidate {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn metric_names_round_trip() {
        for metric in [Metric::L2, Metric::Cosine, Metric::InnerProduct] {
            assert_eq!(Metric::parse(metric.as_str()), Ok(metric));
            assert_eq!(metric.to_string(), metric.as_str());
        }
    }

    #[test]
    fn metric_aliases_parse() {
        assert_eq!(Metric::parse("IP"), Ok(Metric::InnerProduct));
        assert_eq!(Metric::parse(" Euclidean "), Ok(Metric::L2));
        assert_eq!(Metric::parse("cos"), Ok(Metric::Cosine));
    }

    #[test]
    fn unknown_metric_is_an_error() {
        let err = Metric::parse("manhattan").unwrap_err();
        assert!(err.to_string().contains("manhattan"));
    }

    #[test]
    fn cosine_preparation_normalises() {
        let mut v = vec![3.0f32, 4.0];
        Metric::Cosine.prepare(&mut v);
        assert!((v[0] - 0.6).abs() < 1e-6);
        assert!((v[1] - 0.8).abs() < 1e-6);
        // Unit length: self-distance is zero.
        assert!(Metric::Cosine.distance(&v, &v).abs() < 1e-6);
    }

    #[test]
    fn zero_vector_does_not_become_nan() {
        let mut v = vec![0.0f32, 0.0];
        Metric::Cosine.prepare(&mut v);
        assert!(v.iter().all(|x| x.is_finite()));
        assert_eq!(v, vec![0.0, 0.0]);
    }

    #[test]
    fn inner_product_ranks_similar_vectors_first() {
        let a = vec![1.0f32, 0.0];
        let close = vec![0.9f32, 0.1];
        let far = vec![-1.0f32, 0.0];
        assert!(
            Metric::InnerProduct.distance(&a, &close) < Metric::InnerProduct.distance(&a, &far)
        );
    }

    #[test]
    fn candidate_ordering_is_total_and_deterministic() {
        let mut v = [
            Candidate {
                distance: 1.0,
                id: 9,
            },
            Candidate {
                distance: 0.5,
                id: 3,
            },
            Candidate {
                distance: 0.5,
                id: 1,
            },
        ];
        v.sort();
        assert_eq!(v[0].id, 1);
        assert_eq!(v[1].id, 3);
        assert_eq!(v[2].id, 9);
    }

    #[test]
    fn nan_distances_do_not_break_ordering() {
        // A NaN can only appear if a caller bypassed Metric::prepare; ordering
        // still has to be total and must not panic.
        let a = Candidate {
            distance: f32::NAN,
            id: 2,
        };
        let b = Candidate {
            distance: 1.0,
            id: 1,
        };
        assert_eq!(a.cmp(&b), std::cmp::Ordering::Greater);
        assert_eq!(b.cmp(&a), std::cmp::Ordering::Less);
    }
}
