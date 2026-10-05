//! Error types shared across Lodestar crates.

use thiserror::Error;

/// Errors produced by the Lodestar core primitives.
#[derive(Debug, Clone, PartialEq, Error)]
pub enum Error {
    /// A vector or matrix had a different dimensionality than the index.
    #[error("dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch {
        /// Dimensionality the caller was expected to provide.
        expected: usize,
        /// Dimensionality the caller actually provided.
        actual: usize,
    },

    /// A configuration value was out of range or internally inconsistent.
    #[error("invalid parameter `{name}`: {reason}")]
    InvalidParameter {
        /// Name of the offending parameter.
        name: &'static str,
        /// Why the value was rejected.
        reason: String,
    },

    /// There was not enough data to train a quantizer.
    #[error("insufficient training data: need at least {needed}, got {got}")]
    InsufficientTrainingData {
        /// Minimum number of vectors required.
        needed: usize,
        /// Number of vectors supplied.
        got: usize,
    },

    /// The index does not contain any live vectors.
    #[error("index is empty")]
    EmptyIndex,

    /// An id was inserted twice with conflicting data and duplicates are not allowed.
    #[error("duplicate id: {0}")]
    DuplicateId(u64),

    /// A requested id is not present in the index.
    #[error("id not found: {0}")]
    NotFound(u64),

    /// Training or encoding produced a numerically degenerate result.
    #[error("numeric failure: {0}")]
    Numeric(String),
}

/// Convenience alias for results produced by Lodestar.
pub type Result<T> = std::result::Result<T, Error>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn errors_render_readable_messages() {
        let err = Error::DimensionMismatch {
            expected: 128,
            actual: 3,
        };
        assert_eq!(
            err.to_string(),
            "dimension mismatch: expected 128, got 3".to_string()
        );

        let err = Error::InvalidParameter {
            name: "m",
            reason: "must be >= 2".to_string(),
        };
        assert!(err.to_string().contains("`m`"));
    }
}
