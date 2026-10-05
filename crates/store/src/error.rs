//! Errors produced by the storage layer.

use std::path::PathBuf;

use thiserror::Error;

/// Errors produced by the Lodestar storage layer.
#[derive(Debug, Error)]
pub enum Error {
    /// A filesystem operation failed.
    #[error("i/o error on {path}: {source}")]
    Io {
        /// Path the operation was working on.
        path: PathBuf,
        /// Underlying error.
        #[source]
        source: std::io::Error,
    },

    /// A file did not have the shape its format requires.
    ///
    /// This is the error a fuzzer should never be able to produce by accident
    /// and never able to produce as a panic: corrupt input is data, not a
    /// programming error.
    #[error("corrupt {what}: {detail}")]
    Corrupt {
        /// Which structure failed validation.
        what: String,
        /// What was wrong with it.
        detail: String,
    },

    /// The file was written by a format version this build does not read.
    #[error("unsupported format version {found}, this build reads version {supported}")]
    UnsupportedVersion {
        /// Version found in the file.
        found: u32,
        /// Version this build understands.
        supported: u32,
    },

    /// The manifest was missing a required field or named something unusable.
    #[error("invalid manifest: {0}")]
    InvalidManifest(String),

    /// A named collection does not exist.
    #[error("collection `{0}` does not exist")]
    NoSuchCollection(String),

    /// A collection with that name is already open or already on disk.
    #[error("collection `{0}` already exists")]
    CollectionExists(String),

    /// The caller asked for something the engine refused.
    #[error(transparent)]
    Core(#[from] lodestar_ann_core::Error),
}

/// Convenience alias for results produced by the storage layer.
pub type Result<T> = std::result::Result<T, Error>;

impl Error {
    /// Builds a [`Error::Corrupt`] without spelling out the struct every time.
    pub(crate) fn corrupt(what: impl Into<String>, detail: impl Into<String>) -> Self {
        Self::Corrupt {
            what: what.into(),
            detail: detail.into(),
        }
    }
}
