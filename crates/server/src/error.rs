//! Turning engine failures into HTTP responses.
//!
//! The engine deliberately has no idea that HTTP exists, so it returns rich
//! typed errors. This module owns the one place where those become status
//! codes, and it is the only place: a handler that invents its own status code
//! is a bug waiting to happen.
//!
//! Every error body has the same shape, so a client can branch on
//! `error.code` instead of parsing prose:
//!
//! ```json
//! {"error":{"code":"not_found","message":"collection `docs` does not exist"}}
//! ```

use axum::Json;
use axum::extract::rejection::JsonRejection;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use lodestar_ann_core::Error as CoreError;
use lodestar_ann_store::Error as StoreError;
use serde::Serialize;

/// A failed request together with the status it maps to.
#[derive(Debug)]
pub enum ApiError {
    /// The request was malformed: bad JSON, a wrong-sized vector, a k of zero.
    BadRequest(String),
    /// The server is running read-only.
    Forbidden(String),
    /// The request named something that does not exist.
    NotFound(String),
    /// The request contradicts existing state, such as creating a collection
    /// that already exists.
    Conflict(String),
    /// The request was well formed but larger than the service accepts.
    TooLarge(String),
    /// The engine failed. Always logged at `error` level with the detail.
    Internal(String),
}

impl ApiError {
    /// The HTTP status this error reports.
    #[must_use]
    pub fn status(&self) -> StatusCode {
        match self {
            Self::BadRequest(_) => StatusCode::BAD_REQUEST,
            Self::Forbidden(_) => StatusCode::FORBIDDEN,
            Self::NotFound(_) => StatusCode::NOT_FOUND,
            Self::Conflict(_) => StatusCode::CONFLICT,
            Self::TooLarge(_) => StatusCode::PAYLOAD_TOO_LARGE,
            Self::Internal(_) => StatusCode::INTERNAL_SERVER_ERROR,
        }
    }

    /// Stable machine-readable code, matching the status.
    #[must_use]
    pub const fn code(&self) -> &'static str {
        match self {
            Self::BadRequest(_) => "bad_request",
            Self::Forbidden(_) => "forbidden",
            Self::NotFound(_) => "not_found",
            Self::Conflict(_) => "conflict",
            Self::TooLarge(_) => "payload_too_large",
            Self::Internal(_) => "internal",
        }
    }

    /// Human-readable explanation.
    #[must_use]
    pub fn message(&self) -> &str {
        match self {
            Self::BadRequest(message)
            | Self::Forbidden(message)
            | Self::NotFound(message)
            | Self::Conflict(message)
            | Self::TooLarge(message)
            | Self::Internal(message) => message,
        }
    }

    /// Wraps any `Display` error as a 500.
    pub fn internal(context: &str, error: impl std::fmt::Display) -> Self {
        Self::Internal(format!("{context}: {error}"))
    }
}

impl std::fmt::Display for ApiError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}: {}", self.code(), self.message())
    }
}

impl std::error::Error for ApiError {}

/// The body sent with every failed request.
#[derive(Serialize)]
struct ErrorBody<'a> {
    error: ErrorDetail<'a>,
}

/// The `error` object inside an error body.
#[derive(Serialize)]
struct ErrorDetail<'a> {
    code: &'a str,
    message: &'a str,
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let status = self.status();
        if status.is_server_error() {
            // Client mistakes are not the operator's problem; server mistakes
            // must be visible in the log, not just in the response body.
            tracing::error!(
                status = status.as_u16(),
                code = self.code(),
                "{}",
                self.message()
            );
        }
        let body = Json(ErrorBody {
            error: ErrorDetail {
                code: self.code(),
                message: self.message(),
            },
        });
        (status, body).into_response()
    }
}

impl From<StoreError> for ApiError {
    fn from(error: StoreError) -> Self {
        match &error {
            StoreError::NoSuchCollection(name) => {
                Self::NotFound(format!("collection `{name}` does not exist"))
            }
            StoreError::CollectionExists(name) => {
                Self::Conflict(format!("collection `{name}` already exists"))
            }
            StoreError::Core(core) => Self::from(core.clone()),
            // A corrupt segment, an unreadable manifest and an I/O failure are
            // all "the operator must look at this", not "the client did
            // something wrong". The store's messages name the file and offset.
            StoreError::Corrupt { .. }
            | StoreError::UnsupportedVersion { .. }
            | StoreError::InvalidManifest(_)
            | StoreError::Io { .. } => Self::Internal(error.to_string()),
        }
    }
}

impl From<CoreError> for ApiError {
    fn from(error: CoreError) -> Self {
        match error {
            CoreError::DimensionMismatch { expected, actual } => Self::BadRequest(format!(
                "dimension mismatch: expected {expected}, got {actual}"
            )),
            CoreError::InvalidParameter { name, reason } => {
                Self::BadRequest(format!("invalid parameter `{name}`: {reason}"))
            }
            CoreError::InsufficientTrainingData { needed, got } => Self::BadRequest(format!(
                "insufficient training data: need at least {needed}, got {got}"
            )),
            CoreError::EmptyIndex => Self::Conflict("the index holds no vectors".to_string()),
            CoreError::DuplicateId(id) => Self::Conflict(format!("duplicate id {id}")),
            CoreError::NotFound(id) => Self::NotFound(format!("id {id} does not exist")),
            CoreError::Numeric(reason) => Self::Internal(format!("numeric failure: {reason}")),
        }
    }
}

impl From<JsonRejection> for ApiError {
    fn from(rejection: JsonRejection) -> Self {
        Self::BadRequest(format!("invalid JSON body: {}", rejection.body_text()))
    }
}

impl From<std::io::Error> for ApiError {
    fn from(error: std::io::Error) -> Self {
        Self::Internal(format!("i/o error: {error}"))
    }
}
