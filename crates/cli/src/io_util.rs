//! Shared parsing for the CLI: metric names, vector literals and JSONL
//! records. Kept in one place so the accepted spellings stay consistent
//! between `create`, `search` and `sample`.

use anyhow::{Result, bail};
use lodestar_ann_core::Metric;
use serde::Deserialize;

/// One record of an insert or query JSONL file.
///
/// Unknown fields are ignored, so a file written for the HTTP API — which
/// carries metadata the CLI does not need — still inserts cleanly.
#[derive(Deserialize)]
pub struct InsertRecord {
    /// External id of the vector.
    pub id: u64,
    /// The vector itself.
    pub vector: Vec<f32>,
}

/// Parses a metric name: `l2`, `cosine`, `inner_product`, `inner-product` or
/// `ip`.
///
/// # Errors
///
/// [`anyhow::Error`] for anything else, listing the accepted spellings.
pub fn parse_metric(spec: &str) -> Result<Metric> {
    match spec.to_ascii_lowercase().as_str() {
        "l2" | "euclidean" => Ok(Metric::L2),
        "cosine" => Ok(Metric::Cosine),
        "inner_product" | "inner-product" | "ip" | "dot" => Ok(Metric::InnerProduct),
        other => bail!("unknown metric `{other}`; expected one of l2, cosine, inner_product, ip"),
    }
}

/// Parses a comma-separated vector literal such as `0.1, 0.2, 0.3`.
///
/// # Errors
///
/// [`anyhow::Error`] if any component is not an `f32`.
pub fn parse_vector(spec: &str) -> Result<Vec<f32>> {
    let mut vector = Vec::new();
    for part in spec.split(',') {
        let part = part.trim();
        if part.is_empty() {
            continue;
        }
        let value: f32 = part
            .parse()
            .map_err(|_| anyhow::anyhow!("`{part}` is not a number in the vector literal"))?;
        vector.push(value);
    }
    if vector.is_empty() {
        bail!("the vector literal is empty");
    }
    Ok(vector)
}
