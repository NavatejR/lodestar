//! Python bindings for Lodestar.

use pyo3::prelude::*;

/// Lodestar Python extension module.
#[pymodule]
fn lodestar_native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
