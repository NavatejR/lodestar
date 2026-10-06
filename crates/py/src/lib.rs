//! Python bindings for Lodestar.
//!
//! The surface is deliberately a thin mirror of the Rust one: an
//! [`Index`](PyIndex) holds vectors in memory for fast experimentation, a
//! [`Collection`](PyCollection) persists to disk through the store layer, and
//! both accept and return NumPy arrays without copying where possible.
//!
//! Build with `maturin develop` (see the Makefile's `test-py` target); the
//! typed stub in `python/lodestar/__init__.pyi` keeps IDEs and type checkers
//! honest without importing the compiled module.

use numpy::prelude::*;
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyType;

use lodestar_ann_core::{Metric as RustMetric, l2_normalize};
use lodestar_ann_index::hnsw::HnswConfig;
use lodestar_ann_store::Collection as RustCollection;

/// Maps a Python string to the metric enum.
fn parse_metric(spec: &str) -> PyResult<RustMetric> {
    match spec.to_ascii_lowercase().as_str() {
        "l2" | "euclidean" => Ok(RustMetric::L2),
        "cosine" => Ok(RustMetric::Cosine),
        "inner_product" | "inner-product" | "ip" | "dot" => Ok(RustMetric::InnerProduct),
        other => Err(PyValueError::new_err(format!(
            "unknown metric `{other}`; expected one of l2, cosine, inner_product"
        ))),
    }
}

/// Maps a Rust error to a Python exception with its message.
fn to_py(error: impl std::fmt::Display) -> PyErr {
    PyValueError::new_err(error.to_string())
}

/// An in-memory HNSW index.
///
/// Use this for experiments and benchmarks; use [`Collection`](PyCollection)
/// when the vectors have to survive the process.
///
/// >>> index = lodestar.Index(dim=3, metric="cosine")
/// >>> index.add(np.array([1, 2, 3], dtype=np.float32), id=1)
/// >>> index.search(np.array([1, 2, 3], dtype=np.float32), k=1)
/// [(1, 0.0)]
#[pyclass(name = "Index")]
struct PyIndex {
    inner: lodestar_ann_index::hnsw::Hnsw,
    dim: usize,
}

#[pymethods]
impl PyIndex {
    /// Creates an empty index.
    #[new]
    #[pyo3(signature = (dim, metric = "l2", m = 16, m0 = 32, ef_construction = 200, seed = None))]
    fn new(
        dim: usize,
        metric: &str,
        m: usize,
        m0: usize,
        ef_construction: usize,
        seed: Option<u64>,
    ) -> PyResult<Self> {
        if dim == 0 {
            return Err(PyValueError::new_err("dim must be at least 1"));
        }
        let mut config = HnswConfig::default();
        config.m = m;
        config.m0 = m0;
        config.ef_construction = ef_construction;
        if let Some(value) = seed {
            config.seed = value;
        }
        let inner = lodestar_ann_index::hnsw::Hnsw::new(dim, parse_metric(metric)?, config)
            .map_err(to_py)?;
        Ok(Self { inner, dim })
    }

    /// Vector dimensionality.
    #[getter]
    fn dim(&self) -> usize {
        self.dim
    }

    /// Ranking metric name.
    #[getter]
    fn metric(&self) -> &'static str {
        self.inner.metric().as_str()
    }

    /// Number of live vectors.
    #[getter]
    fn len(&self) -> usize {
        self.inner.len()
    }

    /// Number of nodes held, tombstones included.
    #[getter]
    fn node_count(&self) -> usize {
        self.inner.node_count()
    }

    /// Approximate heap memory used, in bytes.
    #[getter]
    fn memory_bytes(&self) -> usize {
        self.inner.memory_bytes()
    }

    /// Inserts one vector. Re-inserting an id replaces its vector.
    fn add(&mut self, py: Python<'_>, vector: PyReadonlyArray1<f32>, id: u64) -> PyResult<()> {
        let vector = vector.as_slice()?;
        if vector.len() != self.dim {
            return Err(PyValueError::new_err(format!(
                "vector has {} dimensions, index has {}",
                vector.len(),
                self.dim
            )));
        }
        py.detach(|| self.inner.insert(id, vector)).map_err(to_py)
    }

    /// Inserts a batch: an `(n, dim)` array and an `(n,)` id array.
    ///
    /// Returns the number of rows inserted.
    fn add_batch(
        &mut self,
        py: Python<'_>,
        vectors: PyReadonlyArray2<f32>,
        ids: PyReadonlyArray1<u64>,
    ) -> PyResult<usize> {
        let shape = vectors.as_array().shape().to_vec();
        let (rows, columns) = (shape[0], shape[1]);
        if columns != self.dim {
            return Err(PyValueError::new_err(format!(
                "vectors have {columns} dimensions, index has {}",
                self.dim
            )));
        }
        if ids.len() != rows {
            return Err(PyValueError::new_err(format!(
                "{} ids for {rows} rows",
                ids.len()
            )));
        }
        let vectors = vectors.as_slice()?;
        let ids = ids.as_slice()?;
        py.detach(|| self.inner.insert_batch(ids, vectors))
            .map_err(to_py)
    }

    /// Tombstones an id. Returns whether a live vector was removed.
    fn remove(&mut self, id: u64) -> bool {
        self.inner.delete(id)
    }

    /// Searches for the `k` nearest neighbours.
    ///
    /// Returns a list of `(id, distance)` pairs sorted by ascending distance.
    #[pyo3(signature = (vector, k, ef = None))]
    fn search(
        &self,
        py: Python<'_>,
        vector: PyReadonlyArray1<f32>,
        k: usize,
        ef: Option<usize>,
    ) -> PyResult<Vec<(u64, f32)>> {
        let vector = vector.as_slice()?;
        if vector.len() != self.dim {
            return Err(PyValueError::new_err(format!(
                "query has {} dimensions, index has {}",
                vector.len(),
                self.dim
            )));
        }
        py.detach(|| self.inner.search_with_ef(vector, k, ef.unwrap_or(64)))
            .map_err(to_py)
            .map(|hits| hits.into_iter().map(|hit| (hit.id, hit.distance)).collect())
    }

    /// Returns a copy with tombstoned nodes physically removed.
    fn compact(&self, py: Python<'_>) -> PyResult<Self> {
        py.detach(|| self.inner.compact())
            .map(|inner| Self {
                inner,
                dim: self.dim,
            })
            .map_err(to_py)
    }
}

/// A durable, searchable collection of vectors on disk.
///
/// Writes go to a write-ahead log and are synced before the call returns, so
/// an acknowledged write survives a crash; a flush seals the log into an
/// immutable, memory-mapped segment.
///
/// >>> collection = lodestar.Collection.open("./data", "docs", dim=384, metric="cosine")
/// >>> collection.add(np.array([0.1, 0.2], dtype=np.float32), id=1)
/// >>> collection.search(np.array([0.1, 0.2], dtype=np.float32), k=3)
/// [(1, 0.0)]
#[pyclass(name = "Collection")]
struct PyCollection {
    inner: RustCollection,
    dim: usize,
}

#[pymethods]
impl PyCollection {
    /// Creates a collection, failing if one already exists under that name.
    #[classmethod]
    #[pyo3(signature = (root, name, dim, metric = "l2"))]
    fn create(
        _cls: &Bound<'_, PyType>,
        root: &str,
        name: &str,
        dim: usize,
        metric: &str,
    ) -> PyResult<Self> {
        let metric = parse_metric(metric)?;
        let inner = RustCollection::create(root, name, dim, metric, HnswConfig::default())
            .map_err(to_py)?;
        Ok(Self { inner, dim })
    }

    /// Opens a collection, replaying the write-ahead log.
    #[classmethod]
    fn open(_cls: &Bound<'_, PyType>, root: &str, name: &str) -> PyResult<Self> {
        let inner = RustCollection::open(root, name).map_err(to_py)?;
        let dim = inner.dim();
        Ok(Self { inner, dim })
    }

    /// Opens a collection or creates it if it does not exist.
    #[classmethod]
    #[pyo3(signature = (root, name, dim, metric = "l2"))]
    fn open_or_create(
        _cls: &Bound<'_, PyType>,
        root: &str,
        name: &str,
        dim: usize,
        metric: &str,
    ) -> PyResult<Self> {
        let metric = parse_metric(metric)?;
        let inner = RustCollection::open_or_create(root, name, dim, metric, HnswConfig::default())
            .map_err(to_py)?;
        let dim = inner.dim();
        Ok(Self { inner, dim })
    }

    /// Collection name.
    #[getter]
    fn name(&self) -> &str {
        self.inner.name()
    }

    /// Vector dimensionality.
    #[getter]
    fn dim(&self) -> usize {
        self.dim
    }

    /// Ranking metric name.
    #[getter]
    fn metric(&self) -> &'static str {
        self.inner.metric().as_str()
    }

    /// Number of live vectors.
    #[getter]
    fn len(&self) -> usize {
        self.inner.live_len()
    }

    /// Inserts one vector. Re-inserting an id replaces its vector, and the
    /// write is durable when this returns.
    fn add(&mut self, py: Python<'_>, vector: PyReadonlyArray1<f32>, id: u64) -> PyResult<()> {
        let vector = vector.as_slice()?;
        if vector.len() != self.dim {
            return Err(PyValueError::new_err(format!(
                "vector has {} dimensions, collection has {}",
                vector.len(),
                self.dim
            )));
        }
        py.detach(|| self.inner.upsert(id, vector)).map_err(to_py)
    }

    /// Inserts a batch: an `(n, dim)` array and an `(n,)` id array, synced once.
    fn add_batch(
        &mut self,
        py: Python<'_>,
        vectors: PyReadonlyArray2<f32>,
        ids: PyReadonlyArray1<u64>,
    ) -> PyResult<usize> {
        let shape = vectors.as_array().shape().to_vec();
        let (rows, columns) = (shape[0], shape[1]);
        if columns != self.dim {
            return Err(PyValueError::new_err(format!(
                "vectors have {columns} dimensions, collection has {}",
                self.dim
            )));
        }
        if ids.len() != rows {
            return Err(PyValueError::new_err(format!(
                "{} ids for {rows} rows",
                ids.len()
            )));
        }
        let vectors = vectors.as_slice()?;
        let ids = ids.as_slice()?;
        py.detach(|| self.inner.upsert_batch(ids, vectors))
            .map_err(to_py)
    }

    /// Tombstones an id everywhere. The delete is durable when this returns.
    fn remove(&mut self, id: u64) -> PyResult<()> {
        self.inner.delete(id).map_err(to_py)
    }

    /// Searches for the `k` nearest neighbours as `(id, distance)` pairs.
    #[pyo3(signature = (vector, k, ef = None))]
    fn search(
        &self,
        py: Python<'_>,
        vector: PyReadonlyArray1<f32>,
        k: usize,
        ef: Option<usize>,
    ) -> PyResult<Vec<(u64, f32)>> {
        let vector = vector.as_slice()?;
        if vector.len() != self.dim {
            return Err(PyValueError::new_err(format!(
                "query has {} dimensions, collection has {}",
                vector.len(),
                self.dim
            )));
        }
        py.detach(|| self.inner.search(vector, k, ef.unwrap_or(64)))
            .map_err(to_py)
            .map(|hits| hits.into_iter().map(|hit| (hit.id, hit.distance)).collect())
    }

    /// Seals the in-memory tail into an immutable segment.
    fn flush(&mut self, py: Python<'_>) -> PyResult<Option<String>> {
        py.detach(|| self.inner.flush())
            .map_err(to_py)
            .map(|info| info.map(|segment| segment.path.display().to_string()))
    }

    /// Rewrites every live vector into one segment, dropping tombstones.
    fn compact(&mut self, py: Python<'_>) -> PyResult<Option<String>> {
        py.detach(|| self.inner.compact())
            .map_err(to_py)
            .map(|info| info.map(|segment| segment.path.display().to_string()))
    }

    /// Runs the full checksum pass over every sealed segment.
    fn verify(&self, py: Python<'_>) -> PyResult<()> {
        py.detach(|| self.inner.verify()).map_err(to_py)
    }
}

/// Generates a synthetic clustered corpus for demos and tests.
///
/// Returns `(ids, vectors)` where `vectors` is `(count, dim)` and every point
/// sits near one of `clusters` random centres. Cosine corpora are normalised.
#[pyfunction]
#[pyo3(signature = (count, dim, metric = "l2", clusters = 16, seed = 7))]
fn sample<'py>(
    py: Python<'py>,
    count: usize,
    dim: usize,
    metric: &str,
    clusters: usize,
    seed: u64,
) -> PyResult<(Vec<u64>, Bound<'py, PyArray2<f32>>)> {
    if count == 0 || dim == 0 {
        return Err(PyValueError::new_err("count and dim must be at least 1"));
    }
    let metric = parse_metric(metric)?;
    let vectors: Vec<f32> = py.detach(move || -> Result<Vec<f32>, PyErr> {
        use lodestar_ann_core::Rng;
        let mut rng = Rng::new(seed);
        let spread = 0.35f32;
        let centers: Vec<Vec<f32>> = (0..clusters)
            .map(|_| {
                (0..dim)
                    .map(|_| (rng.next_f64() * 2.0 - 1.0) as f32)
                    .collect()
            })
            .collect();
        let mut vectors = vec![0.0f32; count * dim];
        for row in 0..count {
            let center = &centers[rng.below(clusters)];
            let mut vector: Vec<f32> = center
                .iter()
                .map(|x| x + (rng.next_f64() * 2.0 - 1.0) as f32 * spread)
                .collect();
            if metric == RustMetric::Cosine {
                l2_normalize(&mut vector);
            }
            vectors[row * dim..(row + 1) * dim].copy_from_slice(&vector);
        }
        Ok(vectors)
    })?;
    let ids: Vec<u64> = (0..count as u64).collect();
    let flat = vectors.into_pyarray(py);
    Ok((ids, flat.reshape([count, dim])?))
}

/// The Lodestar extension module.
#[pymodule]
fn lodestar_native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add_class::<PyIndex>()?;
    m.add_class::<PyCollection>()?;
    m.add_function(wrap_pyfunction!(sample, m)?)?;
    Ok(())
}
