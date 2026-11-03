use crate::core::{self, ChunkError, Config, Strategy};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// A chunk of text. `start`/`end` are character offsets into the source string,
/// so `source[chunk.start:chunk.end] == chunk.text`.
#[pyclass(frozen, get_all, module = "chunkwise", skip_from_py_object)]
#[derive(Clone, Debug)]
pub struct Chunk {
    text: String,
    start: usize,
    end: usize,
    index: usize,
}

#[pymethods]
impl Chunk {
    fn __repr__(&self) -> String {
        let preview: String = self.text.chars().take(40).collect();
        let dots = if self.text.chars().count() > 40 {
            "..."
        } else {
            ""
        };
        format!(
            "Chunk(index={}, start={}, end={}, text={:?}{})",
            self.index, self.start, self.end, preview, dots
        )
    }
    fn __len__(&self) -> usize {
        self.text.chars().count()
    }
}

impl From<core::Chunk> for Chunk {
    fn from(c: core::Chunk) -> Self {
        Chunk {
            text: c.text,
            start: c.start,
            end: c.end,
            index: c.index,
        }
    }
}

fn to_py_err(e: ChunkError) -> PyErr {
    PyValueError::new_err(e.to_string())
}

fn make_config(chunk_size: usize, chunk_overlap: usize, strategy: &str) -> PyResult<Config> {
    let strategy: Strategy = strategy.parse().map_err(to_py_err)?;
    let cfg = Config {
        chunk_size,
        chunk_overlap,
        strategy,
    };
    cfg.validate().map_err(to_py_err)?;
    Ok(cfg)
}

/// Split one string into chunks.
#[pyfunction]
#[pyo3(signature = (text, chunk_size=1000, chunk_overlap=200, strategy="recursive"))]
fn chunk_text(
    py: Python<'_>,
    text: &str,
    chunk_size: usize,
    chunk_overlap: usize,
    strategy: &str,
) -> PyResult<Vec<Chunk>> {
    let cfg = make_config(chunk_size, chunk_overlap, strategy)?;
    let out = py.detach(|| core::chunk(text, &cfg)).map_err(to_py_err)?;
    Ok(out.into_iter().map(Chunk::from).collect())
}

/// Split many strings in parallel. The GIL is released while chunking.
#[pyfunction]
#[pyo3(signature = (texts, chunk_size=1000, chunk_overlap=200, strategy="recursive"))]
fn chunk_texts(
    py: Python<'_>,
    texts: Vec<String>,
    chunk_size: usize,
    chunk_overlap: usize,
    strategy: &str,
) -> PyResult<Vec<Vec<Chunk>>> {
    let cfg = make_config(chunk_size, chunk_overlap, strategy)?;
    let out = py
        .detach(|| core::chunk_many(&texts, &cfg))
        .map_err(to_py_err)?;
    Ok(out
        .into_iter()
        .map(|v| v.into_iter().map(Chunk::from).collect())
        .collect())
}

#[pymodule]
#[pyo3(name = "_core")]
fn init(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Chunk>()?;
    m.add_function(wrap_pyfunction!(chunk_text, m)?)?;
    m.add_function(wrap_pyfunction!(chunk_texts, m)?)?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
