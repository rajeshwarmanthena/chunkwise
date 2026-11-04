//! Python bindings for chunkwise via PyO3.
//!
//! All Rust computation happens in `core` and `token_core`; this module only
//! translates between Python and Rust types.
//!
//! ## Memory and GIL notes
//!
//! `chunk_texts` accepts `Vec<String>`, which clones all Python strings into Rust-
//! owned heap allocations before releasing the GIL.  This is necessary because
//! PyO3's borrowed Python string references cannot be held across a GIL release, and
//! Rayon worker threads cannot safely access Python-managed memory.
//!
//! The overhead is one allocation + copy per input string, which is unavoidable with
//! the current PyO3 lifetime model.  Future work could reduce this via a two-phase
//! approach (validate / slice in GIL, allocate outside GIL) but the current approach
//! is simpler and correct.

use crate::core::{self, ChunkError, Config, KeepSeparator, Strategy};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

// ─── Chunk ───────────────────────────────────────────────────────────────────

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

// ─── TokenChunk ──────────────────────────────────────────────────────────────

#[cfg(feature = "tiktoken-rs")]
mod token_binding {
    use super::*;
    use crate::token_core::{self, Encoding, TokenChunkError, TokenConfig, TokenStrategy};

    /// A chunk produced by token-aware splitting.
    ///
    /// `start`/`end` are character offsets; `token_count` is the BPE token count
    /// of this chunk as measured by the configured encoding.
    #[pyclass(frozen, get_all, module = "chunkwise", skip_from_py_object)]
    #[derive(Clone, Debug)]
    pub struct TokenChunk {
        pub(super) text: String,
        pub(super) start: usize,
        pub(super) end: usize,
        pub(super) index: usize,
        pub(super) token_count: usize,
    }

    #[pymethods]
    impl TokenChunk {
        fn __repr__(&self) -> String {
            let preview: String = self.text.chars().take(40).collect();
            let dots = if self.text.chars().count() > 40 {
                "..."
            } else {
                ""
            };
            format!(
                "TokenChunk(index={}, start={}, end={}, token_count={}, text={:?}{})",
                self.index, self.start, self.end, self.token_count, preview, dots
            )
        }
        fn __len__(&self) -> usize {
            self.text.chars().count()
        }
    }

    impl From<token_core::TokenChunk> for TokenChunk {
        fn from(c: token_core::TokenChunk) -> Self {
            TokenChunk {
                text: c.text,
                start: c.start,
                end: c.end,
                index: c.index,
                token_count: c.token_count,
            }
        }
    }

    fn token_err(e: TokenChunkError) -> PyErr {
        PyValueError::new_err(e.to_string())
    }

    fn make_token_config(
        chunk_size: usize,
        chunk_overlap: usize,
        encoding_name: &str,
        strategy: &str,
        drop_whitespace: bool,
        keep_separator: &str,
    ) -> PyResult<TokenConfig> {
        let encoding: Encoding = encoding_name.parse().map_err(token_err)?;
        let strat: TokenStrategy = strategy.parse().map_err(token_err)?;
        let ks: KeepSeparator = keep_separator
            .parse()
            .map_err(|e: ChunkError| PyValueError::new_err(e.to_string()))?;
        let cfg = TokenConfig {
            chunk_size,
            chunk_overlap,
            encoding,
            strategy: strat,
            drop_whitespace,
            keep_separator: ks,
        };
        cfg.validate().map_err(token_err)?;
        Ok(cfg)
    }

    /// Split one string into token-sized chunks.
    ///
    /// Parameters
    /// ----------
    /// text : str
    ///     Source text to split.
    /// chunk_size : int
    ///     Maximum tokens per chunk (> 0).
    /// chunk_overlap : int
    ///     Token overlap between consecutive chunks (< chunk_size).
    /// encoding_name : str
    ///     BPE encoding: ``"cl100k_base"`` (gpt-4, gpt-3.5-turbo) or
    ///     ``"o200k_base"`` (gpt-4o, o1, later models).
    /// strategy : str
    ///     ``"recursive"`` (default) or ``"fixed"``.
    /// drop_whitespace : bool
    ///     Drop whitespace-only chunks (default ``True``).
    /// keep_separator : str
    ///     ``"end"`` (default), ``"start"``, or ``"false"``/``"discard"``.
    ///
    /// Returns
    /// -------
    /// list[TokenChunk]
    #[allow(clippy::too_many_arguments)]
    #[pyfunction]
    #[pyo3(signature = (
        text,
        chunk_size=512,
        chunk_overlap=64,
        encoding_name="cl100k_base",
        strategy="recursive",
        drop_whitespace=true,
        keep_separator="end"
    ))]
    pub fn chunk_text_tokens(
        py: Python<'_>,
        text: &str,
        chunk_size: usize,
        chunk_overlap: usize,
        encoding_name: &str,
        strategy: &str,
        drop_whitespace: bool,
        keep_separator: &str,
    ) -> PyResult<Vec<TokenChunk>> {
        let cfg = make_token_config(
            chunk_size,
            chunk_overlap,
            encoding_name,
            strategy,
            drop_whitespace,
            keep_separator,
        )?;
        let out = py
            .detach(|| token_core::chunk_tokens(text, &cfg))
            .map_err(token_err)?;
        Ok(out.into_iter().map(TokenChunk::from).collect())
    }

    /// Split many strings in parallel using token-aware chunking.
    ///
    /// The GIL is released while chunking. Input order is preserved.
    #[allow(clippy::too_many_arguments)]
    #[pyfunction]
    #[pyo3(signature = (
        texts,
        chunk_size=512,
        chunk_overlap=64,
        encoding_name="cl100k_base",
        strategy="recursive",
        drop_whitespace=true,
        keep_separator="end"
    ))]
    pub fn chunk_texts_tokens(
        py: Python<'_>,
        texts: Vec<String>,
        chunk_size: usize,
        chunk_overlap: usize,
        encoding_name: &str,
        strategy: &str,
        drop_whitespace: bool,
        keep_separator: &str,
    ) -> PyResult<Vec<Vec<TokenChunk>>> {
        let cfg = make_token_config(
            chunk_size,
            chunk_overlap,
            encoding_name,
            strategy,
            drop_whitespace,
            keep_separator,
        )?;
        let out = py
            .detach(|| token_core::chunk_texts_tokens(&texts, &cfg))
            .map_err(token_err)?;
        Ok(out
            .into_iter()
            .map(|v| v.into_iter().map(TokenChunk::from).collect())
            .collect())
    }

    pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
        m.add_class::<TokenChunk>()?;
        m.add_function(wrap_pyfunction!(chunk_text_tokens, m)?)?;
        m.add_function(wrap_pyfunction!(chunk_texts_tokens, m)?)?;
        Ok(())
    }
}

// ─── Error helpers ───────────────────────────────────────────────────────────

fn to_py_err(e: ChunkError) -> PyErr {
    PyValueError::new_err(e.to_string())
}

fn make_config(
    chunk_size: usize,
    chunk_overlap: usize,
    strategy: &str,
    drop_whitespace: bool,
    keep_separator: &str,
) -> PyResult<Config> {
    let strategy: Strategy = strategy.parse().map_err(to_py_err)?;
    let ks: KeepSeparator = keep_separator.parse().map_err(to_py_err)?;
    let cfg = Config {
        chunk_size,
        chunk_overlap,
        strategy,
        keep_separator: ks,
        drop_whitespace,
    };
    cfg.validate().map_err(to_py_err)?;
    Ok(cfg)
}

// ─── chunk_text ──────────────────────────────────────────────────────────────

/// Split one string into chunks.
///
/// Parameters
/// ----------
/// text : str
///     Source text to split.
/// chunk_size : int
///     Maximum characters per chunk (> 0).
/// chunk_overlap : int
///     Character overlap between consecutive chunks (< chunk_size).
/// strategy : str
///     ``"recursive"`` (default), ``"sentence"``, or ``"fixed"``.
/// drop_whitespace : bool
///     Drop whitespace-only chunks (default ``True``).
/// keep_separator : str
///     ``"end"`` (default), ``"start"``, or ``"false"``/``"discard"``.
///
/// Returns
/// -------
/// list[Chunk]
///     Each chunk satisfies ``source[c.start:c.end] == c.text``.
#[allow(clippy::too_many_arguments)]
#[pyfunction]
#[pyo3(signature = (
    text,
    chunk_size=1000,
    chunk_overlap=200,
    strategy="recursive",
    drop_whitespace=true,
    keep_separator="end"
))]
fn chunk_text(
    py: Python<'_>,
    text: &str,
    chunk_size: usize,
    chunk_overlap: usize,
    strategy: &str,
    drop_whitespace: bool,
    keep_separator: &str,
) -> PyResult<Vec<Chunk>> {
    let cfg = make_config(
        chunk_size,
        chunk_overlap,
        strategy,
        drop_whitespace,
        keep_separator,
    )?;
    let out = py.detach(|| core::chunk(text, &cfg)).map_err(to_py_err)?;
    Ok(out.into_iter().map(Chunk::from).collect())
}

// ─── chunk_texts ─────────────────────────────────────────────────────────────

/// Split many strings in parallel. The GIL is released while chunking.
///
/// Note: all input strings are cloned into Rust-owned memory before the GIL is
/// released. This is one copy per string and is unavoidable given PyO3's lifetime
/// rules for GIL-released parallelism.
///
/// Parameters
/// ----------
/// texts : list[str]
///     Documents to split.
/// chunk_size : int
///     Maximum characters per chunk (> 0).
/// chunk_overlap : int
///     Character overlap (< chunk_size).
/// strategy : str
///     ``"recursive"`` (default), ``"sentence"``, or ``"fixed"``.
/// drop_whitespace : bool
///     Drop whitespace-only chunks (default ``True``).
/// keep_separator : str
///     ``"end"`` (default), ``"start"``, or ``"false"``/``"discard"``.
///
/// Returns
/// -------
/// list[list[Chunk]]
///     ``results[i]`` corresponds to ``texts[i]``; input order is preserved.
#[allow(clippy::too_many_arguments)]
#[pyfunction]
#[pyo3(signature = (
    texts,
    chunk_size=1000,
    chunk_overlap=200,
    strategy="recursive",
    drop_whitespace=true,
    keep_separator="end"
))]
fn chunk_texts(
    py: Python<'_>,
    texts: Vec<String>,
    chunk_size: usize,
    chunk_overlap: usize,
    strategy: &str,
    drop_whitespace: bool,
    keep_separator: &str,
) -> PyResult<Vec<Vec<Chunk>>> {
    let cfg = make_config(
        chunk_size,
        chunk_overlap,
        strategy,
        drop_whitespace,
        keep_separator,
    )?;
    let out = py
        .detach(|| core::chunk_many(&texts, &cfg))
        .map_err(to_py_err)?;
    Ok(out
        .into_iter()
        .map(|v| v.into_iter().map(Chunk::from).collect())
        .collect())
}

// ─── Module init ─────────────────────────────────────────────────────────────

#[pymodule]
#[pyo3(name = "_core")]
fn init(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Chunk>()?;
    m.add_function(wrap_pyfunction!(chunk_text, m)?)?;
    m.add_function(wrap_pyfunction!(chunk_texts, m)?)?;
    #[cfg(feature = "tiktoken-rs")]
    token_binding::register(m)?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
