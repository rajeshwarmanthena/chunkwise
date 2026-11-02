//! chunkwise: fast, deterministic text chunking for RAG pipelines.
//!
//! The engine lives in [`core`] and is plain Rust. Enable the `python` feature
//! (done automatically by maturin) to build the Python extension module.

pub mod core;

#[cfg(feature = "python")]
mod python;
