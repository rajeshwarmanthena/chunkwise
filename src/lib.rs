//! chunkwise: fast, deterministic text chunking for RAG pipelines.
//!
//! The character-based engine lives in [`core`] and is plain Rust with no optional
//! dependencies.  Token-aware splitting lives in [`token_core`] and requires the
//! `tiktoken-rs` Cargo feature.  Python bindings are built with the `python` feature
//! (done automatically by maturin).
//!
//! ## Feature flags
//!
//! | feature | adds |
//! |---|---|
//! | `python` | PyO3 extension module (`_core`) |
//! | `tiktoken-rs` | token-aware chunking via `tiktoken-rs` |

pub mod core;
pub mod token_core;

#[cfg(feature = "python")]
mod python;
