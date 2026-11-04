# Changelog

All notable changes to this project will be documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2026-11-04

### Added
- **Rust-native token-aware chunking** (`token_core` module):
  - Added BPE token-aware chunking via `tiktoken-rs` (supporting `cl100k_base` and `o200k_base`).
  - Added `chunk_text_tokens` and `chunk_texts_tokens` in Python and `chunk_tokens` / `chunk_texts_tokens` in Rust.
  - Guaranteed `token_count <= chunk_size` for every returned chunk by re-encoding candidate strings (preventing non-compositional BPE merge overflow).
  - Every `TokenChunk` retains exact original source character offsets: `source[chunk.start:chunk.end] == chunk.text`.
  - Added `token_count` metadata property on returned `TokenChunk` objects.
  - Singleton BPE caching: `&'static CoreBPE` tokenizers initialize once per process.
- **Configurable whitespace handling**:
  - Added `drop_whitespace` boolean configuration option (default `true` preserves legacy behavior).
  - When `drop_whitespace = false`, whitespace-only segments are retained, enabling mathematically verified lossless partitions when `overlap = 0`.
  - Added `verify_coverage` Rust helper to mathematically assert gap-free and overlap-free reconstruction.
- **Configurable recursive separator attachment** (`keep_separator`):
  - Added `"end"` (default, attaches delimiter to preceding segment).
  - Added `"start"` (attaches delimiter to succeeding segment).
  - Added `"false"` / `"none"` / `"discard"` (discards delimiter; documented as non-lossless).
- **Comprehensive testing and fuzzing**:
  - Added `chunk_tokens` libfuzzer target (`fuzz/fuzz_targets/chunk_tokens.rs`).
  - Added 30+ integration test cases covering Unicode scripts (CJK, Arabic, Devanagari, Thai, Cyrillic), emojis, CRLF, combining marks, pathological overlaps, and LangChain parity comparison.
- **Enhanced benchmark harness** (`benchmarks/bench.py`):
  - Reports median, min, max, standard deviation, throughput (MB/s), peak RSS memory, chunk counts, and speedups against LangChain.

### Changed
- Bumped version to `0.2.0` due to new API signatures and optional dependencies.
- Updated `Cargo.toml` with optional `tiktoken-rs` feature (enabled by default in Python wheels via Maturin).
- Updated `deny.toml` license configuration to explicitly allow `Unlicense` (used by transitive dependency `rustc-hash`).
- Updated CI workflow to gate on `cargo clippy --all-targets --all-features -- -D warnings` and `cargo test --all-features`.

## [0.1.0] - 2026-03-01
- Initial release: `fixed`, `sentence` and `recursive` character chunking strategies.
- Character offset guarantee: `source[chunk.start:chunk.end] == chunk.text`.
- Parallel batch API (`chunk_texts`) releasing Python's GIL.
- Python bindings with abi3 wheels for Python 3.10+.
