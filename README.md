# chunkwise

**Fast, deterministic, parallel text chunking for RAG pipelines.** Rust core, Python bindings.

[![CI](https://github.com/rajeshwarmanthena/chunkwise/actions/workflows/ci.yml/badge.svg)](https://github.com/rajeshwarmanthena/chunkwise/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![Rust](https://img.shields.io/badge/rust-1.75%2B-orange?logo=rust)
![License](https://img.shields.io/badge/license-Apache--2.0-green)

`chunkwise` splits long documents into chunks for embedding and retrieval. Every chunk carries character offsets back into the source, so citations and audits are exact — `source[c.start:c.end] == c.text`, always.

## Why chunkwise?

| | |
|---|---|
| **Traceable** | Every chunk carries character offsets: `source[c.start:c.end] == c.text`. Useful for citations and audits. |
| **Fast** | Roughly 13x faster than LangChain's `RecursiveCharacterTextSplitter` on one core in our benchmark (see [Performance](#performance)). |
| **Parallel** | `chunk_texts` splits many documents across all cores and releases the GIL. |
| **Safe by construction** | No chunk exceeds `chunk_size`, Unicode is handled correctly, and the algorithm always terminates. These properties are covered by tests and fuzzing. |
| **One wheel per platform** | abi3 wheels for Python 3.10+. No Rust toolchain needed to install. |

## Contents

- [Installation](#installation)
- [Quick start](#quick-start)
- [API](#api)
- [Strategies](#strategies)
- [Use from Rust](#use-from-rust)
- [Performance](#performance)
- [Development](#development)
- [Quality and supply chain](#quality-and-supply-chain)
- [Roadmap](#roadmap)
- [Contributing and security](#contributing-and-security)
- [License](#license)

## Installation

```bash
pip install chunkwise   # once published
```

Until the package is on PyPI, build and install from source (requires a Rust toolchain):

```bash
pip install maturin
maturin develop --release
```

## Quick start

```python
from chunkwise import chunk_text, chunk_texts

chunks = chunk_text(doc, chunk_size=1000, chunk_overlap=200, strategy="recursive")
for c in chunks:
    print(c.index, c.start, c.end, c.text[:60])

# many documents, in parallel, order preserved
batches = chunk_texts(list_of_docs, chunk_size=800, chunk_overlap=100)
```

## API

### `chunk_text(text, chunk_size=1000, chunk_overlap=200, strategy="recursive") -> list[Chunk]`

Split a single string.

### `chunk_texts(texts, chunk_size=1000, chunk_overlap=200, strategy="recursive") -> list[list[Chunk]]`

Split a sequence of strings in parallel across all cores. Input order is preserved: `batches[i]` corresponds to `texts[i]`.

### `Chunk`

| attribute | meaning |
|---|---|
| `text` | the chunk's text, exactly `source[start:end]` |
| `start` | character offset into the source, inclusive |
| `end` | character offset into the source, exclusive |
| `index` | position of the chunk within its document, starting at 0 |
| `len(chunk)` | number of characters in the chunk |

### Semantics

- Sizes are measured in **characters**, not bytes or tokens.
- `ValueError` if `chunk_size <= 0`, `chunk_overlap >= chunk_size`, or `strategy` is unknown.
- Chunks keep the original whitespace; nothing is rewritten.
- Whitespace-only chunks are dropped.

## Strategies

| strategy | behaviour |
|---|---|
| `recursive` (default) | prefers paragraph, then line, then word boundaries, then characters |
| `sentence` | packs whole sentences (Unicode UAX #29) up to `chunk_size` |
| `fixed` | fixed windows of `chunk_size` characters |

## Use from Rust

```toml
[dependencies]
chunkwise = "0.1"
```

```rust
use chunkwise::core::{chunk, Config, Strategy};

let cfg = Config { chunk_size: 1000, chunk_overlap: 200, strategy: Strategy::Recursive };
let chunks = chunk(text, &cfg)?;
```

## Performance

`python benchmarks/bench.py` (1,000 docs, 20 MB, size 1000 / overlap 200, median of 5 runs, **single-core sandbox**):

| | median | throughput |
|---|---|---|
| chunkwise | 0.11 s | ~180 MB/s |
| LangChain `RecursiveCharacterTextSplitter` | 1.47 s | ~14 MB/s |

That is roughly **13x** on one core. Algorithms are similar, not identical, so compare speed rather than exact chunk boundaries. The multi-core gain of `chunk_texts` could not be measured on a 1-CPU machine; run the script on your hardware before quoting numbers. Method and validity rules: [docs/benchmarks.md](docs/benchmarks.md).

## Development

Prerequisites: a Rust toolchain ([rustup](https://rustup.rs)) and Python 3.10+.

```bash
git clone https://github.com/rajeshwarmanthena/chunkwise
cd chunkwise
pip install -e ".[dev]"   # builds the native chunkwise._core module and installs pytest
pytest
```

> **Note:** the editable install compiles the Rust extension via maturin. Skipping it is the usual cause of `ModuleNotFoundError: No module named 'chunkwise._core'` on a fresh checkout.

Rust-side checks (the same gates CI runs):

```bash
cargo fmt --check
cargo clippy --all-targets -- -D warnings
cargo test
```

Optional fuzzing (requires nightly):

```bash
cargo +nightly fuzz run chunk -- -max_total_time=60
```

## Quality and supply chain

- Property-style tests on random multilingual text (offsets, size limit, termination, parallel equals sequential).
- A cargo-fuzz target (`fuzz/`) that checks the same invariants on arbitrary input.
- `clippy -D warnings`, `rustfmt`, and `cargo-deny` (advisories, licenses, sources) in CI.
- Releases publish via PyPI trusted publishing with signed build-provenance attestations.

## Roadmap

- Token-based sizing (tiktoken-compatible and custom tokenizers)
- Markdown and code-aware strategies
- SBOM generation for releases
- Streaming API for very large files

## Contributing and security

See [CONTRIBUTING.md](CONTRIBUTING.md) and [SECURITY.md](SECURITY.md).

## License

[Apache-2.0](LICENSE)
