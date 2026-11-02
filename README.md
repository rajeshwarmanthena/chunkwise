# chunkwise

Fast, deterministic, parallel text chunking for RAG pipelines. Rust core, Python bindings.

- **Traceable:** every chunk carries character offsets, so `source[c.start:c.end] == c.text`. Useful for citations and audits.
- **Fast:** roughly 12x faster than LangChain's `RecursiveCharacterTextSplitter` on one core in our benchmark (see below).
- **Parallel:** `chunk_texts` splits many documents across all cores and releases the GIL.
- **Safe by construction:** no chunk exceeds `chunk_size`, Unicode is handled correctly, and the algorithm always terminates. These properties are covered by tests.
- **One wheel per platform:** abi3 wheels for Python 3.10+. No Rust toolchain needed to install.

## Install

```bash
pip install chunkwise   # once published; until then: pip install maturin && maturin develop --release
```

## Usage

```python
from chunkwise import chunk_text, chunk_texts

chunks = chunk_text(doc, chunk_size=1000, chunk_overlap=200, strategy="recursive")
for c in chunks:
    print(c.index, c.start, c.end, c.text[:60])

# many documents, in parallel, order preserved
batches = chunk_texts(list_of_docs, chunk_size=800, chunk_overlap=100)
```

| strategy | behaviour |
|---|---|
| `recursive` (default) | prefers paragraph, then line, then word boundaries, then characters |
| `sentence` | packs whole sentences (Unicode UAX #29) up to `chunk_size` |
| `fixed` | fixed windows of `chunk_size` characters |

Sizes are measured in characters. Invalid arguments raise `ValueError`. Chunks keep the original whitespace; nothing is rewritten.

## Use from Rust

```toml
chunkwise = "0.1"
```
```rust
use chunkwise::core::{chunk, Config, Strategy};
let cfg = Config { chunk_size: 1000, chunk_overlap: 200, strategy: Strategy::Recursive };
let chunks = chunk(text, &cfg)?;
```

## Benchmark

`python benchmarks/bench.py` (1,000 docs, 20 MB, size 1000 / overlap 200, median of 5 runs, **single-core sandbox**):

| | median | throughput |
|---|---|---|
| chunkwise | 0.11 s | ~180 MB/s |
| LangChain RecursiveCharacterTextSplitter | 1.47 s | ~14 MB/s |

That is roughly 13x on one core. Algorithms are similar, not identical, so compare speed rather than exact chunk boundaries. The multi-core gain of `chunk_texts` could not be measured on a 1-CPU machine; run the script on your hardware before quoting numbers. Method and validity rules: [docs/benchmarks.md](docs/benchmarks.md).

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

See [CONTRIBUTING.md](CONTRIBUTING.md) and [SECURITY.md](SECURITY.md). Licensed under Apache-2.0.
