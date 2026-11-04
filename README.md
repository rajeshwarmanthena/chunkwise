# chunkwise

**Fast, deterministic, parallel text chunking for RAG pipelines.** Rust core, Python bindings.

[![CI](https://github.com/rajeshwarmanthena/chunkwise/actions/workflows/ci.yml/badge.svg)](https://github.com/rajeshwarmanthena/chunkwise/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![Rust](https://img.shields.io/badge/rust-1.75%2B-orange?logo=rust)
![License](https://img.shields.io/badge/license-Apache--2.0-green)

`chunkwise` splits long documents into chunks for embedding and retrieval. Every chunk carries Unicode scalar-value (character) offsets back into the source document, so citations, highlight spans, and audits are exact — `source[c.start:c.end] == c.text`, always.

---
`chunkwise` splits long documents into chunks for embedding and retrieval. Every chunk carries Unicode scalar-value (character) offsets back into the source document, so citations, highlight spans, and audits are exact — `source[c.start:c.end] == c.text`, always.

---

## Why chunkwise?

| | |
|---|---|
| **Traceable** | Every chunk carries Python character offsets: `source[c.start:c.end] == c.text`. Offsets are never modified or silently trimmed. |
| **Token-aware in Rust** | Native BPE token chunking (`cl100k_base`, `o200k_base`) via `tiktoken-rs`. Chunks are guaranteed to satisfy `actual_tokens <= chunk_size`. |
| **Configurable Delimiters** | Recursive splitting supports `keep_separator` modes: `"end"` (default), `"start"`, or `"discard"`. |
| **Lossless-Ready** | Optional `drop_whitespace=False` retains whitespace-only segments, enabling verified lossless zero-overlap partitioning. |
| **Fast & Parallel** | Multi-threaded document splitting via Rayon releases Python's GIL. Over 5x–20x faster than LangChain's `RecursiveCharacterTextSplitter`. |
| **Safe by Construction** | Unicode scalar-value safety (emojis, CJK, CRLF, combining marks), no panics on empty or pathological input, proven by property tests and fuzz targets. |

---
| **Traceable** | Every chunk carries Python character offsets: `source[c.start:c.end] == c.text`. Offsets are never modified or silently trimmed. |
| **Token-aware in Rust** | Native BPE token chunking (`cl100k_base`, `o200k_base`) via `tiktoken-rs`. Chunks are guaranteed to satisfy `actual_tokens <= chunk_size`. |
| **Configurable Delimiters** | Recursive splitting supports `keep_separator` modes: `"end"` (default), `"start"`, or `"discard"`. |
| **Lossless-Ready** | Optional `drop_whitespace=False` retains whitespace-only segments, enabling verified lossless zero-overlap partitioning. |
| **Fast & Parallel** | Multi-threaded document splitting via Rayon releases Python's GIL. Over 5x–20x faster than LangChain's `RecursiveCharacterTextSplitter`. |
| **Safe by Construction** | Unicode scalar-value safety (emojis, CJK, CRLF, combining marks), no panics on empty or pathological input, proven by property tests and fuzz targets. |

---

## Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Python API Reference](#python-api-reference)
- [Rust API Reference](#rust-api-reference)
- [Whitespace & Losslessness Semantics](#whitespace--losslessness-semantics)
- [Separator Attachment Modes](#separator-attachment-modes)
- [Token-Aware Chunking](#token-aware-chunking)
- [Performance Benchmarks](#performance-benchmarks)
- [Migration Guide (v0.1.0 to v0.2.0)](#migration-guide-v010-to-v020)
- [Development & Testing](#development--testing)
- [Quick Start](#quick-start)
- [Python API Reference](#python-api-reference)
- [Rust API Reference](#rust-api-reference)
- [Whitespace & Losslessness Semantics](#whitespace--losslessness-semantics)
- [Separator Attachment Modes](#separator-attachment-modes)
- [Token-Aware Chunking](#token-aware-chunking)
- [Performance Benchmarks](#performance-benchmarks)
- [Migration Guide (v0.1.0 to v0.2.0)](#migration-guide-v010-to-v020)
- [Development & Testing](#development--testing)
- [License](#license)

---

---

## Installation

### From PyPI (once published)

### From PyPI (once published)

```bash
pip install chunkwise
pip install chunkwise
```

### From Source

Requires Python 3.10+ and a Rust toolchain (Rust 1.75+):
### From Source

Requires Python 3.10+ and a Rust toolchain (Rust 1.75+):

```bash
pip install maturin
maturin develop --release
```

---

## Quick Start

### 1. Character-Based Chunking
---

## Quick Start

### 1. Character-Based Chunking

```python
from chunkwise import chunk_text, chunk_texts

doc = "First paragraph.\n\nSecond paragraph with multiple sentences. Another sentence."

# Single document
chunks = chunk_text(doc, chunk_size=100, chunk_overlap=20, strategy="recursive")
for c in chunks:
    print(f"[{c.index}] {c.start}:{c.end} -> {c.text!r}")
    assert doc[c.start:c.end] == c.text  # Exact slice match

# Many documents in parallel (GIL released across all CPU cores)
batches = chunk_texts([doc] * 50, chunk_size=100, chunk_overlap=20)
```

### 2. Token-Aware Chunking (OpenAI BPE)

```python
from chunkwise import chunk_text_tokens, chunk_texts_tokens

# Split by BPE tokens (cl100k_base for GPT-4 / 3.5, o200k_base for GPT-4o / o1)
chunks = chunk_text_tokens(
    doc,
    chunk_size=256,
    chunk_overlap=32,
    encoding_name="cl100k_base",
    strategy="recursive",
)
doc = "First paragraph.\n\nSecond paragraph with multiple sentences. Another sentence."

# Single document
chunks = chunk_text(doc, chunk_size=100, chunk_overlap=20, strategy="recursive")
for c in chunks:
    print(f"[{c.index}] {c.start}:{c.end} -> {c.text!r}")
    assert doc[c.start:c.end] == c.text  # Exact slice match

# Many documents in parallel (GIL released across all CPU cores)
batches = chunk_texts([doc] * 50, chunk_size=100, chunk_overlap=20)
```

### 2. Token-Aware Chunking (OpenAI BPE)

```python
from chunkwise import chunk_text_tokens, chunk_texts_tokens

# Split by BPE tokens (cl100k_base for GPT-4 / 3.5, o200k_base for GPT-4o / o1)
chunks = chunk_text_tokens(
    doc,
    chunk_size=256,
    chunk_overlap=32,
    encoding_name="cl100k_base",
    strategy="recursive",
)
for c in chunks:
    print(f"[{c.index}] tokens={c.token_count} chars={c.start}:{c.end} -> {c.text[:40]}...")
    assert doc[c.start:c.end] == c.text
```

---

## Python API Reference

### `chunk_text`

```python
chunk_text(
    text: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    strategy: str = "recursive",
    drop_whitespace: bool = True,
    keep_separator: str = "end",
) -> list[Chunk]
```

- **`text`**: Document string to split.
- **`chunk_size`**: Maximum number of characters per chunk (`> 0`).
- **`chunk_overlap`**: Number of characters to carry over between consecutive chunks (`< chunk_size`).
- **`strategy`**: Splitting strategy: `"recursive"` (default), `"sentence"`, or `"fixed"`.
- **`drop_whitespace`**: When `True` (default), chunks containing only whitespace are omitted. When `False`, whitespace chunks are retained.
- **`keep_separator`**: Where to attach separators in `"recursive"` mode: `"end"` (default), `"start"`, or `"discard"` / `"false"`.
    print(f"[{c.index}] tokens={c.token_count} chars={c.start}:{c.end} -> {c.text[:40]}...")
    assert doc[c.start:c.end] == c.text
```

---

## Python API Reference

### `chunk_text`

```python
chunk_text(
    text: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    strategy: str = "recursive",
    drop_whitespace: bool = True,
    keep_separator: str = "end",
) -> list[Chunk]
```

- **`text`**: Document string to split.
- **`chunk_size`**: Maximum number of characters per chunk (`> 0`).
- **`chunk_overlap`**: Number of characters to carry over between consecutive chunks (`< chunk_size`).
- **`strategy`**: Splitting strategy: `"recursive"` (default), `"sentence"`, or `"fixed"`.
- **`drop_whitespace`**: When `True` (default), chunks containing only whitespace are omitted. When `False`, whitespace chunks are retained.
- **`keep_separator`**: Where to attach separators in `"recursive"` mode: `"end"` (default), `"start"`, or `"discard"` / `"false"`.

### `chunk_texts`

```python
chunk_texts(
    texts: list[str],
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    strategy: str = "recursive",
    drop_whitespace: bool = True,
    keep_separator: str = "end",
) -> list[list[Chunk]]
```
### `chunk_texts`

```python
chunk_texts(
    texts: list[str],
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    strategy: str = "recursive",
    drop_whitespace: bool = True,
    keep_separator: str = "end",
) -> list[list[Chunk]]
```

Splits a list of strings in parallel using Rayon worker threads. Releases Python's GIL. Preserves input document ordering.
Splits a list of strings in parallel using Rayon worker threads. Releases Python's GIL. Preserves input document ordering.

### `chunk_text_tokens`

```python
chunk_text_tokens(
    text: str,
    chunk_size: int = 512,
    chunk_overlap: int = 64,
    encoding_name: str = "cl100k_base",
    strategy: str = "recursive",
    drop_whitespace: bool = True,
    keep_separator: str = "end",
) -> list[TokenChunk]
```

Splits text into chunks measuring at most `chunk_size` tokens.

- **`encoding_name`**: `"cl100k_base"` (GPT-4, GPT-3.5-Turbo, text-embedding-ada-002) or `"o200k_base"` (GPT-4o, o1, o3).
- **`strategy`**: `"recursive"` (prefers structural boundaries) or `"fixed"` (splits on token limits).

### `chunk_texts_tokens`

```python
chunk_texts_tokens(
    texts: list[str],
    chunk_size: int = 512,
    chunk_overlap: int = 64,
    encoding_name: str = "cl100k_base",
    strategy: str = "recursive",
    drop_whitespace: bool = True,
    keep_separator: str = "end",
) -> list[list[TokenChunk]]
```

Batch token-aware chunking in parallel across all CPU cores with released GIL.
### `chunk_text_tokens`

```python
chunk_text_tokens(
    text: str,
    chunk_size: int = 512,
    chunk_overlap: int = 64,
    encoding_name: str = "cl100k_base",
    strategy: str = "recursive",
    drop_whitespace: bool = True,
    keep_separator: str = "end",
) -> list[TokenChunk]
```

Splits text into chunks measuring at most `chunk_size` tokens.

- **`encoding_name`**: `"cl100k_base"` (GPT-4, GPT-3.5-Turbo, text-embedding-ada-002) or `"o200k_base"` (GPT-4o, o1, o3).
- **`strategy`**: `"recursive"` (prefers structural boundaries) or `"fixed"` (splits on token limits).

### `chunk_texts_tokens`

```python
chunk_texts_tokens(
    texts: list[str],
    chunk_size: int = 512,
    chunk_overlap: int = 64,
    encoding_name: str = "cl100k_base",
    strategy: str = "recursive",
    drop_whitespace: bool = True,
    keep_separator: str = "end",
) -> list[list[TokenChunk]]
```

Batch token-aware chunking in parallel across all CPU cores with released GIL.

### Object Structures

#### `Chunk`
- `text`: Substring slice of the original text.
- `start`: Inclusive character offset into the source string (`int`).
- `end`: Exclusive character offset into the source string (`int`).
- `index`: Sequential position within the document (`int`, starts at 0).
- `len(chunk)`: Number of Unicode characters in the chunk.

#### `TokenChunk`
- Inherits all properties of `Chunk` (`text`, `start`, `end`, `index`, `len()`).
- `token_count`: Exact number of tokens in this chunk as verified by the tokenizer (`int`, guaranteed `<= chunk_size`).

---

## Rust API Reference

Add to `Cargo.toml`:
### Object Structures

#### `Chunk`
- `text`: Substring slice of the original text.
- `start`: Inclusive character offset into the source string (`int`).
- `end`: Exclusive character offset into the source string (`int`).
- `index`: Sequential position within the document (`int`, starts at 0).
- `len(chunk)`: Number of Unicode characters in the chunk.

#### `TokenChunk`
- Inherits all properties of `Chunk` (`text`, `start`, `end`, `index`, `len()`).
- `token_count`: Exact number of tokens in this chunk as verified by the tokenizer (`int`, guaranteed `<= chunk_size`).

---

## Rust API Reference

Add to `Cargo.toml`:

```toml
[dependencies]
chunkwise = { version = "0.2.0", features = ["tiktoken-rs"] }
```

### Character Chunking

```rust
use chunkwise::core::{chunk, chunk_many, Config, Strategy, KeepSeparator};

let cfg = Config {
    chunk_size: 1000,
    chunk_overlap: 200,
    strategy: Strategy::Recursive,
    keep_separator: KeepSeparator::End,
    drop_whitespace: true,
};

let chunks = chunk("Your document text...", &cfg)?;
for c in chunks {
    println!("Chunk {}: {}..{} ({} chars)", c.index, c.start, c.end, c.text.len());
}
```

### Token Chunking

```rust
use chunkwise::token_core::{chunk_tokens, TokenConfig, TokenStrategy, Encoding};
use chunkwise::core::KeepSeparator;

let cfg = TokenConfig {
    chunk_size: 256,
    chunk_overlap: 32,
    encoding: Encoding::Cl100kBase,
    strategy: TokenStrategy::Recursive,
    drop_whitespace: true,
    keep_separator: KeepSeparator::End,
};

let chunks = chunk_tokens("Your document text...", &cfg)?;
for c in chunks {
    println!("Chunk {}: {} tokens, {}..{}", c.index, c.token_count, c.start, c.end);
}
```

### Verifying Partition Completeness

To verify that zero-overlap chunks cleanly partition the text without gaps or dropped characters:

```rust
use chunkwise::core::{chunk, verify_coverage, Config, Strategy, KeepSeparator};

let text = "First paragraph.\n\nSecond paragraph.";
let cfg = Config {
    chunk_size: 20,
    chunk_overlap: 0,
    strategy: Strategy::Recursive,
    keep_separator: KeepSeparator::End,
    drop_whitespace: false, // Must be false to guarantee zero-gap partition
};

let chunks = chunk(text, &cfg)?;
assert!(verify_coverage(text, &chunks).is_ok());
```

---

## Memory Architecture & GIL-Release Design

### Single-Document Zero-Copy Borrowing
In `chunk_text()` / `chunk_text_tokens()`, Python's `str` is passed directly as a borrowed `&str` reference into Rust. No intermediate string buffer is allocated. The single input text lives in Python's heap while Rust performs zero-copy slicing.

### Parallel Batch Execution (`chunk_texts`, `chunk_texts_tokens`)
1. **Ownership Hand-off**: PyO3 copies the batch of Python strings into Rust `String` instances while holding the Global Interpreter Lock (GIL). This single copy per document is necessary because borrowed Python pointers cannot safely cross thread boundaries after releasing the GIL (preventing data races or garbage-collection during free-threaded execution).
2. **GIL Released**: The GIL is detached during Rayon parallel execution (`py.detach(...)`), enabling true multi-core hardware scaling across all available CPU threads.
3. **Deterministic Order**: Results are returned in exact document order, matching sequential execution bit-for-bit.
4. **Low Peak Memory**: Peak resident memory remains bounded and significantly lower than pure-Python splitters (see benchmarks below).

---

## Whitespace & Losslessness Semantics

### Is chunking lossless?

| Configuration | Is it lossless? | Explanation |
|---|---|---|
| `overlap = 0`, `drop_whitespace = False` | **Yes** | Chunks form an exact partition: `"".join(c.text for c in chunks) == source`. |
| `overlap = 0`, `drop_whitespace = True` (default) | **No** (if whitespace gaps exist) | Whitespace-only regions (e.g. `\n\n\n\n`) are dropped, leaving index gaps. |
| `overlap > 0` (any setting) | **No** | Overlapping characters are repeated in adjacent chunks. |
| `keep_separator = "discard"` | **No** | Separator characters are discarded from the output chunks. |

> [!IMPORTANT]
> Naive string concatenation `"".join(c.text for c in chunks)` is **not** a general inverse of chunking when `overlap > 0` or delimiters are dropped. To verify partition completeness without overlap, check `verify_coverage` or ensure `overlap = 0` and `drop_whitespace = False`.

---

## Separator Attachment Modes

When using `strategy="recursive"`, separators (`\n\n`, `\n`, ` `) are split progressively. The `keep_separator` parameter dictates delimiter placement:

- **`"end"` (default)**: The separator is attached to the *preceding* chunk (e.g., `["paragraph one.\n\n", "paragraph two."]`).
- **`"start"`**: The separator is attached to the *succeeding* chunk (e.g., `["paragraph one.", "\n\nparagraph two."]`).
- **`"discard"` / `"false"`**: The delimiter is completely removed from all chunks. Offsets still accurately reflect the retained substrings.

---

## Token-Aware Chunking

### Non-Compositional BPE Safety

In Byte Pair Encoding (BPE), tokenization is context-sensitive:
$$\text{count}(A + B) \neq \text{count}(A) + \text{count}(B)$$

Merging two substrings can cause boundary byte pairs to combine into a single token or re-tokenize differently. `chunkwise` ensures safety by **encoding the complete candidate chunk** rather than summing pre-tokenized segment counts. Every returned `TokenChunk` is verified to satisfy:
$$\text{c.token\_count} \le \text{chunk\_size}$$

### UTF-8 Character Boundary Alignment

Individual BPE tokens may represent partial UTF-8 bytes (for example, a single byte of a 4-byte emoji). `chunkwise` maps token boundaries back to verified Unicode scalar-value boundaries in the original text, ensuring that every chunk text is a valid UTF-8 substring with valid Python string character offsets.

---

## Performance Benchmarks

Measured on Apple Silicon (M-series, 8 logical cores, macOS 15, Python 3.12, 100 documents, 0.52 MB UTF-8 corpus):

| Splitter | Median Time | Throughput | Peak RSS | Multi-core Speedup |
|---|---|---|---|---|
| **chunkwise (single-thread char)** | **0.003s** | **182.4 MB/s** | **24.4 MB** | 1.0x (5.5x vs LangChain) |
| **chunkwise (parallel Rayon char)** | **0.001s** | **705.5 MB/s** | **26.7 MB** | 3.8x (21.2x vs LangChain) |
| **chunkwise (single-thread token)** | 0.584s | 0.9 MB/s | 53.4 MB | 1.0x |
| **chunkwise (parallel Rayon token)**| 0.233s | 2.2 MB/s | 53.4 MB | 2.5x |
| LangChain `RecursiveCharacterTextSplitter` | 0.016s | 33.3 MB/s | 78.4 MB | Baseline (1 core) |

### LangChain Comparison & Behavioral Differences

| Dimension | `chunkwise` | LangChain `RecursiveCharacterTextSplitter` |
|---|---|---|
| **Offset Traceability** | **Exact:** `source[c.start:c.end] == c.text` for 100% of chunks. Ideal for citations and audit logs. | **Approximate / None:** Delimiters and whitespace are often stripped or normalized, breaking slice equality. |
| **Execution Engine** | Native compiled **Rust** core. | Pure Python string matching with regexes. |
| **Parallelism** | Releases GIL; scales across all available CPU cores via Rayon (**>700 MB/s**). | Single-threaded; bound by Python's Global Interpreter Lock (GIL). |
| **Peak Memory (RSS)** | **~26 MB** (bounded Rust heap allocations). | **~78 MB** (>3x higher memory overhead). |
| **Token Sizing** | Native BPE verification (`cl100k_base`, `o200k_base`). Guaranteed `token_count <= chunk_size`. | Relies on Python tiktoken callbacks or length estimators. |

*Run the benchmark on your machine:*
```bash
python benchmarks/bench.py --docs 500 --doc-chars 10000 --repeat 5
```

---

## Migration Guide (v0.1.0 to v0.2.0)

Version `0.2.0` is backwards-compatible with existing calls to `chunk_text()` and `chunk_texts()`.

1. **Token Chunking**: Use `chunk_text_tokens()` and `chunk_texts_tokens()` to measure sizes in BPE tokens rather than characters.
2. **Separator Placement**: If you previously relied on separators being appended to chunks, this remains the default (`keep_separator="end"`). You can now specify `keep_separator="start"` or `"discard"`.
3. **Lossless Partitioning**: If your pipeline requires zero-overlap source reconstruction, pass `chunk_overlap=0` and `drop_whitespace=False`.

---

## Development & Testing

### Prerequisites
- Rust 1.75+ (`rustup default stable`)
- Python 3.10+

### Setup
```bash
git clone https://github.com/rajeshwarmanthena/chunkwise.git
git clone https://github.com/rajeshwarmanthena/chunkwise.git
cd chunkwise
pip install -e ".[dev]"
pip install -e ".[dev]"
```

### Running Test Suites
### Running Test Suites
```bash
# Rust unit & integration tests
cargo test
cargo test --features tiktoken-rs

# Python integration tests
pytest -v

# Linter & formatter checks
cargo fmt --check
cargo clippy --all-targets --all-features -- -D warnings
cargo deny check
cargo clippy --all-targets --all-features -- -D warnings
cargo deny check

# Fuzz testing (nightly toolchain)
# Fuzz testing (nightly toolchain)
cargo +nightly fuzz run chunk -- -max_total_time=60
cargo +nightly fuzz run chunk_tokens -- -max_total_time=60
```

---
cargo +nightly fuzz run chunk_tokens -- -max_total_time=60
```

---

## License

[Apache-2.0](LICENSE)
