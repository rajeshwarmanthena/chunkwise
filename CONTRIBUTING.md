# Contributing

```bash
cargo test                      # pure-Rust core tests
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"         # builds the extension via maturin
pytest
cargo fmt && cargo clippy --all-targets -- -D warnings
```

Rules for changes to the chunking engine: keep the invariants in `src/core.rs` (offsets map back to the source, chunks never exceed `chunk_size`, always terminates) and add a test for any new behaviour. Benchmark claims must come from `benchmarks/bench.py` with hardware noted.

## Fuzzing

```bash
rustup toolchain install nightly && cargo install cargo-fuzz
cargo +nightly fuzz run chunk -- -max_total_time=120
```

## Supply chain

`cargo install cargo-deny && cargo deny check` before opening a PR that changes dependencies.
