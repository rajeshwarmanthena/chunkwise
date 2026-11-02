# Benchmark methodology

Goal: make every performance claim reproducible and hard to game.

## What is measured
Wall-clock time to chunk a fixed synthetic corpus, from a Python call to the returned list of chunk objects. This includes the cost of creating Python objects, because that is what users pay.

## Setup
- Corpus: `benchmarks/bench.py` generates documents from a fixed seed (sentences of 6-18 words, about 12% paragraph breaks). Default: 1,000 documents x 20,000 characters.
- Settings: `chunk_size=1000`, `chunk_overlap=200` for every library.
- Baseline: LangChain `RecursiveCharacterTextSplitter` with the same size and overlap.
- Repetitions: each configuration runs `--repeat` times (default 5); the **median** is reported, plus min and max. One untimed warm-up run comes first.

## Validity rules
1. Always report CPU model, core count, OS and library versions with the numbers.
2. Algorithms are similar, not identical. Compare speed, not exact chunk boundaries.
3. The parallel API (`chunk_texts`) only shows gains when more than one core is available. Do not quote its speedup from a single-core machine.
4. Do not compare against a baseline that is not configured with the same size and overlap.

## Reproduce
```bash
pip install -e ".[dev]" langchain-text-splitters
python benchmarks/bench.py --docs 1000 --doc-chars 20000 --repeat 5
```

## Known limits
- Synthetic English-like text; real documents (PDF extraction, tables, code) behave differently.
- Sizes are in characters, not tokens.
- Results from a shared or virtualised machine can be noisy; use a quiet machine.
