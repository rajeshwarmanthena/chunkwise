"""Comprehensive benchmark for Chunkwise against baselines.

Measures:
- Wall-clock runtime distribution (median, min, max, stdev over repeated trials)
- Processing throughput (MB/s and docs/s)
- Peak RSS memory footprint (via resource.getrusage)
- Output chunk counts and token verification
- Multi-core scaling and batch performance
- Comparison with LangChain RecursiveCharacterTextSplitter when available

Usage:
    python benchmarks/bench.py --docs 500 --doc-chars 10000 --repeat 5
"""

import argparse
import os
import platform
import random
import statistics
import sys
import time

try:
    import resource
except ImportError:
    resource = None  # Windows support

import chunkwise

WORDS = (
    "retrieval augmented generation vector index embedding latency enterprise "
    "document pipeline model token context window query semantic search chunking "
    "transformer tokenizer attention similarity postgres milestone production"
).split()

MULTILINGUAL_SAMPLES = [
    "Héllo wörld, this is multilingual text. 日本語のテキスト。😀 ",
    "Une phrase en français avec des accents et des caractères spéciaux. ",
    "Ein längerer deutscher Text mit Umlauten wie ä, ö, ü und ß. ",
    "Русский текст для проверки корректности разбиения на токены. ",
]


def make_corpus(n_docs: int, doc_chars: int, seed: int = 0) -> list[str]:
    rng = random.Random(seed)
    docs = []
    for doc_idx in range(n_docs):
        parts, size = [], 0
        while size < doc_chars:
            # 10% chance of inserting a multilingual phrase
            if rng.random() < 0.10:
                sent = rng.choice(MULTILINGUAL_SAMPLES)
            else:
                sent = (
                    " ".join(rng.choice(WORDS) for _ in range(rng.randint(6, 16))).capitalize()
                    + "."
                )
            if rng.random() < 0.12:
                sent += "\n\n"
            else:
                sent += " "
            parts.append(sent)
            size += len(sent)
        docs.append("".join(parts))
    return docs


def get_peak_rss_mb() -> float:
    """Return peak resident set size in megabytes."""
    if resource is None:
        return 0.0
    rusage = resource.getrusage(resource.RUSAGE_SELF)
    # On macOS ru_maxrss is in bytes; on Linux it is in kilobytes.
    if sys.platform == "darwin":
        return rusage.ru_maxrss / (1024 * 1024)
    return rusage.ru_maxrss / 1024


def bench(fn, repeat: int):
    """Run an untimed warmup, then `repeat` timed runs.
    Returns (median, min, max, stdev, last_output, peak_rss_mb)."""
    # Warmup
    out = fn()
    times = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        out = fn()
        times.append(time.perf_counter() - t0)

    med = statistics.median(times)
    lo = min(times)
    hi = max(times)
    std = statistics.stdev(times) if len(times) > 1 else 0.0
    peak_rss = get_peak_rss_mb()
    return med, lo, hi, std, out, peak_rss


def main():
    ap = argparse.ArgumentParser(description="Chunkwise benchmark suite")
    ap.add_argument("--docs", type=int, default=500, help="Number of documents (default 500)")
    ap.add_argument("--doc-chars", type=int, default=10000, help="Characters per doc (default 10,000)")
    ap.add_argument("--size", type=int, default=1000, help="Chunk size (default 1000)")
    ap.add_argument("--overlap", type=int, default=200, help="Chunk overlap (default 200)")
    ap.add_argument("--token-size", type=int, default=256, help="Token chunk size (default 256)")
    ap.add_argument("--token-overlap", type=int, default=32, help="Token overlap (default 32)")
    ap.add_argument("--repeat", type=int, default=5, help="Number of timed repetitions (default 5)")
    args = ap.parse_args()

    docs = make_corpus(args.docs, args.doc_chars)
    total_bytes = sum(len(d.encode("utf-8")) for d in docs)
    mb = total_bytes / 1e6
    cpu_count = os.cpu_count() or 1

    print("=" * 80)
    print("CHUNKWISE ENTERPRISE BENCHMARK SUITE")
    print("=" * 80)
    print(f"Platform:      {platform.platform()} ({platform.machine()})")
    print(f"Python:        {platform.python_version()} ({platform.python_implementation()})")
    print(f"CPU count:     {cpu_count} logical cores")
    print(f"Corpus size:   {args.docs} docs, {mb:.2f} MB ({total_bytes:,} UTF-8 bytes)")
    print(f"Char config:   chunk_size={args.size}, chunk_overlap={args.overlap}")
    print(f"Token config:  chunk_size={args.token_size}, chunk_overlap={args.token_overlap}")
    print(f"Repetitions:   {args.repeat} trials (warmup discarded)")
    print("=" * 80)

    results = {}

    # 1. Single-threaded character chunking
    med, lo, hi, std, out, rss = bench(
        lambda: [chunkwise.chunk_text(d, args.size, args.overlap) for d in docs],
        args.repeat,
    )
    n_chunks = sum(map(len, out))
    results["chunkwise: char (single-thread)"] = (med, lo, hi, std, mb / med, rss, n_chunks)

    # 2. Parallel batch character chunking (Rayon)
    med, lo, hi, std, out, rss = bench(
        lambda: chunkwise.chunk_texts(docs, args.size, args.overlap),
        args.repeat,
    )
    n_chunks_par = sum(map(len, out))
    results["chunkwise: char (parallel Rayon)"] = (med, lo, hi, std, mb / med, rss, n_chunks_par)

    # 3. Token-aware single-threaded chunking (cl100k_base)
    if hasattr(chunkwise, "chunk_text_tokens"):
        med, lo, hi, std, out, rss = bench(
            lambda: [
                chunkwise.chunk_text_tokens(d, args.token_size, args.token_overlap, encoding_name="cl100k_base")
                for d in docs
            ],
            args.repeat,
        )
        n_tok_chunks = sum(map(len, out))
        results["chunkwise: token cl100k (single)"] = (med, lo, hi, std, mb / med, rss, n_tok_chunks)

        # 4. Token-aware parallel batch chunking
        med, lo, hi, std, out, rss = bench(
            lambda: chunkwise.chunk_texts_tokens(
                docs, args.token_size, args.token_overlap, encoding_name="cl100k_base"
            ),
            args.repeat,
        )
        n_tok_par = sum(map(len, out))
        results["chunkwise: token cl100k (parallel)"] = (med, lo, hi, std, mb / med, rss, n_tok_par)

    # 5. LangChain baseline comparison (if installed)
    base_med = None
    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        sp = RecursiveCharacterTextSplitter(
            chunk_size=args.size,
            chunk_overlap=args.overlap,
            keep_separator=True,
        )
        med, lo, hi, std, out, rss = bench(
            lambda: [sp.split_text(d) for d in docs],
            args.repeat,
        )
        lc_chunks = sum(map(len, out))
        results["langchain: RecursiveCharacterSplitter"] = (med, lo, hi, std, mb / med, rss, lc_chunks)
        base_med = med
    except ImportError:
        print("\n(Note: langchain-text-splitters not installed; skipping LangChain baseline)")

    print()
    header = f"{'Configuration':<38} {'Median':>8} {'Min':>8} {'Max':>8} {'StdDev':>8} {'MB/s':>9} {'Peak RSS':>10} {'Chunks':>8}"
    print(header)
    print("-" * len(header))
    for name, (med, lo, hi, std, thr, rss, chk) in results.items():
        speedup = f" ({base_med / med:4.1f}x)" if base_med and "langchain" not in name else ""
        print(
            f"{name:<38} {med:>7.3f}s {lo:>7.3f}s {hi:>7.3f}s {std:>7.4f}s {thr:>8.1f} {rss:>8.1f}MB {chk:>8d}{speedup}"
        )
    print("=" * 80)


if __name__ == "__main__":
    main()
