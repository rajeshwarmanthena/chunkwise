"""Benchmark chunkwise against LangChain's RecursiveCharacterTextSplitter.

    pip install langchain-text-splitters
    python benchmarks/bench.py --docs 2000 --doc-chars 20000

Both libraries use size=1000 / overlap=200. Their algorithms are similar but not
identical, so compare timings, not chunk-for-chunk output. Numbers depend on your
CPU; always run this on your own hardware before quoting results.
"""
import argparse
import os
import platform
import random
import statistics
import time

import chunkwise

WORDS = ("retrieval augmented generation vector index embedding latency enterprise "
         "document pipeline model token context window query semantic search").split()


def make_corpus(n_docs: int, doc_chars: int, seed: int = 0) -> list[str]:
    rng = random.Random(seed)
    docs = []
    for _ in range(n_docs):
        parts, size = [], 0
        while size < doc_chars:
            sent = " ".join(rng.choice(WORDS) for _ in range(rng.randint(6, 18))).capitalize() + "."
            if rng.random() < 0.12:
                sent += "\n\n"
            else:
                sent += " "
            parts.append(sent)
            size += len(sent)
        docs.append("".join(parts))
    return docs


def bench(fn, repeat: int):
    """One untimed warm-up, then `repeat` timed runs. Returns (median, min, max, last_output)."""
    out = fn()
    times = []
    for _ in range(repeat):
        t = time.perf_counter()
        out = fn()
        times.append(time.perf_counter() - t)
    return statistics.median(times), min(times), max(times), out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", type=int, default=2000)
    ap.add_argument("--doc-chars", type=int, default=20000)
    ap.add_argument("--size", type=int, default=1000)
    ap.add_argument("--overlap", type=int, default=200)
    ap.add_argument("--repeat", type=int, default=5)
    a = ap.parse_args()

    docs = make_corpus(a.docs, a.doc_chars)
    mb = sum(map(len, docs)) / 1e6
    print(f"python {platform.python_version()} on {platform.platform()}, cpus={os.cpu_count()}")
    print(f"corpus: {a.docs} docs, {mb:.1f} MB, size={a.size}, overlap={a.overlap}, repeat={a.repeat}")

    results = {}
    med, lo, hi, out = bench(lambda: [chunkwise.chunk_text(d, a.size, a.overlap) for d in docs], a.repeat)
    results["chunkwise (one doc at a time)"] = (med, lo, hi)
    n_chunks = sum(map(len, out))
    med, lo, hi, _ = bench(lambda: chunkwise.chunk_texts(docs, a.size, a.overlap), a.repeat)
    results["chunkwise (parallel batch)"] = (med, lo, hi)

    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        sp = RecursiveCharacterTextSplitter(chunk_size=a.size, chunk_overlap=a.overlap)
        results["langchain RecursiveCharacterTextSplitter"] = bench(
            lambda: [sp.split_text(d) for d in docs], a.repeat
        )[:3]
    except ImportError:
        print("(langchain-text-splitters not installed; skipping baseline)")

    base = results.get("langchain RecursiveCharacterTextSplitter")
    print(f"chunks produced by chunkwise: {n_chunks}")
    print(f"{'':45s} {'median':>8s} {'min':>8s} {'max':>8s} {'MB/s':>8s}")
    for name, (med, lo, hi) in results.items():
        extra = f"  ({base[0] / med:.1f}x vs langchain)" if base and "langchain" not in name else ""
        print(f"{name:45s} {med:7.2f}s {lo:7.2f}s {hi:7.2f}s {mb / med:8.1f}{extra}")


if __name__ == "__main__":
    main()
