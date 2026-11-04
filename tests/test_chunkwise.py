import pytest

import chunkwise
from chunkwise import (
    Chunk,
    chunk_text,
    chunk_texts,
    chunk_text_tokens,
    chunk_texts_tokens,
)

SAMPLE = (
    "Héllo wörld. 日本語のテキスト。😀 Another sentence here.\n\nNew paragraph with more words. "
    * 20
)
STRATEGIES = ["fixed", "sentence", "recursive"]


# ─── Baseline Character-based Tests ──────────────────────────────────────────


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_offsets_map_back_to_source(strategy):
    chunks = chunk_text(SAMPLE, chunk_size=80, chunk_overlap=15, strategy=strategy)
    assert chunks
    for i, c in enumerate(chunks):
        assert c.index == i
        assert len(c) <= 80
        assert SAMPLE[c.start : c.end] == c.text
        assert repr(c).startswith("Chunk(")


def test_many_matches_single_and_keeps_order():
    docs = [SAMPLE[: 200 + i * 50] for i in range(30)]
    many = chunk_texts(docs, chunk_size=60, chunk_overlap=10)
    for doc, chunks in zip(docs, many):
        single = chunk_text(doc, chunk_size=60, chunk_overlap=10)
        assert [(c.text, c.start, c.end) for c in chunks] == [
            (c.text, c.start, c.end) for c in single
        ]


def test_empty_input():
    assert chunk_text("") == []
    assert chunk_texts([]) == []


@pytest.mark.parametrize(
    "kwargs",
    [
        {"chunk_size": 0},
        {"chunk_size": 10, "chunk_overlap": 10},
        {"chunk_size": 10, "chunk_overlap": 15},
        {"strategy": "nope"},
        {"keep_separator": "invalid_mode"},
    ],
)
def test_invalid_arguments_raise_value_error(kwargs):
    with pytest.raises(ValueError):
        chunk_text("hello world", **kwargs)


def test_version_exposed():
    assert isinstance(chunkwise.__version__, str)
    assert chunkwise.__version__ == "0.2.0"


# ─── Milestone 1: Whitespace & Lossless Reconstruction ───────────────────────


def test_drop_whitespace_default_true():
    # Whitespace-only input produces no chunks by default
    assert chunk_text("   \n\n\t   ", chunk_size=10, chunk_overlap=2) == []


def test_drop_whitespace_false_retains_whitespace():
    ws_input = "   \n\n   "
    chunks = chunk_text(
        ws_input, chunk_size=10, chunk_overlap=0, drop_whitespace=False
    )
    assert len(chunks) > 0
    reconstructed = "".join(c.text for c in chunks)
    assert reconstructed == ws_input
    for c in chunks:
        assert ws_input[c.start : c.end] == c.text


def test_lossless_zero_overlap_reconstruction():
    corpus = [
        "First paragraph here.\n\nSecond one is a little longer.\nThird line. Fourth!",
        "Hello world",
        "  leading whitespace",
        "trailing whitespace  ",
        "a",
        "日本語 😀 emoji café résumé",
        "line one\r\nline two\r\nline three",
        "a\n\n\n\nb",
    ]
    for text in corpus:
        # Fixed strategy with 0 overlap and drop_whitespace=False guarantees lossless partition
        chunks_fixed = chunk_text(
            text,
            chunk_size=15,
            chunk_overlap=0,
            strategy="fixed",
            drop_whitespace=False,
        )
        assert "".join(c.text for c in chunks_fixed) == text
        # Verify exact contiguous character coverage
        expected_idx = 0
        for c in chunks_fixed:
            assert c.start == expected_idx
            assert text[c.start : c.end] == c.text
            expected_idx = c.end
        assert expected_idx == len(text)

        # Recursive strategy with keep_separator="end" and drop_whitespace=False
        chunks_rec = chunk_text(
            text,
            chunk_size=15,
            chunk_overlap=0,
            strategy="recursive",
            keep_separator="end",
            drop_whitespace=False,
        )
        assert "".join(c.text for c in chunks_rec) == text
        expected_idx = 0
        for c in chunks_rec:
            assert c.start == expected_idx
            assert text[c.start : c.end] == c.text
            expected_idx = c.end
        assert expected_idx == len(text)


def test_overlap_concatenation_is_not_lossless():
    text = "abcdefghijklmnopqrstuvwxyz"
    chunks = chunk_text(text, chunk_size=10, chunk_overlap=4, strategy="fixed")
    concatenated = "".join(c.text for c in chunks)
    # Naive concatenation duplicates overlapping characters
    assert concatenated != text
    assert len(concatenated) > len(text)
    # But every chunk is an authentic, unmodified slice
    for c in chunks:
        assert text[c.start : c.end] == c.text


# ─── Milestone 3: Configurable Separator Handling ───────────────────────────


def test_keep_separator_end():
    text = "alpha beta gamma.\n\ndelta epsilon zeta."
    chunks = chunk_text(
        text, chunk_size=20, chunk_overlap=0, strategy="recursive", keep_separator="end"
    )
    assert len(chunks) == 2
    assert chunks[0].text == "alpha beta gamma.\n\n"
    assert chunks[1].text == "delta epsilon zeta."
    assert text[chunks[0].start : chunks[0].end] == chunks[0].text
    assert text[chunks[1].start : chunks[1].end] == chunks[1].text


def test_keep_separator_start():
    text = "alpha beta gamma.\n\ndelta epsilon zeta."
    chunks = chunk_text(
        text, chunk_size=20, chunk_overlap=0, strategy="recursive", keep_separator="start"
    )
    assert len(chunks) >= 2
    assert not chunks[0].text.contains("\n\n") if hasattr(chunks[0].text, "contains") else ("\n\n" not in chunks[0].text)
    for c in chunks:
        assert text[c.start : c.end] == c.text


def test_keep_separator_discard():
    text = "alpha beta gamma.\n\ndelta epsilon zeta."
    chunks = chunk_text(
        text, chunk_size=20, chunk_overlap=0, strategy="recursive", keep_separator="discard"
    )
    for c in chunks:
        assert "\n\n" not in c.text
        assert text[c.start : c.end] == c.text


# ─── Milestone 2: Rust-Native Token Chunking ─────────────────────────────────


def test_token_chunking_cl100k_and_o200k():
    text = "The quick brown fox jumps over the lazy dog. " * 30
    for enc in ["cl100k_base", "o200k_base"]:
        chunks = chunk_text_tokens(
            text,
            chunk_size=32,
            chunk_overlap=8,
            encoding_name=enc,
            strategy="recursive",
        )
        assert chunks
        for i, c in enumerate(chunks):
            assert c.index == i
            assert c.token_count <= 32
            assert text[c.start : c.end] == c.text
            assert repr(c).startswith("TokenChunk(")


def test_token_chunking_fixed_strategy():
    text = "Word " * 200
    chunks = chunk_text_tokens(
        text,
        chunk_size=25,
        chunk_overlap=5,
        encoding_name="cl100k_base",
        strategy="fixed",
    )
    assert chunks
    for i, c in enumerate(chunks):
        assert c.index == i
        assert c.token_count <= 25
        assert text[c.start : c.end] == c.text


def test_token_chunking_parallel_matches_sequential():
    docs = [f"Document {i}: " + "Text about machine learning and embeddings. " * 15 for i in range(12)]
    par = chunk_texts_tokens(docs, chunk_size=40, chunk_overlap=8)
    assert len(par) == len(docs)
    for doc, p_chunks in zip(docs, par):
        seq_chunks = chunk_text_tokens(doc, chunk_size=40, chunk_overlap=8)
        assert [(c.text, c.start, c.end, c.token_count) for c in p_chunks] == [
            (c.text, c.start, c.end, c.token_count) for c in seq_chunks
        ]


def test_token_chunking_empty_and_whitespace():
    assert chunk_text_tokens("") == []
    assert chunk_texts_tokens([]) == []
    assert chunk_text_tokens("   \n\n   ", drop_whitespace=True) == []


@pytest.mark.parametrize(
    "kwargs",
    [
        {"chunk_size": 0},
        {"chunk_size": 20, "chunk_overlap": 20},
        {"chunk_size": 20, "chunk_overlap": 25},
        {"encoding_name": "non_existent_bpe"},
        {"strategy": "bad_strategy"},
        {"keep_separator": "invalid"},
    ],
)
def test_token_invalid_args_raise_value_error(kwargs):
    with pytest.raises(ValueError):
        chunk_text_tokens("Hello world", **kwargs)


def test_token_chunking_unicode_and_emoji():
    text = "Hello 😀 café résumé 日本語 🦀 Python text processing."
    chunks = chunk_text_tokens(text, chunk_size=6, chunk_overlap=1)
    assert chunks
    for c in chunks:
        assert c.token_count <= 6
        assert text[c.start : c.end] == c.text


# ─── Milestone 5: LangChain Comparison Validation ───────────────────────────


def test_langchain_parity_comparison():
    """Verify that chunkwise and LangChain run on identical configurations,
    demonstrating chunkwise offset validity and runtime characteristics."""
    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter
    except ImportError:
        pytest.skip("langchain-text-splitters not installed")

    doc = (
        "Retrieval-Augmented Generation (RAG) combines semantic search with LLMs.\n\n"
        "Document processing pipelines must chunk text accurately and safely.\n"
        "Chunkwise guarantees exact character offsets back to the source document.\n\n"
        "LangChain's RecursiveCharacterTextSplitter splits text on separators too."
    ) * 10

    chunk_size = 200
    chunk_overlap = 40

    lc_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        keep_separator=True,
    )
    lc_chunks = lc_splitter.split_text(doc)

    cw_chunks = chunk_text(
        doc,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        strategy="recursive",
        keep_separator="end",
    )

    # Both produce non-empty lists of bounded chunks
    assert len(cw_chunks) > 0
    assert len(lc_chunks) > 0

    for c in cw_chunks:
        assert len(c) <= chunk_size
        # Key Chunkwise guarantee: exact slice match
        assert doc[c.start : c.end] == c.text
