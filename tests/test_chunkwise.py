import pytest

import chunkwise
from chunkwise import chunk_text, chunk_texts

SAMPLE = "Héllo wörld. 日本語のテキスト。😀 Another sentence here.\n\nNew paragraph with more words. " * 20
STRATEGIES = ["fixed", "sentence", "recursive"]


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_offsets_map_back_to_source(strategy):
    chunks = chunk_text(SAMPLE, chunk_size=80, chunk_overlap=15, strategy=strategy)
    assert chunks
    for i, c in enumerate(chunks):
        assert c.index == i
        assert len(c) <= 80
        assert SAMPLE[c.start : c.end] == c.text


def test_many_matches_single_and_keeps_order():
    docs = [SAMPLE[: 200 + i * 50] for i in range(30)]
    many = chunk_texts(docs, chunk_size=60, chunk_overlap=10)
    for doc, chunks in zip(docs, many):
        single = chunk_text(doc, chunk_size=60, chunk_overlap=10)
        assert [(c.text, c.start, c.end) for c in chunks] == [(c.text, c.start, c.end) for c in single]


def test_empty_input():
    assert chunk_text("") == []
    assert chunk_texts([]) == []


@pytest.mark.parametrize(
    "kwargs",
    [
        {"chunk_size": 0},
        {"chunk_size": 10, "chunk_overlap": 10},
        {"strategy": "nope"},
    ],
)
def test_invalid_arguments_raise_value_error(kwargs):
    with pytest.raises(ValueError):
        chunk_text("hello world", **kwargs)


def test_version_exposed():
    assert isinstance(chunkwise.__version__, str)
