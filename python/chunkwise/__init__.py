"""chunkwise: fast, deterministic, parallel text chunking for RAG pipelines."""

from ._core import Chunk, __version__, chunk_text, chunk_texts

try:
    from ._core import TokenChunk, chunk_text_tokens, chunk_texts_tokens
    _has_tokens = True
except ImportError:
    _has_tokens = False

    def chunk_text_tokens(*args, **kwargs):
        raise NotImplementedError(
            "Token chunking is not available because chunkwise was compiled without the 'tiktoken-rs' feature."
        )

    def chunk_texts_tokens(*args, **kwargs):
        raise NotImplementedError(
            "Token chunking is not available because chunkwise was compiled without the 'tiktoken-rs' feature."
        )

    TokenChunk = None  # type: ignore

__all__ = [
    "Chunk",
    "chunk_text",
    "chunk_texts",
    "__version__",
    "TokenChunk",
    "chunk_text_tokens",
    "chunk_texts_tokens",
]
