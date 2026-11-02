"""chunkwise: fast, deterministic, parallel text chunking for RAG pipelines."""

from ._core import Chunk, __version__, chunk_text, chunk_texts

__all__ = ["Chunk", "chunk_text", "chunk_texts", "__version__"]
