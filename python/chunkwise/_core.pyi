from typing import Literal, Sequence

Strategy = Literal["fixed", "sentence", "recursive"]
__version__: str

class Chunk:
    text: str
    start: int  # character offset, inclusive
    end: int  # character offset, exclusive
    index: int
    def __len__(self) -> int: ...

def chunk_text(
    text: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    strategy: Strategy = "recursive",
) -> list[Chunk]: ...
def chunk_texts(
    texts: Sequence[str],
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    strategy: Strategy = "recursive",
) -> list[list[Chunk]]: ...
