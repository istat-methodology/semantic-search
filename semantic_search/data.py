import numpy as np
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Document:
    r"""A data structure representing a single document.

    Args:
    id (Any): Unique identifier (any type).
    content (str): Textual content of the document.
    metadata (Optional[Dict[str, Any]]): Optional dictionary with extra info.
    """

    id: int
    content: str
    metadata: Optional[Dict[str, Any]] = field(default_factory=dict)

    def copy(self):
        r"""Returns a copy of the document."""
        return Document(
            id=self.id,
            content=self.content,
            metadata=self.metadata.copy(),
        )


@dataclass
class Corpus:
    r"""A collection of `Document` objects."""

    documents: List[Document]

    def __len__(self) -> int:
        return len(self.documents)

    def __getitem__(self, index: int) -> Document:
        return self.documents[index]


@dataclass
class RetrievedPoint:
    r"""A search result with associated similarity score"""

    id: Any
    metadata: Dict[str, Any]
    score: float


@dataclass
class SearchOutput:
    r"""A list of `RetrievedPoint` objects."""

    results: List[RetrievedPoint]

    def __len__(self) -> int:
        return len(self.results)

    def __getitem__(self, index: int) -> RetrievedPoint:
        return self.results[index]


def build_corpus(
    texts: List[str], ids: List[int], metadata: Optional[List[Dict[str, Any]]] = None
) -> Corpus:
    r"""Utility to build a `Corpus` from parallel lists.

    Args:
    texts (List[str]): List of document strings.
    ids (List[int]): List of unique integer document IDs.
    metadata (Optional[List[Dict[str, Any]]]): Optional list of metadata dictionaries. Defaults to None.
    """
    if metadata is None:
        metadata = [{}] * len(texts)

    assert len(np.unique(ids)) == len(ids), "IDs must be unique."
    assert all(isinstance(i, int) for i in ids), "All IDs must be integers."
    assert (len(texts) == len(ids) == len(metadata)), "texts, ids, and metadata must be the same length."

    docs = [Document(i, text, meta) for i, text, meta in zip(ids, texts, metadata)]
    return Corpus(docs)
