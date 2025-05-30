from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class Document:
    r"""A data structure representing a single document.

    Args:
    id (Any): Unique identifier (any type).
    content (str): Textual content of the document.
    metadata (Optional[Dict[str, Any]]): Optional dictionary with extra info.
    """

    id: Any
    content: str
    metadata: Optional[Dict[str, Any]] = None


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
    text: str
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
    texts: List[str], ids: List[Any], metadata: Optional[List[Dict[str, Any]]] = None
) -> Corpus:
    r"""Utility to build a `Corpus` from parallel lists.

    Args:
    texts (List[str]): List of document strings.
    ids (List[Any]): List of unique document IDs.
    metadata (Optional[List[Dict[str, Any]]]): Optional list of metadata dictionaries. Defaults to None.
    """
    if metadata is None:
        metadata = [{}] * len(texts)
    assert (
        len(texts) == len(ids) == len(metadata)
    ), "texts, ids, and metadata must be the same length."
    docs = [Document(i, text, meta) for i, text, meta in zip(ids, texts, metadata)]
    return Corpus(docs)
