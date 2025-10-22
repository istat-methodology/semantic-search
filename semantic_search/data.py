import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from collections import OrderedDict


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

    @classmethod
    def from_pandas(
        cls,
        df: pd.DataFrame,
        source_text_col: str,
        cols_to_include: Optional[List[str]] = [],
    ) -> "Corpus":
        r"""Create a `Corpus` from a pandas DataFrame.

        Args:
        df (pd.DataFrame): Source pandas dataframe.
        source_text_col (str): Name of the column containing the text to embed.
        cols_to_include (Optional[List[str]]): Optional list of columns to include in the metadata.
        """

        assert len(df) > 0 and not df.empty, "DataFrame is empty."

        cols = list(OrderedDict.fromkeys([source_text_col] + cols_to_include))
        assert all(
            col in df.columns for col in cols
        ), "Not all columns were found in DataFrame."

        doc_list: List[Document] = []
        for idx, row in df[cols].iterrows():
            source_text = row[source_text_col]
            meta = {col: row[col] for col in cols_to_include}

            doc_list.append(Document(id=idx, content=source_text, metadata=meta))

        return cls(doc_list)


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


@dataclass
class EvaluationEntry:
    r"""Evaluation result for a single query."""

    y_true: Any
    y_pred: List[Any]
    scores: List[float]
    is_correct: List[int]
    rank: int


@dataclass
class EvaluationOutput:
    r"""A list of `EvaluationEntry` objects."""

    results: List[EvaluationEntry]

    def __len__(self) -> int:
        return len(self.results)

    def __getitem__(self, index: int) -> EvaluationEntry:
        return self.results[index]

    def to_dicts(self) -> List[Dict]:
        return [vars(result) for result in self.results]


def build_corpus(
    texts: List[str],
    ids: Optional[List[int]] = None,
    metadata: Optional[List[Dict[str, Any]]] = None,
) -> Corpus:
    r"""Utility to build a `Corpus` from parallel lists.

    Args:
    texts (List[str]): List of document strings.
    ids (Optional[List[int]]): List of unique integer document IDs. Defaults to None (IDs automatically generated).
    metadata (Optional[List[Dict[str, Any]]]): Optional list of metadata dictionaries. Defaults to None.
    """
    if metadata is None:
        metadata = [{}] * len(texts)
    if ids is None:
        ids = list(range(len(texts)))

    assert len(np.unique(ids)) == len(ids), "IDs must be unique."
    assert all(isinstance(i, int) for i in ids), "All IDs must be integers."
    assert (
        len(texts) == len(ids) == len(metadata)
    ), "texts, ids, and metadata must be the same length."

    docs = [Document(i, text, meta) for i, text, meta in zip(ids, texts, metadata)]
    return Corpus(docs)
