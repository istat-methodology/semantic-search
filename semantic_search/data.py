import numpy as np
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import pandas as pd


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
    def from_dataframe(
        cls,
        df: pd.DataFrame,
        source_text_col: str,                               
        label_text_col: Optional[str] = None,
        label_id_col: Optional[str]= None,
        father_label_id_col : Optional[str] = None,
        g_father_label_id_col: Optional[str] = None,
        sep: str = '-'
           ) -> "Corpus":
        
        if df.empty: raise ValueError('DataFrame is empty')
        
        # 'source text' è il testo da embeddare, è necessario
        if source_text_col is None:
            raise ValueError("The parameter 'source_text_col' is required")
        if label_text_col is None and label_id_col is None:
            raise ValueError("The label requires either a code or a text or both")

        # controllo che le colonne pasate esistono nel df ('None' sono accettati con i vincoli di sopra)
        variables= [ source_text_col, label_text_col, label_id_col, father_label_id_col, g_father_label_id_col ]
        for v in variables:
            if v is not None:
                if v not in df.columns:
                    raise ValueError(f"{v} not present in dataframe columns")

        # del df prende solo colonne necessarie
        # testo da embeddare è pulito da eventuali spazi inizio o fine
        cols_needed = [ c for c in variables if c ]

        # testo da embeddare è pulito da eventuali spazi inizio o fine
        doc_list: List[Document] = []
        for idx, row in df[cols_needed].iterrows():
            content_from_df = str(row[source_text_col]).strip()
            metadata_from_df= {
            'label_id':             row.get(label_id_col),
            'label_text':           row.get(label_text_col),
            'father_label_id':      row.get(father_label_id_col),
            'g_father_label_id':    row.get(g_father_label_id_col)
                 }
            doc = Document( id= idx, content= content_from_df, metadata= metadata_from_df )
            doc_list.append(doc)

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
