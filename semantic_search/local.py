import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from semantic_search.data import Corpus, RetrievedPoint, SearchOutput
from typing import List, Optional, Any, Dict


class LocalKnowledgeBase:
    r"""Builds a searchable semantic index from a `Corpus`.

    Args
    corpus (Corpus): The initial `Corpus` of documents.
    model_id (str): SentenceTransformer model ID (e.g., 'all-MiniLM-L6-v2').
    batch_size (Optional[int]): Embedding batch size. Defaults to 32.
    """

    def __init__(self, corpus: Corpus, model_id: str, batch_size: Optional[int] = 32):
        self.corpus = corpus
        self.texts = [doc.content for doc in corpus.documents]
        self.ids = [doc.id for doc in corpus.documents]
        self.metadata = [doc.metadata for doc in corpus.documents]

        self.model_id = model_id
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size

        self.model = SentenceTransformer(model_id, device=self.device)
        self.vector_size = self.model.get_sentence_embedding_dimension()

        self._init_storage(corpus)

    def _init_storage(self, corpus: Corpus) -> None:
        """
        Initialize or reset the corpus storage and embeddings.
        """
        self.corpus = corpus
        self.ids: List[Any] = [doc.id for doc in corpus.documents]
        self.texts: List[str] = [doc.content for doc in corpus.documents]
        self.metadata: List[Dict[str, Any]] = [
            doc.metadata or {} for doc in corpus.documents
        ]

        self.embeddings = self.model.encode(
            self.texts,
            batch_size=self.batch_size,
            show_progress_bar=True,
            convert_to_tensor=True,
        )

    @staticmethod
    def _pairwise_cosine_similarity(
        X_embeddings: torch.Tensor, Y_embeddings: torch.Tensor
    ) -> torch.Tensor:
        A = F.normalize(X_embeddings, dim=1)
        B = F.normalize(Y_embeddings, dim=1)
        return A @ B.T

    def _parse_output(
        self,
        idx: torch.Tensor,
        sim: torch.Tensor,
    ) -> SearchOutput:
        points = []
        for idx_i, sim_i in zip(idx, sim):
            points.append(
                RetrievedPoint(
                    id=self.ids[idx_i],
                    text=self.texts[idx_i],
                    metadata=self.metadata[idx_i],
                    score=sim_i.item(),
                )
            )
        return SearchOutput(points)

    def search(
        self,
        queries: List[str],
        top_k: Optional[int] = 10,
        batch_size: Optional[int] = None,
    ) -> List[SearchOutput]:
        r"""Performs semantic search for each input query.

        Args:
        queries (List[str]): List of query strings.
        top_k (Optional[int]): Number of top results per query (defaults to 10).
        batch_size (Optional[int]): Optional override of batch size. If not specified, it will use the `LocalKnowledgeBase` batch size.
        """
        top_k = min(top_k, len(self.ids))
        batch_size = batch_size or self.batch_size
        queries = queries if isinstance(queries, list) else [queries]

        query_embeddings = self.model.encode(
            queries,
            batch_size=self.batch_size,
            show_progress_bar=True,
            convert_to_tensor=True,
        )
        sim_matrix = self._pairwise_cosine_similarity(query_embeddings, self.embeddings)
        sims, idxs = sim_matrix.topk(k=top_k, dim=1, largest=True, sorted=True)

        return [self._parse_output(i, s) for i, s in zip(idxs, sims)]

    def add(
        self,
        new_corpus: Corpus,
        overwrite: Optional[bool] = False,
    ) -> None:
        r"""Add documents from new_corpus to the knowledge base.

        Args:
        new_corpus (Corpus): Corpus of new documents.
        overwrite (Optional[bool]): if `True`, replaces existing docs with matching IDs. Embeddings are computed only after filtering. Defaults to `False`.
        """
        id_to_index = {doc_id: idx for idx, doc_id in enumerate(self.ids)}

        to_index = []
        for idx, doc in enumerate(new_corpus.documents):
            if doc.id in id_to_index:
                if overwrite:
                    to_index.append(idx)
            else:
                to_index.append(idx)

        if not to_index:
            return

        if overwrite:
            remove_ids = [
                new_corpus.documents[i].id
                for i in to_index
                if new_corpus.documents[i].id in id_to_index
            ]
            self.remove(remove_ids)

        new_texts = [new_corpus.documents[i].content for i in to_index]
        new_ids = [new_corpus.documents[i].id for i in to_index]
        new_metadata = [new_corpus.documents[i].metadata or {} for i in to_index]

        new_embeddings = self.model.encode(
            new_texts,
            batch_size=self.batch_size,
            show_progress_bar=True,
            convert_to_tensor=True,
        )

        self.ids.extend(new_ids)
        self.texts.extend(new_texts)
        self.metadata.extend(new_metadata)
        self.embeddings = torch.cat([self.embeddings, new_embeddings], dim=0)

    def remove(self, remove_ids: List[Any]) -> None:
        r"""Remove documents and embeddings by ID.

        Args
        ids (List[Any]): List of document IDs to delete.
        """
        if not remove_ids:
            return

        keep_mask = [doc_id not in remove_ids for doc_id in self.ids]

        self.ids = [doc_id for doc_id, keep in zip(self.ids, keep_mask) if keep]
        self.texts = [text for text, keep in zip(self.texts, keep_mask) if keep]
        self.metadata = [meta for meta, keep in zip(self.metadata, keep_mask) if keep]

        mask_tensor = torch.tensor(
            keep_mask, dtype=torch.bool, device=self.embeddings.device
        )
        self.embeddings = self.embeddings[mask_tensor]

        new_docs = [
            RetrievedPoint(id_, text, meta, 0.0)
            for id_, text, meta in zip(self.ids, self.texts, self.metadata)
        ]
        self.corpus = Corpus(new_docs)
