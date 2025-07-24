import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from semantic_search.data import Corpus, Document, RetrievedPoint, SearchOutput
from typing import List, Optional, Any


class LocalKnowledgeBase:
    r"""Builds a searchable semantic index from a `Corpus`.

    Args
    corpus (Corpus): The initial `Corpus` of documents.
    model_id (str): SentenceTransformer model ID (e.g., 'all-MiniLM-L6-v2').
    batch_size (Optional[int]): Embedding batch size. Defaults to 32.
    """

    def __init__(self, corpus: Corpus, model_id: str, batch_size: Optional[int] = 32, keep_source_text: Optional[bool] = True):
        self.corpus = corpus
        self.model_id = model_id
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.keep_source_text = keep_source_text

        self.model = SentenceTransformer(model_id, device=self.device)
        self.vector_size = self.model.get_sentence_embedding_dimension()

        self._init_storage(corpus)

    def _init_storage(self, corpus: Corpus) -> None:
        """
        Initialize or reset the corpus storage and embeddings.
        """
        self.corpus = corpus
        self.ids: List[int] = [doc.id for doc in corpus.documents]
        self.texts: List[str] = [doc.content for doc in corpus.documents]

        self.metadata = []
        for doc in corpus.documents:
            meta = dict(doc.metadata or {})
            if self.keep_source_text:
                meta["source_text"] = doc.content
            self.metadata.append(meta)

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
