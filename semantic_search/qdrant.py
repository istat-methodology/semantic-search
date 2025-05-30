import os
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct
from semantic_search.data import Corpus, RetrievedPoint, SearchOutput
from tqdm import tqdm
from dotenv import load_dotenv
from typing import List, Optional


load_dotenv()


class EmbeddingModel:
    r"""Unified abstraction for both HuggingFace and OpenAI embedding models.

    Args:
    model_id (str): Name of the embedding model (e.g., 'all-MiniLM-L6-v2', 'text-embedding-3-large').
    model_type (str): Either 'huggingface' or 'openai'.
    vector_size (Optional[int]): Required for OpenAI models. Optional for HuggingFace models (auto-inferred).
    openai_api_key (Optional[str]): OpenAI API key (default: from `.env` via `OPENAI_API_KEY`). Not needed for HuggingFace models.
    """

    def __init__(
        self,
        model_id,
        model_type: str,
        vector_size: Optional[int] = None,
        openai_api_key: Optional[str] = None,
    ):
        assert model_type in [
            "huggingface",
            "openai",
        ], "model_type must be 'huggingface' or 'openai'."

        self.model_id = model_id
        self.model_type = model_type
        self.vector_size = vector_size
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")

        if self.model_type == "huggingface":
            self.model = SentenceTransformer(self.model_id)
            model_embed_dim = self.model.get_sentence_embedding_dimension()
            assert (
                self.vector_size == model_embed_dim or self.vector_size is None
            ), f"The vector size of the model ({model_embed_dim}) does not match the provided vector size ({self.vector_size})."
            self.vector_size = model_embed_dim

        elif self.model_type == "openai":
            self.model = OpenAI(api_key=self.openai_api_key)
            self.model_type = "openai"
            assert (
                self.vector_size is not None
            ), "When using OpenAI embeddings, 'vector_size' must be specified."

    def encode(self, texts: List[str]):
        r"""Returns a list of embedding vectors for the given text inputs."""
        if self.model_type == "huggingface":
            embeddings = self.model.encode(texts, show_progress_bar=True)
        elif self.model_type == "openai":
            embeddings = []
            for text in tqdm(texts, desc="Creating embeddings..."):
                response = self.model.embeddings.create(
                    input=text, model=self.model_id, dimensions=self.vector_size
                )
                embeddings.append(response.data[0].embedding)
        return embeddings


class CollectionMananger:
    r"""Handles the creation and deletion of Qdrant collections.

    Args:
    qdrant_url (Optional[str]): URL to your Qdrant instance (default: from `.env` via `QDRANT_HOST`).
    qdrant_key (Optional[str]): API key for Qdrant (default: from `.env` via `QDRANT_API_KEY`).
    """

    def __init__(
        self, qdrant_url: Optional[str] = None, qdrant_key: Optional[str] = None
    ):
        self.qdrant_url = qdrant_url or os.getenv("QDRANT_HOST")
        self.qdrant_key = qdrant_key or os.getenv("QDRANT_API_KEY")
        self.client = QdrantClient(
            url=self.qdrant_url, api_key=self.qdrant_key, timeout=60
        )

    def create(
        self,
        name: str,
        corpus: Corpus,
        model_id: str,
        model_type: str,
        vector_size: Optional[int] = None,
        overwrite: Optional[bool] = False,
        upload_batch_size: Optional[int] = 32,
    ):
        r"""Creates a Qdrant collection and uploads embedded documents using using cosine similarity as a distance metric.

        Args:
        name (str): Collection name.
        corpus (Corpus): `Corpus` of documents to store.
        model_id (str): Name of the embedding model (e.g., 'all-MiniLM-L6-v2', 'text-embedding-3-large').
        model_type (str): Either 'huggingface' or 'openai'.
        vector_size (Optional[int]): Required for OpenAI models. Optional for HuggingFace models (auto-inferred).
        overwrite (Optional[bool]): If `True`, deletes existing collection with the same name before creation. Defaults to `False`.
        upload_batch_size (Optional[int]): Controls batch upload size to Qdrant. Defaults to 32.
        """
        if self.client.collection_exists(name):
            if overwrite:
                self.client.delete_collection(name)
            else:
                print(
                    f"A collection with name {name} already exists. If you want to overwrite it, set 'overwrite=True'."
                )
                return

        ids = [doc.id for doc in corpus.documents]
        texts = [doc.content for doc in corpus.documents]
        metadata = [doc.metadata for doc in corpus.documents]

        payloads = [{"text": t, "metadata": m} for t, m in zip(texts, metadata)]

        model = EmbeddingModel(
            model_id=model_id, model_type=model_type, vector_size=vector_size
        )
        vectors_config = VectorParams(size=model.vector_size, distance=Distance.COSINE)
        self.client.create_collection(
            collection_name=name, vectors_config=vectors_config
        )
        embeddings = model.encode(texts)

        points = []

        for i, embedding, payload in zip(ids, embeddings, payloads):
            point = PointStruct(id=i, vector=embedding, payload=payload)
            points.append(point)

        for i in tqdm(
            range(0, len(points), upload_batch_size),
            desc=f"Uploading batches to {name}",
        ):
            batch = points[i : i + upload_batch_size]
            self.client.upsert(collection_name=name, points=batch)

    def delete(self, name):
        r"""Deletes the specified collection, if it exists.

        Args:
        name (str): name of the collection to delete.
        """
        if self.client.collection_exists(name):
            self.client.delete_collection(name)
            print(f"Succesfully removed collection: {name}!")
        else:
            print(f"No collection named '{name}' exists to remove.")


class SemanticSeeker:
    r"""Performs semantic search on a remote Qdrant collection.

    Args:
    collection_name (str): Name of the collection to search.
    model_id (str): Name of the embedding model (e.g., 'all-MiniLM-L6-v2', 'text-embedding-3-large').
    model_type (str): Either 'huggingface' or 'openai'.
    vector_size (Optional[int]): Required for OpenAI models. Optional for HuggingFace models (auto-inferred).
    qdrant_url (Optional[str]): URL to your Qdrant instance (default: from `.env` via `QDRANT_HOST`).
    qdrant_key (Optional[str]): API key for Qdrant (default: from `.env` via `QDRANT_API_KEY`).
    """

    def __init__(
        self,
        collection_name: str,
        model_id: str,
        model_type: str,
        vector_size: Optional[int] = None,
        qdrant_url: Optional[str] = None,
        qdrant_key: Optional[str] = None,
    ):
        self.qdrant_url = qdrant_url or os.getenv("QDRANT_HOST")
        self.qdrant_key = qdrant_key or os.getenv("QDRANT_API_KEY")
        self.client = QdrantClient(
            url=self.qdrant_url, api_key=self.qdrant_key, timeout=60
        )
        self.name = collection_name
        self.model_id = model_id

        collection_vector_size = self.client.get_collection(
            collection_name
        ).config.params.vectors.size

        self.model = EmbeddingModel(
            model_id=model_id, model_type=model_type, vector_size=vector_size
        )

        assert (
            collection_vector_size == self.model.vector_size
        ), f"The vector sizes of collection ({collection_vector_size}) and model ({self.model.vector_size}) do not match!"

    def search(self, query: str, top_k: Optional[int] = 10) -> SearchOutput:
        r"""Returns the top `k` most semantically similar documents for a given query using cosine similarity.

        Args:
        query (str): The input textual query.
        top_k (Optional[int]): The number of most similar results to extracts. Defaults to 10.
        """
        query_vector = self.model.encode([query])[0]
        response = self.client.query_points(
            collection_name=self.name,
            query=query_vector,
            limit=top_k,
            with_vectors=False,
        )

        results: List[RetrievedPoint] = []
        for point in response.points:
            results.append(
                RetrievedPoint(
                    id=point.id,
                    text=point.payload["text"],
                    metadata=point.payload["metadata"],
                    score=point.score,
                )
            )

        return SearchOutput(results=results)
