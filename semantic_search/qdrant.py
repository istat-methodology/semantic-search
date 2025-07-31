import os
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct, SearchParams
from qdrant_client.http import models
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
    vector_size (Optional[int]): Required for OpenAI models (leave 'None' for text-embedding-ada-002). Optional for HuggingFace models (auto-inferred).
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
            ), f"The dimension of the model ({model_embed_dim}) does not match the provided 'vector_size' ({self.vector_size})."
            self.vector_size = model_embed_dim

        elif self.model_type == "openai":
            self.model = OpenAI(api_key=self.openai_api_key)
            self._assert_dimensions_openai(self.model_id, self.vector_size)

    def encode(self, texts: List[str], batch_size: Optional[int] = 32) -> List[List[float]]:
        r"""Returns a list of embedding vectors for the given text inputs.
        
        Args:
        texts (List[str]): List of text inputs to encode.
        batch_size (Optional[int]): Controls batch size for embedding generation. Defaults to 32.
        """
        if self.model_type == "huggingface":
            embeddings = self.model.encode(texts, batch_size=batch_size, show_progress_bar=True)
        
        elif self.model_type == "openai":
            if self.model_id == "text-embedding-ada-002":
                embeddings = self._openai_batch_encode(texts, dimensions=None, batch_size=batch_size)
            else:
                embeddings = self._openai_batch_encode(texts, dimensions=self.vector_size, batch_size=batch_size)

        return embeddings

    def _openai_batch_encode(self, texts: List[str], dimensions: int, batch_size: int) -> List[List[str]]:
        embeddings = []
        for i in tqdm(range(0, len(texts), batch_size), desc="Creating embeddings..."):
            batch = texts[i : i + batch_size]
            response = self.model.embeddings.create(
                    input=batch,
                    model=self.model_id,
                    dimensions=dimensions
                )
            embeddings.extend([point.embedding for point in response.data])
        return embeddings
    
    def _assert_dimensions_openai(self, model_id: str, vector_size: int):
        if model_id == "text-embedding-ada-002":
            if vector_size not in (None, 1536):
                print("Warning: 'text-embedding-ada-002' model has a fixed dimension (1536), 'vector_size' will be set to 1536 when creating the database.")
            self.vector_size = 1536

        else:
            if model_id == "text-embedding-3-small":
                assert (type(vector_size) == int) and (vector_size % 8 == 0) and (8 <= vector_size <= 1536), "'vector_size' must be a multiple of 8 and in the range [8, 1536] for text-embedding-3-small."
            
            elif model_id == "text-embedding-3-large":
                assert (type(vector_size) == int) and (vector_size % 8 == 0) and (8 <= vector_size <= 3072), "'vector_size' must be a multiple of 8 and in the range [8, 3072] for text-embedding-3-large."
            
            else:
                raise ValueError(f"Unsupported model_id '{model_id}' for OpenAI embeddings.")



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
        upload_source_text: Optional[bool] = True,
        embed_batch_size: Optional[int] = 32,
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
        upload_source_text (Optional[bool]): If `True`, adds the original text to the document metadata. Defaults to `True`.
        embed_batch_size (Optional[int]): Controls batch size for embedding generation. Defaults to 256.
        upload_batch_size (Optional[int]): Controls batch upload size to Qdrant. Defaults to 32.
        """
        if self.client.collection_exists(name):
            if overwrite:
                self.client.delete_collection(name)
            else:
                print(f"Collection '{name}' exists. To overwrite, set 'overwrite=True'.")
                return

        texts = [doc.content for doc in corpus.documents]
        embedding_model = EmbeddingModel(
            model_id=model_id, model_type=model_type, vector_size=vector_size
        )
        embeddings = embedding_model.encode(texts, batch_size=embed_batch_size)

        points = []
        for doc, embedding in zip(corpus.documents, embeddings):
            if upload_source_text:
                doc = doc.copy()
                doc.metadata["source_text"] = doc.content

            point = PointStruct(
                id=doc.id,
                vector=embedding,
                payload=doc.metadata
            )
            points.append(point)
        
        vectors_config = VectorParams(
            size=embedding_model.vector_size,
            distance=Distance.COSINE
        )
        self.client.create_collection(collection_name=name, vectors_config=vectors_config)

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
    
    def get_collection_info(self, name: str) -> dict:
        info = self.client.get_collection(collection_name=name)
        vp = info.config.params.vectors
        return {
            "name": name,
            "status": info.status.value,
            "vector_size": vp.size,
            "distance": vp.distance.value,
            "points_count": info.points_count,
        }
    
    def list_collections(self)-> list[str]:
        """Returns a list of collection names."""
        return[ c.name for c in self.client.get_collections().collections ]
    
    def list_collections_with_info(self) -> list[dict]:
        """Returns a list of dicts, one per collection, containing status, vector size, distance, and points count."""
        info = []
        for name in self.list_collections():
            info.append(self.get_collection_info(name))
        return info
 


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
                    metadata=point.payload,
                    score=point.score,
                )
            )

        return SearchOutput(results=results)
    
    def search_many(self, queries: list[str], top_k: int = 5, batch_size: int = 512) -> list[SearchOutput]:
        
        vectors = self.model.encode(queries)
        if not isinstance(vectors, list):
            vectors = vectors.tolist()

        all_responses = []                       
        for i in range(0, len(queries), batch_size):
            vec_batch = vectors[i : i + batch_size]

            # requests parameters
            # per maggiore velocità di ricerca 
            # params = SearchParams(hnsw_ef=32)
            # with less Payload no time-gain
            reqs = [ models.QueryRequest(query=v, limit=top_k, with_vector=False, with_payload=True ) for v in vec_batch ]

            resp_batch = self.client.query_batch_points(
                collection_name=self.name,
                requests=reqs,
            )
            all_responses.extend(resp_batch)   

        # conversione in SearchOutput 
        outputs: list[SearchOutput] = []
        for res in all_responses:               
            retrieved = [
                RetrievedPoint(
                    id=p.id,
                    metadata=p.payload,
                    score=p.score,
                )
                for p in res.points
            ]
            outputs.append(SearchOutput(results=retrieved))

        return outputs    
