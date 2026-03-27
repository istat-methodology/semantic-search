import importlib
import sys
import types
import unittest
from unittest.mock import patch


class FakeEmbeddingModel:
    instances = []

    def __init__(self, model_id, model_type, vector_size=None, openai_api_key=None):
        self.model_id = model_id
        self.model_type = model_type
        self.vector_size = vector_size or 3
        self.openai_api_key = openai_api_key
        self.calls = []
        type(self).instances.append(self)

    def encode(self, texts, batch_size=32):
        self.calls.append({"texts": list(texts), "batch_size": batch_size})
        return [[0.1, 0.2, 0.3] for _ in texts]


class FakeQdrantClient:
    instances = []

    def __init__(self, url=None, api_key=None, timeout=None):
        self.url = url
        self.api_key = api_key
        self.timeout = timeout
        self.existing = set()
        self.deleted = []
        self.created = []
        self.upserts = []
        type(self).instances.append(self)

    def collection_exists(self, name):
        return name in self.existing

    def delete_collection(self, name):
        self.deleted.append(name)
        self.existing.discard(name)

    def create_collection(self, collection_name, vectors_config):
        self.created.append(
            {"collection_name": collection_name, "vectors_config": vectors_config}
        )
        self.existing.add(collection_name)

    def upsert(self, collection_name, points):
        self.upserts.append({"collection_name": collection_name, "points": points})


class FakeVectorParams:
    def __init__(self, size, distance):
        self.size = size
        self.distance = distance


class FakePointStruct:
    def __init__(self, id, vector, payload):
        self.id = id
        self.vector = vector
        self.payload = payload


def install_fake_dependencies():
    fake_numpy = types.ModuleType("numpy")
    fake_numpy.unique = lambda values: list(dict.fromkeys(values))

    fake_openai = types.ModuleType("openai")
    fake_openai.OpenAI = object

    fake_sentence_transformers = types.ModuleType("sentence_transformers")
    fake_sentence_transformers.SentenceTransformer = object

    fake_qdrant_client = types.ModuleType("qdrant_client")
    fake_qdrant_client.QdrantClient = FakeQdrantClient

    fake_qdrant_http_models = types.ModuleType("qdrant_client.http.models")
    fake_qdrant_http_models.Distance = types.SimpleNamespace(COSINE="cosine")
    fake_qdrant_http_models.VectorParams = FakeVectorParams
    fake_qdrant_http_models.PointStruct = FakePointStruct

    fake_tqdm = types.ModuleType("tqdm")
    fake_tqdm.tqdm = lambda iterable, **kwargs: iterable

    fake_dotenv = types.ModuleType("dotenv")
    fake_dotenv.load_dotenv = lambda: None

    return {
        "numpy": fake_numpy,
        "openai": fake_openai,
        "sentence_transformers": fake_sentence_transformers,
        "qdrant_client": fake_qdrant_client,
        "qdrant_client.http.models": fake_qdrant_http_models,
        "tqdm": fake_tqdm,
        "dotenv": fake_dotenv,
    }


class CollectionManagerCompatibilityTest(unittest.TestCase):
    def setUp(self):
        FakeEmbeddingModel.instances.clear()
        FakeQdrantClient.instances.clear()

    def _load_modules(self):
        fake_modules = install_fake_dependencies()
        with patch.dict(sys.modules, fake_modules):
            sys.modules.pop("semantic_search.qdrant", None)
            qdrant = importlib.import_module("semantic_search.qdrant")
            data = importlib.import_module("semantic_search.data")
            return qdrant, data

    def test_create_accepts_collection_name_keyword(self):
        qdrant, data = self._load_modules()
        qdrant.EmbeddingModel = FakeEmbeddingModel

        corpus = data.build_corpus(texts=["alpha", "beta"], ids=[1, 2])
        manager = qdrant.CollectionManager()

        manager.create(
            collection_name="docs",
            corpus=corpus,
            model_id="fake-model",
            model_type="huggingface",
            embed_batch_size=7,
            upload_batch_size=1,
        )

        client = FakeQdrantClient.instances[0]
        model = FakeEmbeddingModel.instances[0]
        self.assertEqual(model.calls[0]["batch_size"], 7)
        self.assertEqual(client.created[0]["collection_name"], "docs")
        self.assertEqual(len(client.upserts), 2)
        self.assertEqual(client.upserts[0]["collection_name"], "docs")
        self.assertEqual(client.upserts[0]["points"][0].payload["source_text"], "alpha")

    def test_create_accepts_legacy_name_keyword(self):
        qdrant, data = self._load_modules()
        qdrant.EmbeddingModel = FakeEmbeddingModel

        corpus = data.build_corpus(texts=["alpha"], ids=[1])
        manager = qdrant.CollectionManager()

        manager.create(
            name="legacy-docs",
            corpus=corpus,
            model_id="fake-model",
            model_type="huggingface",
        )

        client = FakeQdrantClient.instances[0]
        self.assertEqual(client.created[0]["collection_name"], "legacy-docs")

    def test_legacy_class_alias_points_to_collection_manager(self):
        qdrant, _ = self._load_modules()
        self.assertIs(qdrant.CollectionMananger, qdrant.CollectionManager)


if __name__ == "__main__":
    unittest.main()
