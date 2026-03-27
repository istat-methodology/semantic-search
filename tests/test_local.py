import importlib
import sys
import types
import unittest
from unittest.mock import patch


class FakeTensor:
    def __init__(self, values):
        self.values = values

    def __matmul__(self, other):
        return FakeSimMatrix()

    @property
    def T(self):
        return self


class FakeScore:
    def __init__(self, value):
        self.value = value

    def item(self):
        return self.value


class FakeIndex:
    def __init__(self, value):
        self.value = value

    def __index__(self):
        return self.value

    def __int__(self):
        return self.value


class FakeTopKResult:
    def __init__(self):
        self.sims = [[FakeScore(1.0)]]
        self.idxs = [[FakeIndex(0)]]

    def __iter__(self):
        return iter((self.sims, self.idxs))


class FakeSimMatrix:
    def topk(self, k, dim, largest, sorted):
        return FakeTopKResult()


class FakeSentenceTransformer:
    instances = []

    def __init__(self, model_id, device=None):
        self.model_id = model_id
        self.device = device
        self.calls = []
        type(self).instances.append(self)

    def get_sentence_embedding_dimension(self):
        return 3

    def encode(self, texts, batch_size, show_progress_bar, convert_to_tensor):
        self.calls.append(
            {
                "texts": list(texts),
                "batch_size": batch_size,
                "show_progress_bar": show_progress_bar,
                "convert_to_tensor": convert_to_tensor,
            }
        )
        return FakeTensor(texts)


def install_fake_dependencies():
    fake_numpy = types.ModuleType("numpy")
    fake_numpy.unique = lambda values: list(dict.fromkeys(values))

    fake_torch = types.ModuleType("torch")
    fake_torch.Tensor = FakeTensor
    fake_torch.device = lambda value: value
    fake_torch.cuda = types.SimpleNamespace(is_available=lambda: False)

    fake_torch_nn = types.ModuleType("torch.nn")
    fake_torch_nn_functional = types.ModuleType("torch.nn.functional")
    fake_torch_nn_functional.normalize = lambda tensor, dim=1: tensor

    fake_sentence_transformers = types.ModuleType("sentence_transformers")
    fake_sentence_transformers.SentenceTransformer = FakeSentenceTransformer

    return {
        "numpy": fake_numpy,
        "torch": fake_torch,
        "torch.nn": fake_torch_nn,
        "torch.nn.functional": fake_torch_nn_functional,
        "sentence_transformers": fake_sentence_transformers,
    }


class LocalKnowledgeBaseBatchSizeTest(unittest.TestCase):
    def setUp(self):
        FakeSentenceTransformer.instances.clear()

    def test_search_uses_explicit_batch_size_override(self):
        fake_modules = install_fake_dependencies()

        with patch.dict(sys.modules, fake_modules):
            sys.modules.pop("semantic_search.local", None)
            local = importlib.import_module("semantic_search.local")
            data = importlib.import_module("semantic_search.data")

            corpus = data.build_corpus(texts=["alpha", "beta"], ids=[1, 2])
            base = local.LocalKnowledgeBase(
                corpus=corpus,
                model_id="fake-model",
                batch_size=32,
            )

            base.search(["query"], batch_size=7)

            self.assertEqual(len(FakeSentenceTransformer.instances), 1)
            model = FakeSentenceTransformer.instances[0]
            self.assertEqual(model.calls[0]["batch_size"], 32)
            self.assertEqual(model.calls[1]["batch_size"], 7)


if __name__ == "__main__":
    unittest.main()
