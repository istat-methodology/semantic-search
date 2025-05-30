<div align="center">
    <img src="resources/semanticsearch-logo.svg" alt="Semantic Search Logo">
</div>

<div align="center">
    <h3 align="center">
        Toolkit for building semantic search applications in Python.
    <h3>
</div>

## Installation
To install the `semantic_search` package, you can use pip:

```bash
pip install git+git://github.com/istat-methodology/semantic-search.git
```

## Usage
The `semantic_search` package provides two main functionalities: a **local** semantic search pipeline and a **qdrant-based** semantic search pipeline.

### 🔒 Local Semantic Search Pipeline
With `semantic_search.local`, you can build a semantic search pipeline that runs entirely in memory. The pipeline is optimized for cuda-compatible GPU usage and is particularly useful for small datasets or sensitive data that cannot be sent to a remote server.

#### Corpus
We start by creating a `Corpus` object, which is a collection of texts that we want to search. In other words, the texts in our knowledge base. To create the `Corpus` object, we can use the `build_corpus()` function from `semantic_search.data`. A valid `Corpus` object needs to have both texts and unique text IDs, and, optionally, a list of metadata dictionaries that can be retrieved and inspected after the search.

```python
from semantic_search.data import build_corpus

texts = ["This is a sample text", "Here is another example", "And this is a third one"]
ids = [0, 1, 2]
metadata = [{"author": "author_1"}, {"author": "author_1"}, {"author": "author_2"}]

corpus = build_corpus(
    texts=texts,
    ids=ids,
    metadata=metadata
)
```

#### Knowledge Base
We can now create a `LocalKnowledgeBase` object, which will store all the sentence-level embedding vectors associated with the texts in our corpus. The `model_id` argument specifies which `SentenceTransformer` model to use for generating the embeddings.

```python
from semantic_search.local import LocalKnowledgeBase

base = LocalKnowledgeBase(
    corpus=corpus,
    model_id="all-MiniLM-L6-v2"
)
```
We could also specify the `batch_size` argument to control how many texts are processed at once, which can be useful for large datasets, especially if you have a CUDA-compatible GPU. The default value is 32.

> To add new documents to an existing knowledge base, we can use the `.add()` method, passing the new documents as a `Corpus` object and specifying whether we want to overwrite existing ones with the same ID. To remove documents from an existing knowledge base, we can use the `.remove()` method, simply specifying the IDs of the documents we want to remove.

#### Semantic Search
Now we can perform semantic search using the `.search()` method of the `LocalKnowledgeBase` object. This method takes a query string (or list of strings) and returns the most relevant texts from the corpus based on their embeddings. The `top_k` argument specifies how many results to return.

For a single query, we can do:
```python
query = "What is a sample text?"

results = base.search(query, top_k=2)
```

For multiple queries, we can pass a list of strings and the method will handle them efficiently:
```python
queries = ["What is a sample text?", "What is the third text?"]

results = base.search(queries, top_k=2)
```

For both single and multiple queries, the results will be a list of `SearchOutput` objects, each containing `RetrievedPoint` objetcs, from which we can extract the text, id, score, and metadata of the retrieved points.

In practice, to access the top result for a single query, we can do:

```python
top_result = results[0][0]

score = top_result.score
text = top_result.text
idx = top_result.id
metadata = top_result.metadata
```

To access, for instance, the second result of the third query in the case of multiple queries, we can do:

```python
top_result = results[2][1]

score = top_result.score
text = top_result.text
idx = top_result.id
metadata = top_result.metadata
```

---

### 🌐 Qdrant-based Semantic Search Pipeline
The `semantic_search.qdrant` module provides a pipeline for building semantic search applications using the Qdrant vector database. This is particularly useful for larger datasets or when you need to scale your application.

To use the Qdrant-based pipeline, you need to have a Qdrant server running, and provide `QDRANT_HOST` and `QDRANT_API_KEY` either as environment variables or directly in the code (not recommended).

### Corpus
Similar to the local pipeline, we start by creating a `Corpus` object. The process is the same as described above. If we don't specify any metadata list, the build_corpus function will handle it creating a list of empty dictionaries.

```python
from semantic_search.data import build_corpus

texts = ["This is a sample text", "Here is another example", "And this is a third one"]
ids = [0, 1, 2]
corpus = build_corpus(
    texts=texts,
    ids=ids
)
```

#### Knowledge Base
Next, we create a knowledge base in the form of a Qdrant collection. This is done using the `CollectionManager` class, which handles the creation and management of collections in Qdrant. Each collection corresponds to a specific knowledge base.

```python
from semantic_search.qdrant import CollectionManager

manager = CollectionManager()

manager.create(
    collection_name="test-collection",
    corpus=corpus,
    model_id="text-embedding-3-large",
    model_type="openai",
    vector_size=3072
)
```

If you don't want to use OpenAI embedding models, you can use any other model supported by Hugging Face's `sentence-transformers` library, specifying the `model_id` accordingly and setting `model_type` to `"huggingface"`.

#### Semantic Search
Now we can initialize a `SemanticSeeker` object, which will handle the search queries against a specific collection in Qdrant. The `SemanticSeeker` needs to be initialized with the same embedding model used to create the collection.

```python
from semantic_search.qdrant import SemanticSeeker

seeker = SemanticSeeker(
    collection_name="test-collection",
    model_id="text-embedding-3-large",
    model_type="openai",
    vector_size=3072
)
```

We can finally perform semantic search using the `.search()` method of the `SemanticSeeker` object, passing a query string and specifying how many results to return with the `top_k` argument.

```python
query = "What is a sample text?"

results = seeker.search(query, top_k=2)
```

To extract the top result, we can do:

```python
top_result = results[0]

score = top_result.score
text = top_result.text
idx = top_result.id
metadata = top_result.metadata
```

---