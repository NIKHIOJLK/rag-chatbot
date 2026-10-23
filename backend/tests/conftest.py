"""Fakes so the tests run offline: no model download, no Groq calls."""
import hashlib
import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.retriever import Retriever, build_collection, open_collection, tokenize  # noqa: E402

SAMPLE_HTML = (Path(__file__).parent / "sample_page.html").read_text(encoding="utf-8")
SAMPLE_URL = "https://docs.python.org/3/tutorial/datastructures.html"


class FakeEmbedder:
    """Bag-of-words vectors: texts sharing words get similar vectors. Good enough to test ranking."""

    dims = 256

    def _vec(self, text):
        v = [0.0] * self.dims
        for word in tokenize(text):
            v[int(hashlib.md5(word.encode()).hexdigest(), 16) % self.dims] += 1.0
        norm = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / norm for x in v]

    def embed_documents(self, texts):
        return [self._vec(t) for t in texts]

    def embed_query(self, text):
        return self._vec(text)


class FakeLLM:
    def __init__(self, reply="Use `append()` to add an item [1]."):
        self.reply = reply
        self.calls = []

    def complete(self, messages):
        self.calls.append(messages)
        return self.reply

    def stream(self, messages):
        self.calls.append(messages)
        yield from self.reply.split(" ")


@pytest.fixture
def chunks():
    from app.ingest import chunk_by_sections

    return chunk_by_sections([(SAMPLE_URL, SAMPLE_HTML)])


@pytest.fixture
def retriever(chunks, tmp_path):
    collection = open_collection(str(tmp_path / "index"), "test", reset=True)
    build_collection(collection, chunks, FakeEmbedder())
    return Retriever(chunks, FakeEmbedder(), collection, hybrid=True)
