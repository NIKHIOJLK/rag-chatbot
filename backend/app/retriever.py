"""Step 2: embeddings, the vector index and hybrid search.

    python -m app.index   (see index.py) builds the index the API serves from.

Hybrid search = vector search (finds text with the same *meaning*) + BM25 keyword search
(finds exact terms like `enumerate` or `__init__`), merged with Reciprocal Rank Fusion.
"""
import re
import time
from dataclasses import dataclass
from typing import Protocol

import chromadb
from rank_bm25 import BM25Okapi


class Embedder(Protocol):
    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...
    def embed_query(self, text: str) -> list[float]: ...


class FastEmbedder:
    """Local embedding model via fastembed (ONNX runtime, no PyTorch). Downloads once, then cached."""

    def __init__(self, model_name: str):
        from fastembed import TextEmbedding  # imported here so tests can run without the model

        self.model = TextEmbedding(model_name)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [v.tolist() for v in self.model.passage_embed(texts)]

    def embed_query(self, text: str) -> list[float]:
        # bge models expect a short instruction in front of queries; query_embed adds it
        return next(iter(self.model.query_embed(text))).tolist()


@dataclass
class Hit:
    chunk: dict
    score: float  # cosine similarity from the vector search (0 if only BM25 found it)


def tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


def reciprocal_rank_fusion(rankings: list[list[str]], k: int = 60) -> list[str]:
    """Merge ranked id lists: each list gives an id 1/(k + rank). Robust because it ignores
    the raw scores, which aren't comparable between BM25 and cosine similarity."""
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, chunk_id in enumerate(ranking):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores, key=lambda cid: scores[cid], reverse=True)


class Retriever:
    def __init__(self, chunks: list[dict], embedder: Embedder, collection, hybrid: bool = True):
        self.chunks = {c["id"]: c for c in chunks}
        self.embedder = embedder
        self.collection = collection
        self.hybrid = hybrid
        self.ids = [c["id"] for c in chunks]
        self.bm25 = BM25Okapi([tokenize(c["text"]) for c in chunks]) if hybrid else None

    def search(self, query: str, k: int = 4, candidates: int = 20) -> tuple[list[Hit], float]:
        """Top-k chunks, plus the best cosine similarity (used by the guardrail)."""
        res = self.collection.query(
            query_embeddings=[self.embedder.embed_query(query)],
            n_results=min(candidates, len(self.ids)),
        )
        vector_ids = res["ids"][0]
        # the collection uses cosine distance: similarity = 1 - distance
        similarity = {cid: 1.0 - dist for cid, dist in zip(vector_ids, res["distances"][0])}
        best = max(similarity.values(), default=0.0)

        ranked = vector_ids
        if self.bm25 is not None:
            bm_scores = self.bm25.get_scores(tokenize(query))
            keyword_ids = [self.ids[i] for i in sorted(range(len(self.ids)), key=lambda i: bm_scores[i], reverse=True)[:candidates]
                           if bm_scores[i] > 0]
            ranked = reciprocal_rank_fusion([vector_ids, keyword_ids])

        hits = [Hit(self.chunks[cid], similarity.get(cid, 0.0)) for cid in ranked[:k]]
        return hits, best


def open_collection(path: str, name: str, reset: bool = False):
    client = chromadb.PersistentClient(path=path)
    if reset:
        try:
            client.delete_collection(name)
        except Exception:
            pass  # didn't exist yet
    return client.get_or_create_collection(name=name, metadata={"hnsw:space": "cosine"})


def build_collection(collection, chunks: list[dict], embedder: Embedder, batch: int = 64) -> float:
    """Embed every chunk and store it. Returns the seconds it took."""
    start = time.perf_counter()
    for i in range(0, len(chunks), batch):
        part = chunks[i : i + batch]
        collection.add(
            ids=[c["id"] for c in part],
            embeddings=embedder.embed_documents([c["text"] for c in part]),
            documents=[c["text"] for c in part],
            metadatas=[{"source": c["source"], "title": c["title"]} for c in part],
        )
    return time.perf_counter() - start
