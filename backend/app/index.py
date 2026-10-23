"""Build the search index the API serves from.

    python -m app.ingest   # 1. download + chunk
    python -m app.index    # 2. embed + store (safe to run again: it rebuilds from scratch)
"""
import json

from .config import DATA_DIR, EMBED_MODEL, INDEX_DIR
from .retriever import FastEmbedder, Retriever, build_collection, open_collection

COLLECTION = "python_docs"


def load_chunks() -> list[dict]:
    path = DATA_DIR / "chunks.json"
    if not path.exists():
        raise SystemExit("data/chunks.json not found - run `python -m app.ingest` first.")
    return json.loads(path.read_text(encoding="utf-8"))


def load_retriever() -> Retriever:
    """Used by the API at startup."""
    chunks = load_chunks()
    collection = open_collection(str(INDEX_DIR), COLLECTION)
    if collection.count() != len(chunks):
        raise SystemExit("The index is missing or out of date - run `python -m app.index`.")
    return Retriever(chunks, FastEmbedder(EMBED_MODEL), collection, hybrid=True)


def main() -> None:
    chunks = load_chunks()
    print(f"Embedding {len(chunks)} chunks with {EMBED_MODEL} (downloads the model the first time)...")
    collection = open_collection(str(INDEX_DIR), COLLECTION, reset=True)
    seconds = build_collection(collection, chunks, FastEmbedder(EMBED_MODEL))
    print(f"Done in {seconds:.1f}s. Stored {collection.count()} chunks in data/index/")


if __name__ == "__main__":
    main()
