import json
from sentence_transformers import SentenceTransformer
import chromadb

def main():
    # Load the chunks we scraped earlier
    with open("chunks.json", "r", encoding="utf-8") as f:
        chunks = json.load(f)

    print(f"Loaded {len(chunks)} chunks")

    # Load a small, fast embedding model (downloads once, then cached locally)
    print("Loading embedding model...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # Set up a local ChromaDB database (saved to a folder called 'chroma_db')
    client = chromadb.PersistentClient(path="./chroma_db")
    collection = client.get_or_create_collection(name="python_docs")

    # Convert each chunk's text into a vector embedding, and store it
    texts = [chunk["text"] for chunk in chunks]
    sources = [chunk["source"] for chunk in chunks]
    ids = [f"chunk_{i}" for i in range(len(chunks))]

    print("Generating embeddings (this may take a minute)...")
    embeddings = model.encode(texts, show_progress_bar=True).tolist()

    print("Storing in ChromaDB...")
    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=texts,
        metadatas=[{"source": s} for s in sources]
    )

    print(f"\nDone. Stored {len(chunks)} chunks in ChromaDB (./chroma_db)")

if __name__ == "__main__":
    main()