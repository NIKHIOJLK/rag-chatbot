import os
from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import chromadb
from groq import Groq
from dotenv import load_dotenv

# Load the GROQ_API_KEY from your .env file
load_dotenv()

app = FastAPI()
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the same embedding model we used for ingestion
# (questions and documents must use the SAME model to compare correctly)
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# Connect to the ChromaDB we already built
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection(name="python_docs")

# Connect to Groq using the key from .env
groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

class Question(BaseModel):
    question: str

@app.post("/ask")
def ask(payload: Question):
    question = payload.question

    # 1. Embed the question the same way we embedded the chunks
    question_embedding = embed_model.encode([question]).tolist()

    # 2. Retrieve the top 3 most relevant chunks from ChromaDB
    results = collection.query(
        query_embeddings=question_embedding,
        n_results=3
    )

    retrieved_texts = results["documents"][0]
    retrieved_sources = [m["source"] for m in results["metadatas"][0]]

    # 3. Build a prompt that grounds the LLM in the retrieved context
    context = "\n\n---\n\n".join(retrieved_texts)
    prompt = f"""Answer the question using ONLY the context below. 
If the answer isn't in the context, say "I don't have enough information to answer that."

Context:
{context}

Question: {question}

Answer:"""

    # 4. Send to Groq for generation
    response = groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )
    answer = response.choices[0].message.content

    return {
        "answer": answer,
        "sources": list(set(retrieved_sources))
    }


@app.get("/")
def health_check():
    return {"status": "RAG chatbot API is running"}
