"""Settings, read from environment variables (or backend/.env) with sensible defaults."""
import os
from pathlib import Path

from dotenv import load_dotenv

BACKEND_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BACKEND_DIR / ".env")

DATA_DIR = BACKEND_DIR / "data"
RAW_DIR = DATA_DIR / "raw"  # downloaded HTML pages, so re-chunking never re-downloads
INDEX_DIR = DATA_DIR / "index"  # ChromaDB files

# Embedding model (runs locally through fastembed/ONNX, no PyTorch needed).
# bge-small reads up to 512 tokens and is noticeably better at search than all-MiniLM-L6-v2.
EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5")

# The LLM that writes the answer (Groq's free tier).
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")

TOP_K = int(os.getenv("TOP_K", "4"))  # chunks given to the LLM
# Guardrail: if the best chunk is less similar than this, refuse without calling the LLM.
# Tuned on the evaluation set (python -m eval.evaluate prints the best value).
MIN_SCORE = float(os.getenv("MIN_SCORE", "0.6"))

CORS_ORIGINS = [o.strip() for o in os.getenv("CORS_ORIGINS", "http://localhost:5173").split(",") if o.strip()]
