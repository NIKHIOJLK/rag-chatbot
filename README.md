# Python Docs RAG Chatbot

Ask questions about Python and get answers grounded in the official documentation, with numbered citations that link to the exact section. Off-topic questions are refused instead of answered from the model's imagination.

**Stack:** Python · FastAPI · fastembed (ONNX embeddings) · ChromaDB · BM25 · Groq (LLaMA 3.1) · React (Vite) · pytest · GitHub Actions

## How it works

```
Python docs (16 pages)
   │  app/ingest.py   download, split by headings, keep code blocks whole
   ▼
chunks (≤150 words, "Page > Section" header, link to #section)
   │  app/index.py    embed with bge-small, store in ChromaDB (cosine)
   ▼
question ──► hybrid search: vector (meaning) + BM25 (exact terms), merged with Reciprocal Rank Fusion
                │
                ├─ best similarity < MIN_SCORE ──► refuse (no LLM call)
                ▼
             top 4 chunks, numbered [1]..[4] ──► LLaMA 3.1 on Groq ──► answer with [n] citations
```

### What changed in v2, and why

| Problem in v1 | Fix in v2 |
| --- | --- |
| 500-word chunks, but all-MiniLM-L6-v2 only reads ~256 tokens, so most of each chunk was invisible to search | Chunks of at most 150 words, so the embedding sees all of it |
| Fixed windows cut through headings and code examples | Chunks follow the page's sections; code blocks stay whole; each chunk starts with "Page > Section" |
| 5 pages | The whole tutorial plus the built-in functions and exceptions reference |
| Pure vector search misses exact names (`enumerate`, `__init__`) | Hybrid search: vector + BM25 keyword search, merged with Reciprocal Rank Fusion |
| "Say you don't know" only in the prompt | A similarity threshold, tuned on labelled data, refuses before the LLM is called |
| No way to measure quality | 55 labelled questions + 12 off-topic ones; `eval/evaluate.py` reports hit@1, hit@3, MRR and latency |
| PyTorch install (>1 GB), UTF-16 requirements file | fastembed (ONNX) and a short UTF-8 `requirements.txt` |

## Evaluation

`python -m eval.evaluate` compares the original design with each improvement on the same questions. Results are written to [`backend/eval/results.md`](backend/eval/results.md).

## Run it

**Backend** (from `backend/`):
```bash
python -m venv venv
venv\Scripts\activate             # Windows (macOS/Linux: source venv/bin/activate)
pip install -r requirements-dev.txt
copy .env.example .env            # then put your Groq key in .env (macOS/Linux: cp)
python -m app.ingest              # 1. download and chunk the docs
python -m app.index               # 2. embed and store them
python -m eval.evaluate           # 3. optional: measure retrieval quality
uvicorn app.main:app --reload     # 4. API at http://127.0.0.1:8000/docs
```

**Frontend** (from `frontend/`):
```bash
npm install
npm run dev                       # http://localhost:5173
```

## API

| Method | Path | Purpose |
| --- | --- | --- |
| GET | `/api/health` | Status and number of indexed chunks |
| POST | `/api/ask` | `{question, history?}` → `{answer, sources: [{n, title, url}], refused}` |
| POST | `/api/ask/stream` | Same, as Server-Sent Events: sources first, then the answer token by token |

## Tests

```bash
cd backend && python -m pytest -q
```
The tests use a fake embedding model and a fake LLM, so they run offline in about a second: chunking, hybrid ranking, the guardrail (refuses without calling the LLM), citations, follow-up questions, the API and the evaluation metrics. CI runs them and the frontend build on every push.
