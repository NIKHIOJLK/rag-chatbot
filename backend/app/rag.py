"""Step 3: retrieve -> guardrail -> prompt -> LLM. The part the API calls for every question."""
from collections.abc import Iterator
from typing import Protocol

from .retriever import Hit, Retriever

REFUSAL = "I don't have enough information in the Python docs to answer that."

SYSTEM_PROMPT = """You answer questions about Python using ONLY the numbered documentation excerpts provided.
Rules:
- If the excerpts don't contain the answer, reply exactly: "I don't have enough information in the Python docs to answer that."
- Cite the excerpts you used with their numbers in square brackets, like [1] or [2][3].
- Be concise. Use Markdown, and put code in ```python blocks.
- Never invent functions, parameters or behaviour that the excerpts don't show."""


class LLM(Protocol):
    def complete(self, messages: list[dict]) -> str: ...
    def stream(self, messages: list[dict]) -> Iterator[str]: ...


class GroqLLM:
    def __init__(self, api_key: str, model: str):
        from groq import Groq

        if not api_key:
            raise SystemExit("GROQ_API_KEY is missing - add it to backend/.env")
        self.client = Groq(api_key=api_key)
        self.model = model

    def complete(self, messages: list[dict]) -> str:
        res = self.client.chat.completions.create(model=self.model, messages=messages, temperature=0.1)
        return res.choices[0].message.content

    def stream(self, messages: list[dict]) -> Iterator[str]:
        res = self.client.chat.completions.create(model=self.model, messages=messages, temperature=0.1, stream=True)
        for part in res:
            text = part.choices[0].delta.content
            if text:
                yield text


def search_query(question: str, history: list[dict]) -> str:
    """Follow-ups like "and for tuples?" mean nothing on their own, so search with the previous
    question added. Simple and cheap; an LLM query rewrite would be the next step."""
    previous = [m["content"] for m in history if m["role"] == "user"]
    return f"{previous[-1]} {question}" if previous else question


def build_messages(question: str, hits: list[Hit], history: list[dict]) -> list[dict]:
    context = "\n\n".join(f"[{n}] {h.chunk['text']}" for n, h in enumerate(hits, start=1))
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages += history[-6:]  # the last few turns, so "what about tuples?" makes sense
    messages.append({"role": "user", "content": f"Documentation excerpts:\n\n{context}\n\nQuestion: {question}"})
    return messages


def sources_of(hits: list[Hit]) -> list[dict]:
    return [{"n": n, "title": h.chunk["title"], "url": h.chunk["source"]} for n, h in enumerate(hits, start=1)]


class RAGService:
    def __init__(self, retriever: Retriever, llm: LLM, top_k: int, min_score: float):
        self.retriever = retriever
        self.llm = llm
        self.top_k = top_k
        self.min_score = min_score

    def retrieve(self, question: str, history: list[dict]) -> list[Hit] | None:
        """The chunks to answer from, or None when nothing is relevant enough (guardrail)."""
        hits, best = self.retriever.search(search_query(question, history), k=self.top_k)
        return hits if hits and best >= self.min_score else None

    def answer(self, question: str, history: list[dict] | None = None) -> dict:
        history = history or []
        hits = self.retrieve(question, history)
        if hits is None:  # refuse without spending an LLM call
            return {"answer": REFUSAL, "sources": [], "refused": True}
        text = self.llm.complete(build_messages(question, hits, history))
        refused = REFUSAL.lower()[:30] in text.lower()
        return {"answer": text, "sources": [] if refused else sources_of(hits), "refused": refused}

    def stream(self, question: str, history: list[dict] | None = None) -> Iterator[dict]:
        """Events for the streaming endpoint: sources first, then the answer piece by piece."""
        history = history or []
        hits = self.retrieve(question, history)
        if hits is None:
            yield {"type": "sources", "sources": [], "refused": True}
            yield {"type": "token", "text": REFUSAL}
        else:
            yield {"type": "sources", "sources": sources_of(hits), "refused": False}
            for piece in self.llm.stream(build_messages(question, hits, history)):
                yield {"type": "token", "text": piece}
        yield {"type": "done"}
