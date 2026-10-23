from app.rag import REFUSAL, RAGService, build_messages, search_query
from conftest import FakeLLM


def service(retriever, llm, min_score=0.2):
    return RAGService(retriever, llm, top_k=3, min_score=min_score)


def test_answer_returns_numbered_sources(retriever):
    out = service(retriever, FakeLLM()).answer("how do I append an item to a list?")
    assert out["refused"] is False
    assert [s["n"] for s in out["sources"]] == [1, 2, 3]
    assert all(s["url"].startswith("https://docs.python.org") for s in out["sources"])


def test_guardrail_refuses_without_calling_the_llm(retriever):
    llm = FakeLLM()
    out = service(retriever, llm, min_score=0.99).answer("what is a load balancer?")
    assert out == {"answer": REFUSAL, "sources": [], "refused": True}
    assert llm.calls == []  # no tokens spent


def test_llm_refusal_hides_sources(retriever):
    out = service(retriever, FakeLLM(REFUSAL)).answer("how do I append to a list?")
    assert out["refused"] is True and out["sources"] == []


def test_prompt_numbers_the_excerpts(retriever):
    hits, _ = retriever.search("stack", k=2)
    user = build_messages("q", hits, [])[-1]["content"]
    assert "[1] " in user and "[2] " in user


def test_follow_up_questions_search_with_the_previous_question():
    history = [{"role": "user", "content": "How do I sort a list?"}, {"role": "assistant", "content": "Use sorted()."}]
    assert search_query("and in reverse?", history) == "How do I sort a list? and in reverse?"
    assert search_query("first question", []) == "first question"


def test_stream_sends_sources_then_tokens_then_done(retriever):
    events = list(service(retriever, FakeLLM("one two")).stream("append to a list"))
    assert events[0]["type"] == "sources" and events[0]["sources"]
    assert [e["text"] for e in events if e["type"] == "token"] == ["one", "two"]
    assert events[-1] == {"type": "done"}
