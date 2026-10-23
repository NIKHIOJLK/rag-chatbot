import json

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.rag import RAGService
from conftest import FakeLLM


@pytest.fixture
def client(retriever):
    app.state.rag = RAGService(retriever, FakeLLM(), top_k=3, min_score=0.2)
    with TestClient(app) as c:
        yield c
    app.state.rag = None


def test_health(client):
    assert client.get("/api/health").json()["status"] == "ok"


def test_ask(client):
    res = client.post("/api/ask", json={"question": "how do I append to a list?"})
    assert res.status_code == 200
    assert res.json()["sources"][0]["n"] == 1


def test_ask_validates_input(client):
    assert client.post("/api/ask", json={"question": ""}).status_code == 422
    assert client.post("/api/ask", json={"question": "   "}).status_code == 422
    assert client.post("/api/ask", json={"question": "x" * 501}).status_code == 422


def test_ask_reports_llm_failure_as_502(client, retriever):
    class Broken(FakeLLM):
        def complete(self, messages):
            raise TimeoutError

    app.state.rag = RAGService(retriever, Broken(), top_k=3, min_score=0.2)
    assert client.post("/api/ask", json={"question": "append to a list"}).status_code == 502


def test_stream_is_server_sent_events(client):
    res = client.post("/api/ask/stream", json={"question": "append to a list", "history": []})
    assert res.headers["content-type"].startswith("text/event-stream")
    events = [json.loads(line[6:]) for line in res.text.splitlines() if line.startswith("data: ")]
    assert events[0]["type"] == "sources" and events[-1]["type"] == "done"
