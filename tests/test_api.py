import pytest
from fastapi.testclient import TestClient

from bench.api import create_app
from bench.config import Settings
from bench.latency import LatencyModel


@pytest.fixture
def client():
    settings = Settings(backend="memory", design="outline", latency=LatencyModel(scale=0.0))
    with TestClient(create_app(settings)) as c:
        yield c


def test_health(client):
    body = client.get("/healthz").json()
    assert body == {"status": "ok", "design": "outline", "backend": "memory"}


def test_questions_endpoint_returns_the_requested_count(client):
    questions = client.get("/questions", params={"n": 25}).json()
    assert len(questions) == 25
    assert {"qid", "doc_id", "question", "hard"} <= set(questions[0])


@pytest.mark.parametrize("design", ["outline", "vector"])
def test_ask_returns_a_supported_answer_for_easy_questions(client, design):
    question = next(q for q in client.get("/questions", params={"n": 50, "hard_ratio": 0}).json())
    response = client.post(
        "/ask",
        json={"question": question["question"], "doc_id": question["doc_id"], "design": design},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["supported"] is True
    assert body["rounds"] == 1


def test_ask_rejects_unknown_design_and_document(client):
    bad_design = client.post(
        "/ask", json={"question": "x", "doc_id": "doc-0001", "design": "keyword"}
    )
    assert bad_design.status_code == 422
    bad_doc = client.post("/ask", json={"question": "x", "doc_id": "nope"})
    assert bad_doc.status_code == 404


def test_metrics_expose_latency_and_rounds(client):
    question = client.get("/questions", params={"n": 1}).json()[0]
    client.post("/ask", json={"question": question["question"], "doc_id": question["doc_id"]})
    text = client.get("/metrics").text
    assert "rag_request_seconds_bucket" in text
    assert "rag_retrieval_rounds_bucket" in text
