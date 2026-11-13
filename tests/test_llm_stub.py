from fastapi.testclient import TestClient

from bench.corpus import build_corpus, build_questions
from llm_stub.server import create_app


def test_select_and_generate_round_trip(monkeypatch):
    monkeypatch.setenv("LATENCY_SCALE", "0")
    corpus = build_corpus(40)
    question = build_questions(corpus, n=1, hard_ratio=0)[0]
    outline = [
        {"id": s.id, "title": s.title, "summary": s.summary} for s in corpus[question.doc_id]
    ]
    client = TestClient(create_app())
    picked = client.post("/select", json={"question": question.text, "outline": outline}).json()
    assert question.answer_section in picked["node_ids"]
    passages = [
        {"id": s.id, "text": s.text} for s in corpus[question.doc_id] if s.id in picked["node_ids"]
    ]
    result = client.post("/generate", json={"question": question.text, "passages": passages}).json()
    assert result["supported"] is True
