import pytest

from bench.corpus import build_questions


async def test_outline_design_always_uses_one_round(pipeline, corpus):
    for q in build_questions(corpus, n=100, hard_ratio=0.5):
        answer = await pipeline.ask("outline", q.text, q.doc_id)
        assert answer.rounds == 1
        assert answer.supported


async def test_outline_design_searches_the_whole_collection(pipeline, corpus):
    # The doc_id in the request is ignored in collection scope, so a wrong one must not matter.
    for q in build_questions(corpus, n=50, hard_ratio=0.5):
        answer = await pipeline.ask("outline", q.text, "doc-0001")
        assert answer.supported


async def test_outline_document_scope_still_works(corpus):
    from bench.latency import LatencyModel
    from bench.pipeline import Pipeline
    from bench.retrieval.outline import InMemoryOutlineStore, OutlineRetriever
    from bench.telemetry import Telemetry
    from llm_stub.logic import LocalLLM, SimulatedModel

    telemetry = Telemetry()
    llm = LocalLLM(SimulatedModel(corpus), LatencyModel(scale=0.0))
    store = InMemoryOutlineStore()
    for doc_id, sections in corpus.items():
        await store.put_document(doc_id, sections)
    retriever = OutlineRetriever(store, llm, telemetry, scope="document")
    pipeline = Pipeline(llm, telemetry, outline=retriever)
    for q in build_questions(corpus, n=50, hard_ratio=0.5):
        assert (await pipeline.ask("outline", q.text, q.doc_id)).supported


async def test_vector_design_answers_easy_questions_in_one_round(pipeline, corpus):
    for q in build_questions(corpus, n=100, hard_ratio=0.0):
        answer = await pipeline.ask("vector", q.text, q.doc_id)
        assert answer.rounds == 1
        assert answer.supported


async def test_vector_design_retries_on_pointer_questions(pipeline, corpus):
    hard = [q for q in build_questions(corpus, n=300, hard_ratio=1.0)]
    rounds = [(await pipeline.ask("vector", q.text, q.doc_id)).rounds for q in hard]
    assert max(rounds) > 1
    assert sum(r > 1 for r in rounds) / len(rounds) > 0.3


async def test_unknown_design_is_rejected(pipeline):
    with pytest.raises(ValueError):
        await pipeline.ask("keyword", "What is the notice period?", "doc-0001")


async def test_metrics_are_recorded(pipeline, corpus):
    q = build_questions(corpus, n=1)[0]
    await pipeline.ask("outline", q.text, q.doc_id)
    text = "\n".join(
        line for line in _collect(pipeline) if line.startswith("rag_retrieval_rounds_count")
    )
    assert 'design="outline"' in text


def _collect(pipeline):
    from prometheus_client import generate_latest

    return generate_latest(pipeline._telemetry.registry).decode().splitlines()
