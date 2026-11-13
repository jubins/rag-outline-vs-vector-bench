import pytest

from bench.corpus import build_corpus
from bench.latency import LatencyModel
from bench.pipeline import Pipeline
from bench.retrieval.outline import InMemoryOutlineStore, OutlineRetriever
from bench.retrieval.vector import InMemoryVectorStore, VectorRetriever
from bench.telemetry import Telemetry
from llm_stub.logic import LocalLLM, SimulatedModel


@pytest.fixture(scope="session")
def corpus():
    return build_corpus(40)


@pytest.fixture
async def pipeline(corpus):
    telemetry = Telemetry()
    latency = LatencyModel(scale=0.0)
    llm = LocalLLM(SimulatedModel(corpus), latency)
    vector_store = InMemoryVectorStore()
    outline_store = InMemoryOutlineStore()
    await vector_store.index([s for sections in corpus.values() for s in sections])
    for doc_id, sections in corpus.items():
        await outline_store.put_document(doc_id, sections)
    return Pipeline(
        llm,
        telemetry,
        vector=VectorRetriever(vector_store, latency, telemetry),
        outline=OutlineRetriever(outline_store, llm, telemetry),
    )
