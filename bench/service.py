"""Builds the pipeline for the chosen backend and loads the corpus into the stores."""

from __future__ import annotations

import asyncio
import logging

from bench.config import Settings
from bench.corpus import Section, build_corpus
from bench.llm import HttpLLM, LLMClient
from bench.pipeline import Pipeline
from bench.retrieval.outline import (
    AwsOutlineStore,
    InMemoryOutlineStore,
    OutlineRetriever,
    OutlineStore,
)
from bench.retrieval.vector import (
    InMemoryVectorStore,
    OpenSearchVectorStore,
    VectorRetriever,
    VectorStore,
)
from bench.telemetry import Telemetry
from llm_stub.logic import LocalLLM, SimulatedModel

log = logging.getLogger("bench")


async def wait_for(check, what: str, attempts: int = 60, delay: float = 2.0):
    for attempt in range(1, attempts + 1):
        try:
            return await asyncio.to_thread(check)
        except Exception as exc:  # the service may still be starting
            log.info("waiting for %s (%s/%s): %s", what, attempt, attempts, exc)
            await asyncio.sleep(delay)
    raise RuntimeError(f"{what} did not become ready")


class Service:
    """Holds the corpus, the stores and the pipeline. Designs are loaded on first use."""

    def __init__(self, settings: Settings, telemetry: Telemetry | None = None) -> None:
        self.settings = settings
        self.telemetry = telemetry or Telemetry()
        self.corpus = build_corpus(settings.n_docs)
        self._sections: list[Section] = [s for ss in self.corpus.values() for s in ss]
        self._llm: LLMClient = self._make_llm()
        self._vector: VectorRetriever | None = None
        self._outline: OutlineRetriever | None = None
        self._lock = asyncio.Lock()

    def _make_llm(self) -> LLMClient:
        if self.settings.backend == "memory":
            return LocalLLM(SimulatedModel(self.corpus), self.settings.latency)
        return HttpLLM(self.settings.llm_url)

    async def _load_vector(self) -> VectorRetriever:
        s = self.settings
        if s.backend == "memory":
            store: VectorStore = InMemoryVectorStore()
        else:
            store = OpenSearchVectorStore(s.opensearch_url)
            await wait_for(lambda: store._client.info(), "OpenSearch")
        if await store.count() < len(self._sections):
            await store.index(self._sections)
        return VectorRetriever(store, s.latency, self.telemetry)

    async def _load_outline(self) -> OutlineRetriever:
        s = self.settings
        if s.backend == "memory":
            store: OutlineStore = InMemoryOutlineStore()
        else:
            aws = AwsOutlineStore(s.aws_endpoint_url)
            await wait_for(aws.ensure_resources, "LocalStack")
            store = aws
        if await store.count() < len(self.corpus):
            for doc_id, sections in self.corpus.items():
                await store.put_document(doc_id, sections)
        return OutlineRetriever(store, self._llm, self.telemetry, s.cache_ttl_seconds)

    async def prepare(self, design: str) -> None:
        """Make sure the stores for a design exist and hold the corpus."""
        async with self._lock:
            if design == "vector" and self._vector is None:
                self._vector = await self._load_vector()
            elif design == "outline" and self._outline is None:
                self._outline = await self._load_outline()

    def pipeline(self) -> Pipeline:
        return Pipeline(self._llm, self.telemetry, vector=self._vector, outline=self._outline)
