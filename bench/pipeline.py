"""The request path for both designs."""

from __future__ import annotations

import time

from bench.llm import LLMClient
from bench.models import Answer
from bench.retrieval.outline import OutlineRetriever
from bench.retrieval.vector import VectorRetriever
from bench.telemetry import Telemetry

MAX_ROUNDS = 4


class Pipeline:
    def __init__(
        self,
        llm: LLMClient,
        telemetry: Telemetry,
        vector: VectorRetriever | None = None,
        outline: OutlineRetriever | None = None,
        max_rounds: int = MAX_ROUNDS,
    ) -> None:
        self._llm = llm
        self._telemetry = telemetry
        self._vector = vector
        self._outline = outline
        self._max_rounds = max_rounds

    async def ask(self, design: str, question: str, doc_id: str) -> Answer:
        start = time.perf_counter()
        with self._telemetry.tracer.start_as_current_span("ask") as span:
            span.set_attribute("rag.design", design)
            if design == "vector":
                result, rounds = await self._ask_vector(question)
            elif design == "outline":
                result, rounds = await self._ask_outline(question, doc_id)
            else:
                raise ValueError(f"unknown design: {design}")
            span.set_attribute("rag.rounds", rounds)
        elapsed = time.perf_counter() - start
        self._telemetry.request_seconds.labels(design).observe(elapsed)
        self._telemetry.retrieval_rounds.labels(design).observe(rounds)
        return Answer(design, result.answer, result.supported, rounds, elapsed * 1000)

    async def _ask_vector(self, question: str):
        if self._vector is None:
            raise RuntimeError("the vector design is not available")
        rounds = 0
        result = None
        while rounds < self._max_rounds:
            rounds += 1
            with self._telemetry.step("vector", f"retrieval_round_{rounds}"):
                passages = await self._vector.retrieve(question, k=20 + rounds * 10)
                with self._telemetry.step("vector", "generate"):
                    result = await self._llm.generate(question, passages)
            if result.supported:
                break
        return result, rounds

    async def _ask_outline(self, question: str, doc_id: str):
        if self._outline is None:
            raise RuntimeError("the outline design is not available")
        passages = await self._outline.retrieve(question, doc_id)
        with self._telemetry.step("outline", "generate"):
            result = await self._llm.generate(question, passages)
        return result, 1
