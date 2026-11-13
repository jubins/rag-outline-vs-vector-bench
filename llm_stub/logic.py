"""Simulated model behaviour.

select_nodes picks outline entries whose title or summary share words with the question.
generate reports whether the passages it received contain the answer. A real system would ask a
model for both. Here the behaviour is fixed so that runs are repeatable.
"""

from __future__ import annotations

from bench.corpus import Section, locate_answer
from bench.embedding import tokens
from bench.models import Generation, OutlineNode, Passage

NOT_FOUND = "The provided text does not answer the question."


class SimulatedModel:
    def __init__(self, corpus: dict[str, list[Section]]) -> None:
        self._corpus = corpus

    def select_nodes(self, question: str, outline: list[OutlineNode], top_n: int = 2) -> list[str]:
        wanted = set(tokens(question))
        scored = []
        for node in outline:
            overlap = len(wanted & set(tokens(f"{node.title} {node.summary}")))
            if overlap:
                scored.append((overlap, node.id))
        scored.sort(key=lambda t: t[0], reverse=True)
        return [node_id for _, node_id in scored[:top_n]]

    def generate(self, question: str, passages: list[Passage]) -> Generation:
        target = locate_answer(self._corpus, question)
        for passage in passages:
            if passage.id == target:
                return Generation(passage.text, True)
        return Generation(NOT_FOUND, False)


class LocalLLM:
    """Runs SimulatedModel in-process. Used by tests and by the no-Docker mode."""

    def __init__(self, model: SimulatedModel, latency=None) -> None:
        self._model = model
        self._latency = latency

    async def select_nodes(self, question: str, outline: list[OutlineNode]) -> list[str]:
        if self._latency:
            await self._latency.pause("select")
        return self._model.select_nodes(question, outline)

    async def generate(self, question: str, passages: list[Passage]) -> Generation:
        if self._latency:
            await self._latency.pause("generate")
        return self._model.generate(question, passages)
