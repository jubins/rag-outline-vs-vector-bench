"""A simple lexical re-ranker. It orders hits by how many question words they contain."""

from __future__ import annotations

from dataclasses import replace

from bench.embedding import tokens
from bench.models import Passage


def rerank(question: str, hits: list[Passage], top_n: int = 8) -> list[Passage]:
    wanted = set(tokens(question))
    rescored = [
        replace(hit, score=len(wanted & set(tokens(hit.text))) / (len(wanted) or 1)) for hit in hits
    ]
    rescored.sort(key=lambda p: p.score, reverse=True)
    return rescored[:top_n]
