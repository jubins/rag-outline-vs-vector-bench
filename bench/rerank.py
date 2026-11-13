"""A simple re-ranker that understands synonyms.

First-stage vector search matches words, so it can rank a section that uses different wording
far down the list. The re-ranker plays the role of a stronger model: it also counts synonyms of
the topic in the question. It can only reorder the candidates that search returned, so widening
the search is what brings the missing section into reach.
"""

from __future__ import annotations

from dataclasses import replace

from bench.corpus import TOPICS
from bench.embedding import tokens
from bench.models import Passage


def _expand(question: str) -> set[str]:
    words = set(tokens(question))
    for topic, synonym in TOPICS.items():
        if topic in question.lower():
            words |= set(tokens(synonym))
    return words


def rerank(question: str, hits: list[Passage], top_n: int = 8) -> list[Passage]:
    wanted = _expand(question)
    rescored = [
        replace(hit, score=len(wanted & set(tokens(hit.text))) / (len(wanted) or 1)) for hit in hits
    ]
    rescored.sort(key=lambda p: p.score, reverse=True)
    return rescored[:top_n]
