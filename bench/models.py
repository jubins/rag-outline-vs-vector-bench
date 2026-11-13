"""Small data types shared by both retrieval designs."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Passage:
    id: str
    doc_id: str
    text: str
    score: float = 0.0


@dataclass(frozen=True)
class OutlineNode:
    id: str
    title: str
    summary: str


@dataclass(frozen=True)
class Generation:
    answer: str
    supported: bool


@dataclass(frozen=True)
class Answer:
    design: str
    answer: str
    supported: bool
    rounds: int
    latency_ms: float
