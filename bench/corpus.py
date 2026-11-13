"""Synthetic contract-style corpus and question set.

Every document has eight sections. Four of them state a fact directly. One fact is stated with
different wording than the question uses, and a separate section only points to it
("see the terms in Section 6"). That pointer pattern is what makes similarity search miss the
answer on the first try, so it is the case that triggers retries in the vector design.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

TOPICS = {
    "notice period": "advance warning",
    "renewal term": "extension length",
    "payment schedule": "invoice calendar",
    "liability cap": "damages ceiling",
    "governing law": "applicable jurisdiction",
    "audit rights": "inspection permissions",
    "data retention": "record keeping duration",
    "service credits": "downtime refunds",
    "termination fee": "exit charge",
    "confidentiality term": "secrecy duration",
}

SECTIONS_PER_DOC = 8
POINTER_SECTION = 7
INDIRECT_SECTION = 6


@dataclass(frozen=True)
class Section:
    id: str
    doc_id: str
    number: int
    title: str
    summary: str
    text: str


@dataclass(frozen=True)
class Question:
    qid: str
    doc_id: str
    text: str
    answer_section: str
    hard: bool


def doc_tag(doc_id: str) -> str:
    """A unique token that appears in every section of a document, such as AG0007."""
    return "AG" + doc_id.split("-")[-1].zfill(4)


def build_corpus(n_docs: int = 40, seed: int = 7) -> dict[str, list[Section]]:
    rng = random.Random(seed)
    topics = list(TOPICS)
    corpus: dict[str, list[Section]] = {}
    for d in range(1, n_docs + 1):
        doc_id = f"doc-{d:04d}"
        tag = doc_tag(doc_id)
        chosen = rng.sample(topics, 5)
        units = ["days", "months", "years"]
        values = {t: f"{rng.randint(2, 90)} {rng.choice(units)}" for t in chosen}
        sections = [
            Section(
                f"{doc_id}-s01",
                doc_id,
                1,
                "Overview",
                "Parties and scope.",
                f"Agreement {tag} sets out the terms between the parties.",
            )
        ]
        for i, topic in enumerate(chosen[:4], start=2):
            sections.append(
                Section(
                    f"{doc_id}-s{i:02d}",
                    doc_id,
                    i,
                    topic.title(),
                    f"States the {topic}.",
                    f"Under agreement {tag}, the {topic} is {values[topic]}.",
                )
            )
        hidden = chosen[4]
        synonym = TOPICS[hidden]
        sections.append(
            Section(
                f"{doc_id}-s{INDIRECT_SECTION:02d}",
                doc_id,
                INDIRECT_SECTION,
                synonym.title(),
                f"Covers the {hidden} and the {synonym}.",
                f"In agreement {tag}, the {synonym} is {values[hidden]}.",
            )
        )
        sections.append(
            Section(
                f"{doc_id}-s{POINTER_SECTION:02d}",
                doc_id,
                POINTER_SECTION,
                "References",
                "Points to other sections.",
                f"For the {hidden} under agreement {tag}, "
                f"see the terms in Section {INDIRECT_SECTION}.",
            )
        )
        sections.append(
            Section(
                f"{doc_id}-s08",
                doc_id,
                8,
                "General provisions",
                "Boilerplate.",
                f"General provisions for agreement {tag} apply to both parties.",
            )
        )
        corpus[doc_id] = sections
    return corpus


def build_questions(
    corpus: dict[str, list[Section]], n: int = 500, hard_ratio: float = 0.06, seed: int = 11
) -> list[Question]:
    """Build n questions. hard_ratio is the share whose answer is behind a pointer section."""
    rng = random.Random(seed)
    doc_ids = sorted(corpus)
    questions: list[Question] = []
    for i in range(n):
        doc_id = rng.choice(doc_ids)
        tag = doc_tag(doc_id)
        sections = corpus[doc_id]
        hard = rng.random() < hard_ratio
        if hard:
            target = sections[INDIRECT_SECTION - 1]
            topic = next(t for t, s in TOPICS.items() if s.title() == target.title)
        else:
            target = rng.choice(sections[1:5])
            topic = target.title.lower()
        text = f"What is the {topic} under agreement {tag}?"
        questions.append(Question(f"q{i:05d}", doc_id, text, target.id, hard))
    return questions


def locate_answer(corpus: dict[str, list[Section]], question: str) -> str | None:
    """Find the section that answers a templated question.

    Only the simulated model uses this. It stands in for a real model that can tell whether the
    passages it was given actually answer the question.
    """
    for doc_id, sections in corpus.items():
        if doc_tag(doc_id) in question:
            for section in sections:
                if section.number in (2, 3, 4, 5) and section.title.lower() in question:
                    return section.id
            for topic, synonym in TOPICS.items():
                if topic in question:
                    for section in sections:
                        if section.number == INDIRECT_SECTION and synonym.title() == section.title:
                            return section.id
    return None
