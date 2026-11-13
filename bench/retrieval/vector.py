"""Design A: embed the question, search a vector index, re-rank the hits."""

from __future__ import annotations

import asyncio
from typing import Protocol

from bench.corpus import Section
from bench.embedding import DIMS, cosine, embed
from bench.latency import LatencyModel
from bench.models import Passage
from bench.rerank import rerank
from bench.telemetry import Telemetry

INDEX_NAME = "sections"


class VectorStore(Protocol):
    async def index(self, sections: list[Section]) -> None: ...

    async def search(self, vector: list[float], k: int) -> list[Passage]: ...

    async def count(self) -> int: ...


class InMemoryVectorStore:
    """Exact search over a Python list. Used by tests and by the no-Docker mode."""

    def __init__(self) -> None:
        self._items: list[tuple[list[float], Passage]] = []

    async def index(self, sections: list[Section]) -> None:
        self._items = [(embed(s.text), Passage(s.id, s.doc_id, s.text)) for s in sections]

    async def search(self, vector: list[float], k: int) -> list[Passage]:
        scored = sorted(
            ((cosine(vector, v), p) for v, p in self._items), key=lambda t: t[0], reverse=True
        )
        return [Passage(p.id, p.doc_id, p.text, score) for score, p in scored[:k]]

    async def count(self) -> int:
        return len(self._items)


class OpenSearchVectorStore:
    """Approximate nearest-neighbour search in OpenSearch (k-NN plugin)."""

    def __init__(self, url: str) -> None:
        from opensearchpy import OpenSearch

        self._client = OpenSearch(hosts=[url], use_ssl=False, verify_certs=False, timeout=30)

    def _ensure_index(self) -> None:
        if self._client.indices.exists(index=INDEX_NAME):
            return
        self._client.indices.create(
            index=INDEX_NAME,
            body={
                "settings": {"index": {"knn": True}},
                "mappings": {
                    "properties": {
                        "doc_id": {"type": "keyword"},
                        "text": {"type": "text"},
                        "vector": {
                            "type": "knn_vector",
                            "dimension": DIMS,
                            "method": {
                                "name": "hnsw",
                                "space_type": "cosinesimil",
                                "engine": "lucene",
                            },
                        },
                    }
                },
            },
        )

    def _index_sync(self, sections: list[Section]) -> None:
        from opensearchpy import helpers

        self._ensure_index()
        actions = (
            {
                "_index": INDEX_NAME,
                "_id": s.id,
                "_source": {"doc_id": s.doc_id, "text": s.text, "vector": embed(s.text)},
            }
            for s in sections
        )
        helpers.bulk(self._client, actions, refresh=True)

    def _search_sync(self, vector: list[float], k: int) -> list[Passage]:
        body = {"size": k, "query": {"knn": {"vector": {"vector": vector, "k": k}}}}
        result = self._client.search(index=INDEX_NAME, body=body)
        return [
            Passage(h["_id"], h["_source"]["doc_id"], h["_source"]["text"], h["_score"])
            for h in result["hits"]["hits"]
        ]

    def _count_sync(self) -> int:
        if not self._client.indices.exists(index=INDEX_NAME):
            return 0
        return int(self._client.count(index=INDEX_NAME)["count"])

    async def index(self, sections: list[Section]) -> None:
        await asyncio.to_thread(self._index_sync, sections)

    async def search(self, vector: list[float], k: int) -> list[Passage]:
        return await asyncio.to_thread(self._search_sync, vector, k)

    async def count(self) -> int:
        return await asyncio.to_thread(self._count_sync)


class VectorRetriever:
    design = "vector"

    def __init__(self, store: VectorStore, latency: LatencyModel, telemetry: Telemetry) -> None:
        self._store = store
        self._latency = latency
        self._telemetry = telemetry

    async def retrieve(self, question: str, k: int) -> list[Passage]:
        with self._telemetry.step(self.design, "embed"):
            await self._latency.pause("embed")
            vector = embed(question)
        with self._telemetry.step(self.design, "search"):
            hits = await self._store.search(vector, k)
        with self._telemetry.step(self.design, "rerank"):
            # Re-ranking more candidates takes longer, so wider retries cost more.
            await self._latency.pause("rerank", factor=len(hits) / 30)
            return rerank(question, hits)
