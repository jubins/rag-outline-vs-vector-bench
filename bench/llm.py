"""The model interface. The bench can call a stub over HTTP or run the stub logic in-process."""

from __future__ import annotations

from typing import Protocol

import httpx

from bench.models import Generation, OutlineNode, Passage


class LLMClient(Protocol):
    async def select_nodes(self, question: str, outline: list[OutlineNode]) -> list[str]: ...

    async def generate(self, question: str, passages: list[Passage]) -> Generation: ...


class HttpLLM:
    """Calls the llm-stub service. Latency is simulated on the service side."""

    def __init__(self, base_url: str, timeout: float = 120.0) -> None:
        self._client = httpx.AsyncClient(base_url=base_url, timeout=timeout)

    async def select_nodes(self, question: str, outline: list[OutlineNode]) -> list[str]:
        payload = {
            "question": question,
            "outline": [{"id": n.id, "title": n.title, "summary": n.summary} for n in outline],
        }
        response = await self._client.post("/select", json=payload)
        response.raise_for_status()
        return list(response.json()["node_ids"])

    async def generate(self, question: str, passages: list[Passage]) -> Generation:
        payload = {
            "question": question,
            "passages": [{"id": p.id, "text": p.text} for p in passages],
        }
        response = await self._client.post("/generate", json=payload)
        response.raise_for_status()
        body = response.json()
        return Generation(body["answer"], body["supported"])

    async def aclose(self) -> None:
        await self._client.aclose()
