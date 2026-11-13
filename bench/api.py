"""HTTP API: ask a question, fetch a question set, scrape metrics."""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel

from bench.config import Settings
from bench.corpus import build_questions
from bench.service import Service
from bench.telemetry import setup_tracing

DESIGNS = ("vector", "outline")


class AskRequest(BaseModel):
    question: str
    doc_id: str
    design: str | None = None


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings.from_env()
    setup_tracing("rag-orchestrator")
    service = Service(settings)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        await service.prepare(settings.design)
        yield

    app = FastAPI(title="RAG latency bench", lifespan=lifespan)

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok", "design": settings.design, "backend": settings.backend}

    @app.post("/ask")
    async def ask(request: AskRequest) -> dict:
        design = request.design or settings.design
        if design not in DESIGNS:
            raise HTTPException(status_code=422, detail=f"design must be one of {DESIGNS}")
        if request.doc_id not in service.corpus:
            raise HTTPException(status_code=404, detail="unknown doc_id")
        try:
            await service.prepare(design)
        except Exception as exc:
            raise HTTPException(
                status_code=503, detail=f"{design} stores unavailable: {exc}"
            ) from exc
        answer = await service.pipeline().ask(design, request.question, request.doc_id)
        return {
            "design": answer.design,
            "answer": answer.answer,
            "supported": answer.supported,
            "rounds": answer.rounds,
            "latency_ms": round(answer.latency_ms, 1),
        }

    @app.get("/questions")
    async def questions(n: int = 500, hard_ratio: float = 0.10, seed: int = 11) -> list[dict]:
        built = build_questions(service.corpus, n=min(n, 5000), hard_ratio=hard_ratio, seed=seed)
        return [
            {"qid": q.qid, "doc_id": q.doc_id, "question": q.text, "hard": q.hard} for q in built
        ]

    @app.get("/metrics")
    async def metrics() -> Response:
        return Response(generate_latest(service.telemetry.registry), media_type=CONTENT_TYPE_LATEST)

    return app
