"""A stand-in for the model service. It sleeps for a simulated time and answers by rule."""

from __future__ import annotations

import os

from fastapi import FastAPI
from pydantic import BaseModel

from bench.corpus import build_corpus
from bench.latency import LatencyModel
from bench.models import OutlineNode, Passage
from llm_stub.logic import SimulatedModel


class NodeIn(BaseModel):
    id: str
    title: str
    summary: str


class SelectRequest(BaseModel):
    question: str
    outline: list[NodeIn]


class PassageIn(BaseModel):
    id: str
    text: str


class GenerateRequest(BaseModel):
    question: str
    passages: list[PassageIn]


def create_app() -> FastAPI:
    env = os.getenv
    latency = LatencyModel(
        medians_ms={
            "select": float(env("SELECT_MS", "800")),
            "generate": float(env("GENERATE_MS", "600")),
        },
        sigma=float(env("LATENCY_SIGMA", "0.25")),
        scale=float(env("LATENCY_SCALE", "1.0")),
    )
    model = SimulatedModel(build_corpus(int(env("N_DOCS", "40"))))
    app = FastAPI(title="llm-stub")

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/select")
    async def select(request: SelectRequest) -> dict:
        await latency.pause("select")
        outline = [OutlineNode(n.id, n.title, n.summary) for n in request.outline]
        return {"node_ids": model.select_nodes(request.question, outline)}

    @app.post("/generate")
    async def generate(request: GenerateRequest) -> dict:
        await latency.pause("generate")
        passages = [Passage(p.id, "", p.text) for p in request.passages]
        result = model.generate(request.question, passages)
        return {"answer": result.answer, "supported": result.supported}

    return app


app = create_app()
