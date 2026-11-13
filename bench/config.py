"""Settings read from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from bench.latency import LatencyModel


@dataclass
class Settings:
    backend: str = "aws-local"  # "aws-local" uses OpenSearch and LocalStack, "memory" uses neither
    design: str = "outline"
    n_docs: int = 40
    opensearch_url: str = "http://opensearch:9200"
    aws_endpoint_url: str = "http://localstack:4566"
    llm_url: str = "http://llm-stub:8080"
    cache_ttl_seconds: float = 60.0
    latency: LatencyModel = field(default_factory=LatencyModel)

    @classmethod
    def from_env(cls) -> Settings:
        env = os.getenv
        latency = LatencyModel(
            medians_ms={
                "embed": float(env("EMBED_MS", "40")),
                "rerank": float(env("RERANK_MS", "80")),
                "select": float(env("SELECT_MS", "800")),
                "generate": float(env("GENERATE_MS", "600")),
            },
            sigma=float(env("LATENCY_SIGMA", "0.25")),
            scale=float(env("LATENCY_SCALE", "1.0")),
        )
        return cls(
            backend=env("BACKEND", "aws-local"),
            design=env("DESIGN", "outline"),
            n_docs=int(env("N_DOCS", "40")),
            opensearch_url=env("OPENSEARCH_URL", "http://opensearch:9200"),
            aws_endpoint_url=env("AWS_ENDPOINT_URL", "http://localstack:4566"),
            llm_url=env("LLM_URL", "http://llm-stub:8080"),
            cache_ttl_seconds=float(env("CACHE_TTL_SECONDS", "60")),
            latency=latency,
        )
